# https://github.com/facebookresearch/sam2/blob/main/training/trainer.py

import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr  # an instance of PathManager to handle lfs -> from fbresearch 

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_to_model,
    with_check_parameter_frozen,   
)


from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank
from training.utils.logger import Logger, setup_logging

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend
)



CORE_LOSS_KEY = "core_loss"

def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallen.DistributionDataParallel):
        return model.module
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype = "float16"
    
    
@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None
    
    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    

@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # If not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # If not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None
    

@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory = list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list) # Will have different list instances ; to avoid unexpected behavior --> As list are mutables instances will share the same list reference; Bad Practice --> skip_saving_parameters: List[str] = [] 
    initialize_after_preemption: Optional[bool] = None
    # If not None, training will be resumed from this checkpoint
    resume_from: Optional[int] = None
    
    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self
    
@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False
    

class Trainer:
    """
    Trainer supporting the DDP training strategies
    """
    
    EPSILON = 1e-8
    
    def __init__(
        self,
        *, # the order of these args can change at any time, so they are keyword only,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        acclerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()
        
        
        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0
        
        self._infer_distributed_backend_if_none(distributed, acclerator)
        
        self._setup_device(acclerator)
        
        self._setup_torch_dist_and_backend(cuda, distributed)
        
        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir = self.logging_conf.log_dir,
            rank = self.rank,
            log_level_primary = self.logging_conf.log_level_primary,
            log_level_secondary = self.logging_conf.log_level_secondary
        )
        
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()
        
        assert(
            is_dist_avail_and_initialized
        ), "Torch distributed needs to be initialized before calling the trainer."
        
        
        self._setup_components() # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()
        
        
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")
        
        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the `resume_from` checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there.
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()
            
        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, acclerator)
        barrier()   # in distributed systems, it is a synchronization mechanism used to ensure all processes reach same point in execution before proceeding further.
        
    
    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)
     
     
     
    def _get_meters(self, phase_filters = None):
        if self.filters is None:
            return {}
        meters =   {} 
        
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{"name"}"] = meter
        return meters
    
    
    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"
        
    
    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        
    
    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.matmul_allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            
            
        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )
        
        
    def setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")
        
    
    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        
        assert isinstance(self.model, torch.nn.Module)
        
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )
        
        if distributed_conf.comms_dtype is not None: # noqa
            from torch.distributed.algorithms import ddp_comm_hooks
           
            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
               hook = ddp_comm_hooks.default_hooks.bf16_compress_book
               logging.info("Enabling bfloat16 grad communication")
            else:
               hook = ddp_comm_hooks.default_hooks.fp16_compress_book
               logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)
            
            
    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}"
        )
        
        self.model.to(self.device)
        
        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}"
        )
           
    
    def save_checkpoint(self, epoch, checkpoint_names = None): 
            
                
                
            
        
        
        
    