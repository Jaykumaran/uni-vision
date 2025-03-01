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
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
                ) or int(epoch) % self.checkpoint_conf.save_list:
                    checkpoint_names.append(f"checkpoint_{int(epoch)}")
                    
            checkpoint_paths = []
            for ckpt_name in checkpoint_names:
                checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))
            
            state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
            state_dict = exclude_params_matching_unix_pattern(
                patterns = self.checkpoint_conf.skip_saving_parameters, state_dict = state_dict
            )
            
            checkpoint = {
                "model": state_dict,
                "optimizer": self.optim.optimizer.state_dict(),
                "epoch": epoch,
                "loss": self.loss.state_dict(),
                "steps": self.steps,
                "time_elapsed": self.time_elapsed_meter.val,
                "best_meter_values": self.best_meter_values,
            }
            
            if self.optim_conf.amp.enabled:
                checkpoint["scaler"] = self.scaler.state_dict()
                
            # DDP checkpoints are only saved on rank 0 (all workers are identical)
            if self.distributed_rank != 0:
                return
            
            for checkpoint_path in checkpoint_paths:
                self._save_checkpoint(checkpoint, checkpoint_path)
                
        
    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """ 
        Save a checkpoint while guarding against the job being killed in 
        the middle of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire traininh since usually only the last checkpoint is kept per run).
        
        We first save the new checkpoint to a temp file (with a '.tmp' suffix) and
        move it to overwrite the old checkpoint path.
        """
        
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
                torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success
        
    
    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)
            
    
    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of first epoch
        assert_skipped_parameters_are_frozen(
            patterns = self.checkpoint_conf.skip_saving_parameters,
            model = self.model,
        )
        
        # Checking that parameters that won't be saved are intialized within the model
        # definition, unless `initialize_after_preemption` is explicitly set to `True`. 
        # If not, this is a bug, and after preemption, the `skip_saving_parameters` will 
        # have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns = self.checkpoint_conf.skip_saving_paramters,
            model = self.model,
            disabled = allow_init_skip_parameters,
        ):
            self._call_model_initializer()
    
    def _call_model_initializer(self):
        model_weight_intializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
    
        if model_weight_intializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_intializer}"
            )
            self.model = model_weight_intializer(model = self.model)
    
    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")
        
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_to_model(
            model = self.model,
            state_dict = checkpoint["model"],
            ignore_missing_keys = self.checkpoint_conf.skip_saving_parameters,
        )
    
        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict = True)
        self.epoch = checkpoint["epochs"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")
        
        if self.optim.conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        self.best_meter_values = checkpoint.get("best_meter_values", {})
        
        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])
    
    
    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1
    
    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)
        
        key = batch.dict_key # key for dataset
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"
        
        loss_log_str = os.path.join("Step_Losses", loss_str)
        
        # loss contains multiple sub-components we wish to log
        step_losses  ={}
        if isinstance(loss, dict):
            step_losses.update(
                {"Losses/{phase}_{key}_{k}": v for k,v in loss.items()}
            )
            
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )
            
        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )
            
        self.steps[phase] += 1
        
        ret_tuple = {loss_str: loss}, batch_size, step_losses
        
        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages = outputs,
                        find_metadatas = batch.metadata
                    )
        return ret_tuple
    
    
    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resume from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
                
                self.run_train()
                self.run_val()
            elif self.mode == "val":
                self.run_val()
            elif self.mode == "train_only":
                self.run_train()
                
    def _setup_dataloader(self):
        self.train_dataset = None
        self.val_dataset = None
        
        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))
        
        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)
                
    def run_train(self):
        
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch = int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch) # Logged only on rank 6
            
            # log train to text file
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")
                
                # Save checkpoint before validating
                self.save_checkpoint(self.epoch + 1)
                
                del dataloader
                gc.collect()
                
                # Run val, not running on last epoch since will run after the loop anyway
                if self.is_intermediate_val_epoch(self.epoch):
                    self.run().eval()
                    
                if self.distributed_rank == 0:
                    self.best_meter_values.update(self._get_trainer_state("train"))
                    with g_pathmgr.open(
                        os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                        "a",
                    ) as f:
                        f.write(json.dumps(self.best_meter_values) + "\n")
                
                self.epoch += 1
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1
    
    def run_val(self):
        if not self.val_dataset:
            return
        
        dataloader = self.val_dataset.get_loader(epoch = int(self.epoch))
        outs = self.val_epoch(dataloader, phase = Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch) # Logged only on rank 0
        
        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")
    
    
    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")
        
        iters_per_epoch = len(val_loader)
        
        curr_phases = [phase]
        curr_models = [self.model]
        
        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")
        
        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2f")) for name in loss_names]
        )
        
        extra_loss_mts = {}
        
        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()
                
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix = "Val Epoch: [{}]".format(self.epoch),
        )
        
        
        end = time.time()
        
        for data_iter, batch in enumerate(val_loader):
            
            # measure data loading time inside this for loop
            data_time.update(time.time() - end)
            
            batch = batch.to(self.device, non_blocking = True)
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled = [self.optim_conf.amp.enabled if self.optim_conf else False],
                    dtype = (
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase
                        )
                        
                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()
                        
                        loss_mts[loss_key].update(loss.item(), batch_size)
                        
                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            
                            extra_loss_mts[k].update(v.item(), batch_size)
                        
                            
                        
                # measured elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )
                
                
                if torch.cuda.is_available():
                    mem.update(reset_peak_usage = True)
                    
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)
                
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meter
                    for progress_meter in progress_meter:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[Phase.VAL]
                        )
                if data_iter % 10 == 0:
                    dist.barrier()
                
                self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
                self._log_timers(phase)
                for model in curr_models:
                    if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                        unwrap_ddp_if_wrapped(model).on_validation_epoch_end()
                
                out_dict = self._log_meters_and_save_best_ckpts(curr_phases)
                
                for k, v in loss_mts.items():
                    out_dict[k] = v.avg
                for k, v in extra_loss_mts.items():
                    out_dict[k] = v.avg
                
                for phase in curr_phases:
                    out_dict.update(self._get_trainer_state(phase))
                self._reset_meters(curr_phases)
                logging.info(f"Meters: {out_dict}")
                return out_dict
    
    
    
    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase]
        }   
        
    
    def train_epoch(self, train_loader):
        
        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f") 
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN  
        
        iters_per_epoch = len(train_loader)
        
        loss_names = []
        
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")
        
        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )       
        
        extra_loss_mts = {}     
        
        progress  = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix = "Train Epoch: [{}]".format(self.epoch)
        )
        
        # Model Training Loop
        self.model.train()
        end = time.time()
        
        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            
            data_time_meter.update(time.time() - end)
            data_times.append(time.time() - end)
            batch = batch.to(
                self.device, non_blocking = True
            )  # move tensors in a tensorclass
            
            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)
                
                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step = int(exact_epoch * iters_per_epoch)
                    )
                    
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at end, i.e. {self.where} of [0,1]."
                    )
                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                                
                            )
                # Clipping gradients and detecting divergent gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)     
                    self.gradient_clipper(model = self.model)  
                
                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank = self.distributed_rank, where = self.where
                    )
                    
                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)      
                self.scaler.update()  
                
                
                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()
                
                self.time_elpased_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )     
                
                mem_meter.update(reset_peak_usage = True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)
                
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )
                # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e
            
            self.est_epoch_time(Phase.TRAIN) = batch_time_meter.avg * iters_per_epoch
            self._log_timers(Phase.TRAIN)
            self._log_sync_data_times(Phase.TRAIN, data_times)
            
            out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])
            
            for k, v in loss_mts.items():
                out_dict[k] = v.avg
            
            for k, v in extra_loss_mts.items():
                out_dict[k] = v.avg
            
            out_dict.update(self._get_trainer_state(phase))
            logging.info(f"Losses and meters: {out_dict}")
            self._reset_meters([phase])
            return out_dict
    
    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times).tolist())
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )
        