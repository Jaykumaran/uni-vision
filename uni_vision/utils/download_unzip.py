import os
import requests
from zipfile import ZipFile, BadZipFile


def download_file(url, save_name):
    if not os.path.exists(save_name):
        
        with requests.get(url, allow_redirects=True) as r:
            if r.status_code == 200:
                with open(save_name, 'wb') as f:
                    f.write(r.content)
            
            else:
                print("Failed to download the file, status code:", r.status_code)
                

def unzip(zipfile = None, target_dir = None):
    try: 
        with ZipFile(zipfile , "r") as z:
            z.extractall(target_dir)
            print("Extracted all to: ", target_dir)
    except BadZipFile:
        print("Invalid file or error during extraction: Bad Zip File")
    except Exception as e:
        print("An error occured:", e)

