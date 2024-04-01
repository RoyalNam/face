import json
import logging
import sys
import subprocess
import os
import time

import torch

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FacialLandmarkDetectionLogger")


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def collate_fn(batch):
    return tuple(zip(*batch))


# Function to start TensorBoard
def start_tensorboard(log_dir='logs'):
    cmd = ['tensorboard', '--logdir', log_dir]
    tb_process = subprocess.Popen(cmd)
    time.sleep(2)
    return tb_process


# Function to stop TensorBoard
def stop_tensorboard(tb_process):
    tb_process.terminate()
    time.sleep(2)
    if tb_process.poll() is None:
        tb_process.kill()


def save_model(model, model_name):
    torch.save(model, model_name)
