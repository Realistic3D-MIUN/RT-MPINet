import os
import torch

params_height = 256
params_width = 256
params_m = 32
params_number_input = 1
params_step_size = 2
params_gamma = 0.2
params_num_planes = 32

TRAIN_LOCATION = "./lf_train.txt"
VALIDATION_LOCATION = "./lf_validate.txt"
TEST_LOCATION = "./lf_test.txt"
LOG_FILE_LOCATION = "./logs/training_log_0.txt"
CHECKPOINT_LOCATION = "./checkpoint/"
RESUME_CHECKPOINT_LOCATION = "./checkpoint/checkpoint_best.pth"
START_CHECKPOINT_LOCATION = "./checkpoint/checkpoint_init.pth"
DEVICE = "cuda:0"

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 150
START_EPOCH = 0
PRINT_INTERVAL = 20
T_max = 150

os.makedirs("./logs",exist_ok=True)
os.makedirs("./checkpoint",exist_ok=True)
os.makedirs("./output",exist_ok=True)

def uniform_planes(a: float, b: float, n: int) -> torch.Tensor:
    """
    Return n values uniformly spaced *within* (a, b),
    i.e. excluding the exact endpoints a and b.
    """
    step = (b - a) / (n + 1)
    # torch.arange(1, n+1) gives [1,2,...,n]
    return a + step * torch.arange(1, n + 1, dtype=torch.float32)

def get_disparity_all_src():
    d1 = uniform_planes(0.0, 0.4, 20)
    d2 = uniform_planes(0.4, 1.0, 12)
    disparities = torch.cat([d1, d2], dim=0)
    return disparities



