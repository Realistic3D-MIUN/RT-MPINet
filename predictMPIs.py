# System
import os
from pathlib import Path
import argparse

# Basic Libs
from PIL import Image

# Main Libs
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

# From Codebase
from parameters import *
from model_Large import MMPI as MMPI_L
from model_Medium import MMPI as MMPI_M
from model_Small import MMPI as MMPI_S
import helperFunctions as helper
import parameters as params

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height', type=int, default=params.params_height)
parser.add_argument('--width', type=int, default=params.params_width)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mouse_sensitivity', type=int, default=5000, help="Set the mouse sensitivity for small baseline network to limit going over the baseline.")
parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/checkpoint_RT_MPI_Medium.pth")
parser.add_argument('--input_image', type=str, default="./samples/16.jpeg", help="Location of input imnage for which MPI Layers needs to be predicted")
parser.add_argument('--model_type', type=str, default="medium", help="Type of model to load, by default we use medium model")
parser.add_argument('--save_dir', type=str, default="./processedLayers/", help="Location of input imnage for which novel views are needed to be synthesized")
opt, _ = parser.parse_known_args()

os.makedirs(opt.save_dir,exist_ok=True)

# Load model
if opt.model_type == "small":
    model = MMPI_S(total_image_input=params_number_input, height=opt.height, width=opt.width)
elif opt.model_type == "large":
    model = MMPI_L(total_image_input=params_number_input, height=opt.height, width=opt.width)
else:
    model = MMPI_M(total_image_input=params_number_input, height=opt.height, width=opt.width)

model = helper.load_Checkpoint(opt.checkpoint_path, model, load_cpu=True)
model.to(DEVICE)
model.eval()
print("Status: Model Loaded!")

transform = transforms.Compose([transforms.Resize((opt.height, opt.width)),
                                    transforms.ToTensor()])

img_input = Image.open(opt.input_image).convert('RGB')
img_input = transform(img_input).unsqueeze(0).to(DEVICE)
print("Status: Image Loaded!")

rgb_layers, sigma_layers = model.get_layers(img_input, opt.height, opt.width)

rgb_layers = rgb_layers.detach().to("cpu")
sigma_layers = sigma_layers.detach().to("cpu")

torch.save(rgb_layers, opt.save_dir+"rgb_layers.pt")
torch.save(sigma_layers, opt.save_dir+"sigma_layers.pt")

print("Status: MPIs saved at "+opt.save_dir)
