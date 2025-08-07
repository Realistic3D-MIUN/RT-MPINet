# System
import os
from pathlib import Path
import argparse
import time

# Basic Libs
import math
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Main Libs
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# From Codebase
from parameters import *
from model_Large import MMPI as MMPI_L
from model_Medium import MMPI as MMPI_M
from model_Small import MMPI as MMPI_S
import helperFunctions as helper
import parameters as params
import time
from utils.utils import (
    render_novel_view,
)
from utils.mpi.homography_sampler import HomographySample
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height', type=int, default=params.params_height)
parser.add_argument('--width', type=int, default=params.params_width)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mouse_sensitivity', type=int, default=5000, help="Set the mouse sensitivity for small baseline network to limit going over the baseline.")
parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/checkpoint_RT_MPI_Medium.pth")
parser.add_argument('--input_image', type=str, default="./samples/16.jpeg", help="Location of input imnage for which novel views are needed to be synthesized")
parser.add_argument('--model_type', type=str, default="medium", help="Type of model to load, by default we use medium model")
opt, _ = parser.parse_known_args()

root = tk.Tk()
root.title("RT-MPINet Renderer")
SCALE = opt.scale

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

grid = params.get_disparity_all_src().unsqueeze(0).to(DEVICE)
k_tgt = torch.tensor([
[0.58, 0, 0.5],
[0, 0.58, 0.5],
[0, 0, 1]]).to(DEVICE)
k_tgt[0, :] *= opt.width
k_tgt[1, :] *= opt.height
k_tgt = k_tgt.unsqueeze(0)
k_src_inv = torch.inverse(k_tgt)
pose = torch.eye(4).to(DEVICE).unsqueeze(0)
last_mouse_pos = [0, 0]

RAW_DEPTH_FROM_MODEL = None
DEPTH_FROM_LAYERS = None

homography_sampler = HomographySample(opt.height, opt.width, DEVICE)

def stack_images_side_by_side(im1, im1_d, im1_c):
    if im1.height != im1_d.height:
        raise ValueError("The images must have the same height to stack them side by side.")
    
    combined_width = im1.width + im1_d.width + im1_c.width
    combined_height = im1.height
    combined_image = Image.new("RGB", (combined_width, combined_height))

    combined_image.paste(im1, (0, 0))  
    combined_image.paste(im1_d, (im1.width, 0))  
    combined_image.paste(im1_c, (im1.width+im1_d.width, 0))  
    
    return combined_image


def renderSingleFrame(pose):
    global RAW_DEPTH_FROM_MODEL
    global DEPTH_FROM_LAYERS
    start_time = time.time()
    merged_feature_rgb, merged_feature_sigma = model.get_rgb_sigma(img_input, opt.height, opt.width)
    #torch.cuda.synchronize() #uncomment this for accurate estimated time when using GPU
    end_time = time.time()
    print("Time: ",end_time-start_time)
    
    predicted_img = render_novel_view(merged_feature_rgb,
                            merged_feature_sigma,
                            grid,
                            pose,
                            k_src_inv,
                            k_tgt,
                            homography_sampler)

    if RAW_DEPTH_FROM_MODEL == None:
        predicted_depth = model.get_depth(img_input)
        predicted_depth = (predicted_depth-predicted_depth.min())/(predicted_depth.max()-predicted_depth.min())
        img_predicted_depth = predicted_depth.squeeze().cpu().detach().numpy()
        img_predicted_depth_colored = plt.get_cmap('inferno')(img_predicted_depth / np.max(img_predicted_depth))[:, :, :3]
        img_predicted_depth_colored = (img_predicted_depth_colored * 255).astype(np.uint8)
        img_predicted_depth_colored = Image.fromarray(img_predicted_depth_colored)
        RAW_DEPTH_FROM_MODEL = img_predicted_depth_colored

    if DEPTH_FROM_LAYERS == None:
        predicted_depth = model.get_layer_depth(img_input, grid, opt.height, opt.width)
        img_predicted_depth = predicted_depth.squeeze().cpu().detach().numpy()
        img_predicted_depth_colored = plt.get_cmap('inferno')(img_predicted_depth / np.max(img_predicted_depth))[:, :, :3]
        img_predicted_depth_colored = (img_predicted_depth_colored * 255).astype(np.uint8)
        img_predicted_depth_colored = Image.fromarray(img_predicted_depth_colored)
        DEPTH_FROM_LAYERS = img_predicted_depth_colored

    im1 = torchvision.transforms.functional.to_pil_image(predicted_img[0])

    combined_img = stack_images_side_by_side(im1,RAW_DEPTH_FROM_MODEL, DEPTH_FROM_LAYERS)
    #combined_img.save("Out_frame.png")
    return combined_img

def update_image(pose):
    img = renderSingleFrame(pose)
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk


initial_img = renderSingleFrame(pose)
print("Got initial image: ",type(initial_img))
img_tk = ImageTk.PhotoImage(initial_img)

label = tk.Label(root, image=img_tk)
label.image = img_tk
label.pack()

def on_mouse_drag_translate(event):
    x_offset = (event.x - root.winfo_width() / 2) / opt.mouse_sensitivity
    y_offset = (event.y - root.winfo_height() / 2) / opt.mouse_sensitivity
    pose[0,0,3] = x_offset
    pose[0,1,3] = y_offset
    pose[0,2,3] = 0
    update_image(pose)

def on_mouse_drag_rotate(event):
    global last_mouse_pos
    dx = event.x - last_mouse_pos[0]
    dy = event.y - last_mouse_pos[1]
    last_mouse_pos = [event.x, event.y]
    yaw = dx / opt.mouse_sensitivity
    pitch = dy / opt.mouse_sensitivity
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    R_yaw = torch.tensor([
        [cos_yaw, 0, sin_yaw],
        [0, 1, 0],
        [-sin_yaw, 0, cos_yaw]
    ], dtype=torch.float32, device=DEVICE)

    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    R_pitch = torch.tensor([
        [1, 0, 0],
        [0, cos_pitch, -sin_pitch],
        [0, sin_pitch, cos_pitch]
    ], dtype=torch.float32, device=DEVICE)
    R_delta = torch.matmul(R_yaw, R_pitch)
    R_current = pose[0, :3, :3]
    R_new = torch.matmul(R_delta, R_current)
    pose[0, :3, :3] = R_new
    update_image(pose)

    
def on_mouse_press(event):
    global last_mouse_pos
    last_mouse_pos = [event.x, event.y]

root.bind('<B1-Motion>', on_mouse_drag_translate)
root.bind('<B3-Motion>', on_mouse_drag_rotate)
root.bind('<ButtonPress-3>', on_mouse_press)

root.mainloop()

