# System
import argparse
import time

# Basic Libs
import tkinter as tk
from PIL import ImageTk
import time
import math

# Main Libs
import torch
import torchvision

# From Codebase
from parameters import *
import parameters as params
from utils.mpi.homography_sampler import HomographySample
from utils.utils import (
    render_novel_view,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--height', type=int, default=params.params_height)
parser.add_argument('--width', type=int, default=params.params_width)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--mouse_sensitivity', type=int, default=5000, help="Set the mouse sensitivity for small baseline network to limit going over the baseline.")
parser.add_argument('--layer_path', type=str, default="./processedLayers/")
opt, _ = parser.parse_known_args()

# Initialize the main application window
root = tk.Tk()
root.title("PVDNet Renderer")
SCALE = opt.scale

# Define the grid for depth planes, k_src, k_inv, and pose matrix
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


# Load RGB and Sigma Planes
rgb_layers = torch.load(opt.layer_path+"rgb_layers.pt").to(params.DEVICE)
sigma_layers = torch.load(opt.layer_path+"sigma_layers.pt").to(params.DEVICE)

# Define homography sampler
homography_sampler = HomographySample(opt.height, opt.width, params.DEVICE)


def renderSingleFrame(pose):
    start_time = time.time()
    predicted_img = render_novel_view(rgb_layers, sigma_layers,
                      grid, pose, k_src_inv, k_tgt,
                      homography_sampler,)
    #torch.cuda.synchronize() #uncomment this for accurate estimated time when using GPU
    end_time = time.time()
    print("Time: ",end_time-start_time)
    im = torchvision.transforms.functional.to_pil_image(predicted_img[0])
    return im

# Function to update the image based on new camera pose
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

