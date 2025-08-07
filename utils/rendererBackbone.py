# System Imports
import os
import math
import argparse
import time

# Common Libs
import numpy as np
from pathlib import Path
import cv2
import tkinter as tk
import threading
import queue

# Torch Imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

# 3rd party imports
from transformers import DPTForDepthEstimation, DPTImageProcessor
from tqdm import tqdm
import mediapipe as mp
from PIL import Image, ImageTk
from moviepy.editor import ImageSequenceClip

# From Codebase
from utils.mpi import mpi_rendering
from utils.mpi.homography_sampler import HomographySample
from utils.mpi.homography_sampler import HomographySample
from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto,
    render_novel_view,
)
from model.AdaMPI import MPIPredictor
from parameters import *


#=================================================
# Define the MPI Layers Processing Module Here
#=================================================
def processMPIs(src_imgs, mpi_all_src, disparity_all_src, k_src, k_tgt, save_path=None):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src

    return mpi_all_rgb_src, mpi_all_sigma_src, disparity_all_src, k_src_inv,k_tgt,homography_sampler



def cropFOV(image, original_fov, new_fov):
    image = np.array(image)
    if new_fov >= original_fov:
        raise ValueError("New FoV must be smaller than the original FoV")

    crop_ratio = new_fov / original_fov
    height, width = image.shape[:2]
    
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2

    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
    cropped_image = Image.fromarray(cropped_image)
    return cropped_image



def renderSingleFrame(mpi_all_rgb_src, mpi_all_sigma_src, disparity_all_src, cam_ext, k_src_inv, k_tgt, homography_sampler):
    frame = render_novel_view(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        cam_ext.to(device),
        k_src_inv,
        k_tgt,
        homography_sampler,
    )
    frame_np = frame[0].permute(1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    frame_np = np.clip(np.round(frame_np * 255), a_min=0, a_max=255).astype(np.uint8)
    im = Image.fromarray(frame_np)
    return im


class VideoCapture:
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()



def captureBackground(capture_device):
    frame_background = capture_device.read()
    img = cv2.cvtColor(frame_background, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil



def getImageTensor(pil_image, height, width, unsqueeze=True):
    t = transforms.Compose([transforms.CenterCrop((height, width)),transforms.ToTensor()])
    rgb = t(pil_image)

    if unsqueeze:
        rgb = rgb.unsqueeze(0)
    return rgb