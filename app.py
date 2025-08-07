import gradio as gr
import torch
import numpy as np
import cv2
import tempfile
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model_Small import MMPI as MMPI_S
from model_Medium import MMPI as MMPI_M
from model_Large import MMPI as MMPI_L
import helperFunctions as helper
import socket
import parameters as params
from utils.mpi.homography_sampler import HomographySample
from utils.utils import (
    render_novel_view,
)

# Checkpoint locations for all models
MODEL_S_LOCATION = "./checkpoint/checkpoint_RT_MPI_Small.pth"
MODEL_M_LOCATION = "./checkpoint/checkpoint_RT_MPI_Medium.pth"
MODEL_L_LOCATION = "./checkpoint/checkpoint_RT_MPI_Large.pth"

DEVICE = "cuda:0"

def getPositionVector(x, y, z, pose):
    pose[0,0,3] = x
    pose[0,1,3] = y
    pose[0,2,3] = z
    return pose

def generateCircularTrajectory(radius, num_frames):
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    return [[radius * np.cos(angle), radius * np.sin(angle), 0] for angle in angles]

def generateWiggleTrajectory(radius, num_frames):
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    return [[radius * np.cos(angle), 0, radius * np.sin(angle)] for angle in angles]

def create_video_from_memory(frames, fps=60):
    if not frames:
        return None
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return temp_video.name

def process_image(img, video_type, radius, num_frames, num_loops, model_type, resolution):
    # Parse resolution string
    height, width = map(int, resolution.lower().split("x"))

    # Select model class and checkpoint
    if model_type == "Small":
        model_class = MMPI_S
        checkpoint = MODEL_S_LOCATION
    elif model_type == "Medium":
        model_class = MMPI_M
        checkpoint = MODEL_M_LOCATION
    else:
        model_class = MMPI_L
        checkpoint = MODEL_L_LOCATION

    # Load model
    model = model_class(total_image_input=params.params_number_input, height=height, width=width)
    model = helper.load_Checkpoint(checkpoint, model, load_cpu=True)
    model.to(DEVICE)
    model.eval()

    min_side = min(img.width, img.height)
    left = (img.width - min_side) // 2
    top = (img.height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    img = img.crop((left, top, right, bottom))
    
    if video_type == "Circle":
        trajectory = generateCircularTrajectory(radius, num_frames)
    elif video_type == "Swing":
        trajectory = generateWiggleTrajectory(radius, num_frames)
    else:
        trajectory = generateCircularTrajectory(radius, num_frames)

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    img_input = transform(img).to(DEVICE).unsqueeze(0)

    grid = params.get_disparity_all_src().unsqueeze(0).to(DEVICE)
    k_tgt = torch.tensor([
        [0.58, 0, 0.5],
        [0, 0.58, 0.5],
        [0, 0, 1]]).to(DEVICE)
    k_tgt[0, :] *= height
    k_tgt[1, :] *= width
    k_tgt = k_tgt.unsqueeze(0)
    k_src_inv = torch.inverse(k_tgt)
    pose = torch.eye(4).to(DEVICE).unsqueeze(0)

    homography_sampler = HomographySample(height, width, DEVICE)

    with torch.no_grad():
        rgb_layers, sigma_layers = model.get_layers(img_input, height=height, width=width)
        
        predicted_depth = model.get_depth(img_input)
        predicted_depth = (predicted_depth-predicted_depth.min())/(predicted_depth.max()-predicted_depth.min())
        img_predicted_depth = predicted_depth.squeeze().cpu().detach().numpy()
        img_predicted_depth_colored = plt.get_cmap('inferno')(img_predicted_depth / np.max(img_predicted_depth))[:, :, :3]
        img_predicted_depth_colored = (img_predicted_depth_colored * 255).astype(np.uint8)
        img_predicted_depth_colored = Image.fromarray(img_predicted_depth_colored)
        
        layer_depth = model.get_layer_depth(img_input, grid)
        img_layer_depth = layer_depth.squeeze().cpu().detach().numpy()
        img_layer_depth_colored = plt.get_cmap('inferno')(img_layer_depth / np.max(img_layer_depth))[:, :, :3]
        img_layer_depth_colored = (img_layer_depth_colored * 255).astype(np.uint8)
        img_layer_depth_colored = Image.fromarray(img_layer_depth_colored)

    single_loop_frames = []
    for idx, pose_coords in enumerate(trajectory):
        #print(f"  - Rendering frame {idx + 1}/{len(trajectory)}", end="\r")
        with torch.no_grad():
            target_pose = getPositionVector(pose_coords[0], pose_coords[1], pose_coords[2], pose)
            output_img = render_novel_view(rgb_layers,
                                         sigma_layers,
                                         grid,
                                         target_pose,
                                         k_src_inv,
                                         k_tgt,
                                         homography_sampler)

        img_np = output_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        single_loop_frames.append(img_bgr)

    final_frames = single_loop_frames * int(num_loops)
    
    video_path = create_video_from_memory(final_frames)
    #print("Video generation complete!")
    
    return video_path, img_predicted_depth_colored, img_layer_depth_colored

with gr.Blocks(title="RT-MPINet", theme="default") as demo:
    gr.Markdown(
    """
    ## Parallax Video Generator via Real-Time Multiplane Image Network (RT-MPINet)
    We use a smaller 256x256 model for faster inference on CPU instances.

    #### Notes:
    1. Use a higher number of frames (>80) and loops (>4) to get a smoother video.
    2. The default uses 60 frames and 4 camera loops for fast video generation.
    3. We have 3 models available (larger the model, slower the inference):
        * **Small:** 6.6 Million parameters
        * **Medium:** 69 Million parameters
        * **Large:** 288 Million parameters
    """)
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image")
        video_type = gr.Dropdown(["Circle", "Swing"], label="Video Type", value="Swing")
        with gr.Column():    
            with gr.Accordion("Advanced Settings", open=False):
                radius = gr.Slider(0.001, 0.1, value=0.05, label="Radius (for Circle/Swing)")
                num_frames = gr.Slider(10, 180, value=60, step=1, label="Frames per Loop")
                num_loops = gr.Slider(1, 10, value=4, step=1, label="Number of Loops")
                with gr.Column():
                    model_type_dropdown = gr.Dropdown(["Small", "Medium", "Large"], label="Model Type", value="Medium")
                    resolution_dropdown = gr.Dropdown(["256x256", "384x384", "512x512"], label="Input Resolution", value="384x384")
        generate_btn = gr.Button("Generate Video", variant="primary")

    with gr.Row():
        video_output = gr.Video(label="Generated Video")
        depth_output = gr.Image(label="Depth Map - From Depth Decoder")
        layer_depth_output = gr.Image(label="Layer Depth Map - From MPI Layers")

    def toggle_custom_path(video_type_selection):
        is_custom = (video_type_selection == "Custom")
        return gr.update(visible=is_custom)

    generate_btn.click(fn=process_image,
                      inputs=[img_input, video_type, radius, num_frames, num_loops, model_type_dropdown, resolution_dropdown],
                      outputs=[video_output, depth_output, layer_depth_output])

demo.launch()
