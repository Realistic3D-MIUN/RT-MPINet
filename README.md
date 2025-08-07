<div align="center">
  <a href="#"><img src='https://img.shields.io/badge/-Paper-00629B?style=flat&logo=ieee&logoColor=white' alt='arXiv'></a>
  <a href='https://realistic3d-miun.github.io/Research/RT_MPINet/index.html'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/spaces/3ZadeSSG/RT-MPINet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo_(RT_MPINet)-blue'></a>
</div>

# RT-MPINet
#### Real-Time View Synthesis with Multiplane Image Network using Multimodal Supervision (RT-MPINet)

We present a real-time multiplane image (MPI) network. Unlike existing MPI based approaches that often rely on a separate depth estimation network to guide the network for estimating MPI parameters, our method directly predicts these parameters from a single RGB image. To guide the network we present a multimodal training strategy utilizing joint supervision from view synthesis and depth estimation losses. More details can be found in the paper.

**Please head to the [Project Page](https://realistic3d-miun.github.io/Research/RT_MPINet/index.html) to see supplementary materials**

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Realistic3D-MIUN/RT-MPINet
   cd RT-MPINet
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install PyTorch3D after the general libs have been installed
   ```bash
   pip install "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@89653419d0973396f3eff1a381ba09a07fffc2ed"
   ```


## Checkpoints (Best Checkpoints Will Be Updated Soon)
Pretrained model checkpoints should be placed in the `checkpoint/` directory. Example filenames:
- `checkpoint_RT_MPI_Small.pth`
- `checkpoint_RT_MPI_Medium.pth`
- `checkpoint_RT_MPI_Large.pth`

| Model           | Size   | Parameters | Checkpoint |
|-----------------|--------|------------|----------------|
| Small           | 26 MB  | 6.6 Million| [Download](https://huggingface.co/3ZadeSSG/RT-MPINet/resolve/main/checkpoint_RT_MPI_Small.pth) |
| Medium (Default)| 278 MB | 69 Million | [Download](https://huggingface.co/3ZadeSSG/RT-MPINet/resolve/main/checkpoint_RT_MPI_Medium.pth) |
| Large           | 1.2 GB | 288 Million| [Download](https://huggingface.co/3ZadeSSG/RT-MPINet/resolve/main/checkpoint_RT_MPI_Large.pth) |

## Usage

### 1. Live Rendering Demo
You can load any image and run the model inference each time the camera position is changed. This will be limited to the inference speed on your GPU.
   ```bash
   python renderLiveWithMouseControl.py \
   --input_image <path_to_image> \
   --model_type <small|medium|large> \
   --checkpoint_path <path_to_checkpoint> \
   --height <height> \
   --width <width>
   ```
Example:
```bash
   python renderLiveWithMouseControl.py \
   --input_image ./samples/moon.jpg \
   --model_type medium \
   --checkpoint_path ./checkpoint/checkpoint_RT_MPI_Medium.pth \
   --height 256 \
   --width 256
```

### 2. Inference: Predict MPIs from an image and render afterwards
The predicted MPIs can be used for offline rendering, which is much faster as the model isn't being queried each time camera changes. This requires

* First predicting the MPIs
   ```bash
   python predictMPIs.py \
   --input_image <path_to_image> \
   --model_type <small|medium|large> \
   --checkpoint_path <path_to_checkpoint> \
   --save_dir <output_dir> \
   --height <height> \
   --width <width>
   ```

* Second the MPIs are loaded and views are rendered without invoking the model using
   ```bash
   python renderPreProcessedWithMouseControl.py \
   --layer_path <output_dir> \
   --height <height> \
   --width <width>
   ```

Example:
   ```bash
   python predictMPIs.py \
   --input_image ./samples/moon.jpg \
   --model_type medium \
   --checkpoint_path ./checkpoint/checkpoint_RT_MPI_Medium.pth \
   --save_dir ./processedLayers/ \
   --height 384 \
   --width 384
   ```

   ```bash
   python renderPreProcessedWithMouseControl.py \
   --layer_path ./processedLayers/ \
   --height 384 \
   --width 384
   ```


### 3. Web Demo (Gradio)
You can run the local demo of the Huggingface app to utilize your own GPU for faster inference using
   ```bash
   python app.py
   ```


## Supported Resolutions
We have tested our model with following resolutions:
- 256x256
- 384x384
- 512x512
- 256x384
- 384x512

**Note:** If using non square aspect ratio, you need to modify the torch transform to account for changes.

## License


## Acknowledgements
- We thank the authors of [AdaMPI](https://github.com/yxuhan/AdaMPI) for their implementation of the homography renderer which has been used in this codebase under `./utils` directory
- We tank the author of [Deepview renderer](https://github.com/Findeton/deepview) template, which was used in our project page.


## Citation
If you use our work please use following citation:
```
@inproceedings{gond2025rtmpi,
  title={Real-Time View Synthesis with Multiplane Image Network using Multimodal Supervision},
  author={Gond, Manu and Shamshirgarha, Mohammadreza and Zerman, Emin and Knorr, Sebastian and Sj{\"o}str{\"o}m, M{\aa}rten},
  booktitle={2025 IEEE 27th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={},
  year={2025},
  organization={IEEE}
}
```

