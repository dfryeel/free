# ATDMNet
This repository contains the implementation of a deep learning model using PyTorch, designed for feature extraction and enhancement through convolutional layers, attention mechanisms, and pixel shuffle upsampling. The model integrates a custom encoder with several convolutional and attention layers, specifically targeting computer vision tasks.
## Overview
![image](network1.png)

The core architecture is composed of:
- **Multi-Head Agent Attention (MHA)**: A bidirectional attention mechanism augmented by agent tokens and positional biases to enhance local feature discriminability while modeling cross-region dependencies.
- **Top-k Dynamic Mask (TDM)**: A computationally efficient module that employs dynamic multi-scale masking and top-k selection to prioritize salient features across varying resolutions, reducing redundancy in attention computation.
- **Deeply Supervised Hybrid Loss**: A boundary-aware optimization framework combining BCE and IoU losses, guided by high-level semantic cues to refine edge details and suppress false positives.

## Prerequisites
- Python >= 3.6
- PyTorch 1.8+ (for CUDA and efficient GPU utilization)
- Other required dependencies can be found in `requirements.txt`.
To install the required dependencies, run:
```bash
pip install -r requirements.txt 
```
- Datasets: The dataset used for training and validation should be placed in the `data` directory.Download the dataset from `https://pan.baidu.com/s/1Jnkhz6B2p08ItjL6zmqJpg?pwd=free`.

## File Structure
- train.py: Main training script. This is the entry point for training and validation.
- Net_v1.py: Contains the Network class which defines the model architecture.
- Res2Net.py: Contains the Res2Net model definition used as the feature extractor.
- pvtv2.py: Defines the PVT-v2 model for the backbone encoder. Download the pretrained weights for PVT-v2 at `https://pan.baidu.com/s/1DrKqG5aXkOGKcmqdQuUoYQ?pwd=0808`.

## Training
To start training the model, run the` train.py` script with the necessary arguments.
### Example Command:
`python train.py --epoch 60 --lr 5e-5 --batchsize 8 --train_root /path/to/train_data/ --val_root /path/to/val_data/ --save_path /path/to/save_model/
`
### Command for Validation
The validation happens automatically at the end of each epoch. However, if you want to run validation manually:

`python train.py --epoch 60 --load /path/to/saved_model.pth --gpu_id '0' --val_root /path/to/val_data/
`
## Visualization
The training script will save visualizations of the results, including:
- RGB input images. 
- Ground truth segmentation maps. 
- Output feature maps. 
- Edge ground truth and predicted edges.

These visualizations are saved in the `tmp_path` directory during training for inspection.Download the visualization results from `https://pan.baidu.com/s/14xXwPYv9yushUnv05B-Kwg?pwd=free`.

## License
This code is open-source and is distributed under the MIT License.
### Notes:
1. Adjust the file paths (`train_root`, `val_root`, `save_path`) based on where you store your dataset and models.
2. If you are using multiple GPUs, make sure that the `gpu_id` argument is set correctly (e.g., `--gpu_id '0,1'` for using two GPUs).

