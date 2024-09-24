# ComfyUI-IG-Motion-I2V
<a href='https://xiaoyushi97.github.io/Motion-I2V/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementation of [Motion-I2V](https://xiaoyushi97.github.io/Motion-I2V/)
This is currently a diffusers wrapper with code adapted from [https://github.com/G-U-N/Motion-I2V](https://github.com/G-U-N/Motion-I2V)
## Updates
- [2024/9/24] 🔥 First Release
- [2024/9/23] 🔥 Interactive Motion Painter UI for ComfyUI
- [2024/9/20] 🔥 Added basic IP Adapter integration
- [2024/9/16] 🔥 Uodated model code to be compatible with Comfy's diffusers version

## TODO
- Convert the code to be Comfy Native
- Reduce VRAM usage
- More motion controls
- Train longer context models

## Nodes
- ***MI2V Flow Predictor*** takes as input a first frame and option motion prompt, mask and vectors. Outputs a predicted optical flow for a 16 frame animation with the input image as the first frame. You can view a preview of the motion where the colors correspond to movement in 2 Dimensions ![colors](assets/colors.jpg)
- ***MI2V Flow Animator*** takes the predicted flow and a starting image and generates a 16 frame animation based on these
- ***MI2V Motion Painter*** allows you to draw motion vectors onto an image to be used by MI2V Flow Predictor
- ***MI2V Pause*** allows you to pause the execution of the workflow. Useful for loading a resized image into MI2V Flow Predictor or checking you like the predicted motion before committing to further animation

## Instructions
![arch](assets/screenshot1.png)
![arch](assets/screenshot2.png)

## Credits
- [Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling](https://arxiv.org/abs/2401.15977)
by *Xiaoyu Shi<sup>1\*</sup>, Zhaoyang Huang<sup>1\*</sup>, Fu-Yun Wang<sup>1\*</sup>, Weikang Bian<sup>1\*</sup>, Dasong Li <sup>1</sup>, Yi Zhang<sup>1</sup>, Manyuan Zhang<sup>1</sup>, Ka Chun Cheung<sup>2</sup>, Simon See<sup>2</sup>, Hongwei Qin<sup>3</sup>, Jifeng Dai<sup>4</sup>, Hongsheng Li<sup>1</sup>* *<sup>1</sup>CUHK-MMLab   <sup>2</sup>NVIDIA   <sup>3</sup>SenseTime  <sup>4</sup>  Tsinghua University*
</div>

- Motion Painter node was adapted from code in this node [https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet)
