import os
from einops import rearrange
import torch
import json
import torchvision.transforms as transforms

from diffusers.schedulers import (
        DDIMScheduler,
        LCMScheduler
    )

import comfy.utils
import model_management 
import folder_paths
from ..common.tree import *
from ..common.constants import *
from ..common.utils import rename_state_dict_keys, print_loading_issues, flow_to_color

from ..flowgen.pipelines.pipeline_flow_gen import FlowGenPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from ..flowgen.models.unet3d import UNet3DConditionModel as UNet3DConditionModelFlow
from ..flowgen.models.controlnet import ControlNetModel
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf

class IG_FlowModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {            
            "model": (
            ['Motion-I2V',], 
            {
                "default": 'Motion-I2V'
            }),
            },
            }
    
    RETURN_TYPES = ("FLOWMODEL",)
    RETURN_NAMES =("flow_model",)
    FUNCTION = "load"
    CATEGORY = TREE_FLOW

    DESCRIPTION = """
    Diffusion-based flow estimation used for Motion-I2V:
    https://github.com/G-U-N/Motion-I2V
    
    Models are automatically downloaded to  
    ComfyUI/models/diffusers -folder
    """
    @torch.inference_mode()
    def load(self, model):
        
        diffusers_model_path = os.path.join(folder_paths.models_dir, "diffusers")
        checkpoint_path = os.path.join(diffusers_model_path, model)
        config_path = os.path.join(os.path.dirname(__file__), "..","configs")

        device = model_management.get_torch_device()
        inference_config = OmegaConf.load(os.path.join(config_path, "configs_flowgen","inference","inference.yaml"))

        if not os.path.exists(checkpoint_path):
            print(f"We need to download 17.3 Gb of files from Hugging Face. Go make a cup of tea...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"wangfuyun/Motion-I2V",
                                local_dir=checkpoint_path, 
                                local_dir_use_symlinks=False
                                )
        stage1_path = os.path.join(checkpoint_path, "models","stage1","StableDiffusion-FlowGen")
        stage2_path = os.path.join(checkpoint_path, "models","stage2","StableDiffusion")

        tokenizer = CLIPTokenizer.from_pretrained(stage1_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(stage1_path, subfolder="text_encoder")
        unet = UNet3DConditionModelFlow.from_config_2d(
            stage1_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                inference_config.unet_additional_kwargs
            ),
        )
        vae_img = AutoencoderKL.from_pretrained(stage2_path, subfolder="vae", use_safetensors=False)
        
        with open(os.path.join(stage1_path, "vae","config.json"), "r") as f:
            vae_config = json.load(f)
        vae = AutoencoderKL.from_config(vae_config)
        vae_pretrained_path = (os.path.join(stage1_path, "vae_flow","diffusion_pytorch_model.bin"))
        
        print("[Load vae weights from {}]".format(vae_pretrained_path))
        processed_ckpt = {}
        weight = torch.load(vae_pretrained_path, map_location="cpu")
        vae.load_state_dict(rename_state_dict_keys(weight), strict=True)
        controlnet = ControlNetModel.from_unet(unet)
        unet.controlnet = controlnet
        unet.control_scale = 1.0

        unet_pretrained_path = (os.path.join(stage1_path, "unet","diffusion_pytorch_model.bin"))
        print("[Load unet weights from {}]".format(unet_pretrained_path))
        weight = torch.load(unet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)
        # print_loading_issues(m, u)

        controlnet_pretrained_path = (os.path.join(stage1_path, "controlnet","controlnet.bin"))
        print("[Load controlnet weights from {}]".format(controlnet_pretrained_path))
        weight = torch.load(controlnet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)
        # print_loading_issues(m, u)
        
        print("finish loading")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            assert False
        
        self.flow_pipeline = FlowGenPipeline(
            vae_img=vae_img,
            vae_flow=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
        ).to(device)

        flow_model = {
            "flow_pipeline": self.flow_pipeline,
        }
        return (flow_model,)

class IG_FlowPredictor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flow_model": ("FLOWMODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "(blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"}),
                "first_frame": ("IMAGE",),
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 150}),
                "guidance_scale": ("FLOAT", {"default": 7, "min": 0.1, "max": 20}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = TREE_FLOW

    @torch.inference_mode()
    def run(self, flow_model, prompt, negative_prompt, first_frame, num_inference_steps, guidance_scale):
        flow_pipeline = flow_model['flow_pipeline']

        device = model_management.get_torch_device()

        # The first_frame is already a tensor with batch dimension in ComfyUI
        first_frame_tensor = first_frame.to(device)

        # Confirm the tensor shape
        print(f"Input tensor shape before permute: {first_frame_tensor.shape}")

        # Permute dimensions to [batch_size, channels, height, width]
        first_frame_tensor = rearrange(first_frame_tensor, 'b h w c -> b c h w')
        print(f"Input tensor shape after permute: {first_frame_tensor.shape}")

        # Remove alpha channel if present
        if first_frame_tensor.shape[1] == 4:
            first_frame_tensor = first_frame_tensor[:, :3, :, :]

        # Get height and width
        height, width = first_frame_tensor.shape[2], first_frame_tensor.shape[3]

        # Ensure height and width are divisible by 8
        height = (height // 8) * 8
        width = (width // 8) * 8
        first_frame_tensor = torch.nn.functional.interpolate(
            first_frame_tensor, size=(height, width), mode='bilinear', align_corners=False
        )

        # Normalize the tensor to [-1, 1] range
        first_frame_tensor = first_frame_tensor * 2 - 1  # Assuming input is in [0, 1]
        stride = list(range(8, 121, 8))
        # Run the flow pipeline
        sample = flow_pipeline(
            prompt=prompt,
            first_frame=first_frame_tensor.squeeze(0),
            video_length=len(stride),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            stride=torch.tensor([stride]).to(device),
            control=None,
            return_dict=False,
        )
        print(f"Sample shape: {sample.shape}")
        # Extract the generated frames
        # videos_tensor = output['videos'][0]  # Assuming batch_size = 1

        # # Process the tensor to convert from [-1, 1] to [0, 1]
        # images_tensor = (videos_tensor + 1) / 2
        # images_tensor = torch.clamp(images_tensor, 0, 1)

        sample = (sample * 2 - 1).clamp(-1, 1)
        # # sample = sample * (1 - brush_mask.to(sample.device))
        # sample[:, 0:1, ...] = sample[:, 0:1, ...] * width
        # sample[:, 1:2, ...] = sample[:, 1:2, ...] * height

        # Rearrange the sample to [frames, height, width, channels]
        flow_tensor = rearrange(sample, 'b c f h w -> (b f) h w c')

        # Convert the optical flow to RGB images
        rgb_images = flow_to_color(flow_tensor, clip_flow=None, convert_to_bgr=False)
        
        print(f"RGB image shape: {rgb_images.shape}")
        return (rgb_images,)
