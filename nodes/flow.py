import os
import torch
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

from ..flowgen.pipelines.pipeline_flow_gen import FlowGenPipeline

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
    def load(self, model):
        device = model_management.get_torch_device()
        diffusers_model_path = os.path.join(folder_paths.models_dir,'diffusers')
        checkpoint_path = os.path.join(diffusers_model_path, model)

        if not os.path.exists(checkpoint_path):
            print(f"Selected model: {checkpoint_path} not found, downloading...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"wangfuyun/Motion-I2V",
                                local_dir=checkpoint_path, 
                                local_dir_use_symlinks=False
                                )
        self.flow_pipeline = FlowGenPipeline().to(device)

        flow_model = {
            "flow_pipeline": self.flow_pipeline,
        }
        return (flow_model,)