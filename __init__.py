"""
@author: IDGallagher
@title: IG Interpolation Nodes
@nickname: IG Interpolation Nodes
@description: Custom nodes to aid in the exploration of Latent Space
"""

from .nodes.flow import *

NODE_CLASS_MAPPINGS = {
    "IG Flow Model Loader":         IG_FlowModelLoader,
    "IG Flow Predictor":            IG_FlowPredictor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IG Flow Model Loader":    "💾 IG Flow Model Loader",
    "IG Flow Predictor":    "🌊 IG Flow Predictor",
}