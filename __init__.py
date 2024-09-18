"""
@author: IDGallagher
@title: IG Interpolation Nodes
@nickname: IG Interpolation Nodes
@description: Custom nodes to aid in the exploration of Latent Space
"""

from .nodes.flow import *
from .nodes.motion_painter import *

# Mount web directory
WEB_DIRECTORY = f"./web"

NODE_CLASS_MAPPINGS = {
    "MI2V Flow Model Loader":         MI2V_FlowModelLoader,
    "MI2V Flow Predictor":            MI2V_FlowPredictor,
    "MI2V Flow Animator":             MI2V_FlowAnimator,
    "MI2V Motion Painter":            MI2V_MotionPainter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MI2V Flow Model Loader":    "💾 MI2V Flow Model Loader",
    "MI2V Flow Predictor":       "🌊 MI2V Flow Predictor",
    "MI2V Flow Animator":        "🌊 MI2V Flow Animator",
    "MI2V Motion Painter":       "🎨 MI2V Motion Painter",
}