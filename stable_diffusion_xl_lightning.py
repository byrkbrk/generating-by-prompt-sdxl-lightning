import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file



class StableDiffusion(object):
    def __init__(self):
        pass

    def generate(self):
        """Returns generated image for given text prompt"""
        pass

    def instantiate_pipeline(self):
        """Returns instantiated pipeline"""
        pass