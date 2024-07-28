import os
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file



class StableDiffusion(object):
    def __init__(self,
                 base: str="",
                 repo: str="",
                 ckpt: str="",
                 device: str=None,
                 create_dirs: bool=True
                 ):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.pipeline = ""
        if create_dirs: self.create_dirs(self.module_dir)
        
    def generate(self):
        """Returns generated image for given text prompt"""
        pass

    def instantiate_pipeline(self):
        """Returns instantiated pipeline"""
        pass

    def _set_scheduler(self, pipeline, scheduler_name):
        """Sets the scheduler of the pipeline for given scheduler name"""
        if scheduler_name == "euler_discrete_scheduler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config,
                                                                    timestep_spacing="trailing")
        else:
            raise ValueError(f"Undefined scheduler name: {scheduler_name}")

    def initialize_device(self, device):
        """Returns the device based on GPU availability"""
        pass

    def create_dirs(self, root):
        """Creates directories under 'root' directory required during inference"""
        dir_names = ["generated-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)
    
        