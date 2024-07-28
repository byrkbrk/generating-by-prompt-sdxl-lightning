import os
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file



class StableDiffusion(object):
    def __init__(self,
                 base: str="stabilityai/stable-diffusion-xl-base-1.0",
                 repo: str="ByteDance/SDXL-Lightning",
                 ckpt: str="sdxl_lightning_4step_unet.safetensors",
                 scheduler_name: str="euler_discrete_scheduler",
                 device: str=None,
                 create_dirs: bool=True
                 ):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.pipeline = self.instantiate_pipeline(base, repo, ckpt, scheduler_name, self.device)
        if create_dirs: self.create_dirs(self.module_dir)
        
    def generate(self):
        """Returns generated image for given text prompt"""
        pass

    def instantiate_pipeline(self, base, repo, checkpoint, scheduler_name, device):
        """Returns instantiated pipeline"""
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=self._instantiate_unet(base, repo, checkpoint, device),
            torch_dtype=torch.float16,
            variant="fp16").to(device)
        self._set_scheduler(pipeline, scheduler_name)
        return pipeline

    def _instantiate_unet(self, base, repo, checkpoint, device):
        """Returns instantiated UNet"""
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, checkpoint), device=device))
        return unet

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
    

if __name__ == "__main__":
    StableDiffusion()