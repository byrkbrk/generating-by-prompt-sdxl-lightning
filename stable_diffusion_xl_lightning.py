import os
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file



class StableDiffusion(object):
    def __init__(self,
                 base: str="stabilityai/stable-diffusion-xl-base-1.0",
                 repo: str="ByteDance/SDXL-Lightning",
                 step_choice: str="2-step",
                 scheduler_name: str="euler_discrete_scheduler",
                 device: str=None,
                 create_dirs: bool=True
                 ):
        self.base = base
        self.repo = repo
        self.step_choice = step_choice
        self.scheduler_name = scheduler_name
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.pipeline = self.instantiate_pipeline(base, repo, self.get_checkpoint_name(step_choice), scheduler_name, self.device)
        if create_dirs: self.create_dirs(self.module_dir)
        
    def generate(self, prompt, step_choice, show=True, save=True):
        """Returns generated image for given text prompt"""
        if step_choice != self.step_choice:
            self._update_pipeline(step_choice)
            self.step_choice = step_choice
        images = self.pipeline(prompt, 
                               num_inference_steps=self.get_num_inference_steps(step_choice), 
                               guidance_scale=0
                               ).images
        for i, image in enumerate(images):
            if save: 
                image.save(os.path.join(self.module_dir, "generated-images", f"generated_image_{i}.jpg"))
            if show:
                image.show()
        return images

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
        unet.load_state_dict(load_file(hf_hub_download(repo, checkpoint), 
                                       device=device.type))
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
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)

    def create_dirs(self, root):
        """Creates directories under 'root' directory required during inference"""
        dir_names = ["generated-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)
    
    def get_num_inference_steps(self, step_choice):
        """Returns number of inference steps based on step choice"""
        if step_choice in {"1-step", "2-step", "4-step", "8-step"}:
            return int(step_choice[0])
        else:
            raise ValueError(f"Unexpected step choice: {step_choice}")

    def get_checkpoint_name(self, step_choice):
        """Returns checkpoint name based on step choice"""
        n_steps = self.get_num_inference_steps(step_choice)
        if n_steps == 1:
            return f"sdxl_lightning_{n_steps}step_unet_x0.safetensors"
        return f"sdxl_lightning_{n_steps}step_unet.safetensors"
    
    def _update_pipeline(self, step_choice):
        """Updates pipeline attribute based on step choice"""
        self.pipeline = self.instantiate_pipeline(base=self.base,
                                                  repo=self.repo,
                                                  checkpoint=self.get_checkpoint_name(step_choice),
                                                  scheduler_name=self.scheduler_name,
                                                  device=self.device)


if __name__ == "__main__":
    prompt = ["an image of a turtle in Picasso style"]
    StableDiffusion(step_choice="1-step").generate(prompt)