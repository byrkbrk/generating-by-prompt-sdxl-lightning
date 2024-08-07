import os
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file



class StableDiffusion(object):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    checkpoints = {
        "1-step": [1, "sdxl_lightning_1step_unet_x0.safetensors"],
        "2-step": [2, "sdxl_lightning_2step_unet.safetensors"],
        "4-step": [4, "sdxl_lightning_4step_unet.safetensors"],
        "8-step": [8, "sdxl_lightning_8step_unet.safetensors"]
    }
    
    def __init__(self,
                 step_choice: str="4-step",
                 scheduler_name: str="euler_discrete_scheduler",
                 device: str=None,
                 create_dirs: bool=True
                 ):
        self.step_choice = step_choice
        self.scheduler_name = scheduler_name
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.pipeline = self.instantiate_pipeline(step_choice)
        if create_dirs: self.create_dirs(self.module_dir)
        
    def generate(self, prompt, step_choice, show=True, save=True):
        """Returns generated image for given text prompt"""
        if self._is_step_choice_changed(step_choice):
            self._update_pipeline(self.pipeline, step_choice)
            self._update_step_choice(step_choice)

        images = self.pipeline(prompt, 
                               num_inference_steps=self.get_num_inference_steps(step_choice), 
                               guidance_scale=0).images
        for i, image in enumerate(images):
            if save: 
                image.save(os.path.join(self.module_dir, "generated-images", f"generated_image_{i}.jpg"))
            if show:
                image.show()
        return images

    def instantiate_pipeline(self, step_choice, scheduler_name="euler_discrete_scheduler"):
        """Returns instantiated pipeline"""
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            StableDiffusion.base,
            torch_dtype=torch.float16,
            variant="fp16").to(self.device)
        self._update_pipeline(pipeline, step_choice, scheduler_name)
        return pipeline

    def _update_scheduler(self, pipeline, step_choice, scheduler_name):
        """Sets the scheduler of the pipeline for given scheduler name"""
        if scheduler_name == "euler_discrete_scheduler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(
                pipeline.scheduler.config,
                timestep_spacing="trailing",
                prediction_type="sample" if self.get_num_inference_steps(step_choice)==1 else "epsilon")
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
            return self.__class__.checkpoints[step_choice][0]
        else:
            raise ValueError(f"Unexpected step choice: {step_choice}")

    def get_checkpoint_name(self, step_choice):
        """Returns checkpoint name based on step choice"""
        return self.__class__.checkpoints[step_choice][1]
    
    def _update_pipeline(self, pipeline, step_choice, scheduler_name="euler_discrete_scheduler"):
        """Updates unet in pipeline based on step choice"""
        self._update_scheduler(pipeline, step_choice, scheduler_name)
        self._update_unet(pipeline, step_choice)
    
    def _update_unet(self, pipeline, step_choice):
        pipeline.unet.load_state_dict(load_file(hf_hub_download(self.__class__.repo, 
                                                                self.get_checkpoint_name(step_choice)),
                                                device=self.device.type))

    def _is_step_choice_changed(self, step_choice):
        """Returns True if step choice changed"""
        return self.step_choice != step_choice

    def _update_step_choice(self, step_choice):
        """Updates step_choice attribute"""
        self.step_choice = step_choice 
    


if __name__ == "__main__":
    prompt = ["an image of a turtle in Picasso style"]
    StableDiffusion(step_choice="1-step").generate(prompt)