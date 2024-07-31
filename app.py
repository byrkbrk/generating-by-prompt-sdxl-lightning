import gradio as gr
from stable_diffusion_xl_lightning import StableDiffusion



if __name__ == "__main__":
    stable_diffusion = StableDiffusion(create_dirs=False)
    gr_interface = gr.Interface(
        fn=lambda prompt, step_choice: stable_diffusion.generate(prompt, 
                                                                 step_choice, 
                                                                 save=False, 
                                                                 show=False)[0],
        inputs=[
            gr.Textbox(lines=3,
                       placeholder="an image of a lion in Claude Monet style",
                       label="Prompt"),
            gr.Dropdown(["1-step", "2-step", "4-step", "8-step"],
                        value="4-step",
                        label="Inference steps")
        ],
        outputs=gr.Image(type="pil"),
        title="Generate by Prompt using SDXL-lightning"
    )
    gr_interface.launch()