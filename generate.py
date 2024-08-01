from argparse import ArgumentParser
from stable_diffusion_xl_lightning import StableDiffusion



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate images by prompts using SDXL-lightning")
    parser.add_argument("prompt", 
                        nargs="+", 
                        type=str, 
                        help="Text prompt that be used for generating")
    parser.add_argument("--step_choice", 
                        type=str, 
                        default="4-step", 
                        choices=["1-step", "2-step", "4-step", "8-step"],
                        help="Step choice for inference. Default: '4-step'")
    parser.add_argument("--scheduler_name",
                        type=str,
                        default="euler_discrete_scheduler",
                        help="Scheduler name for inference. Default: 'euler_discrete_scheduler'")
    parser.add_argument("--device",
                        type=str,
                        default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="GPU device that be used during inference. Default: None")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    StableDiffusion(step_choice=args.step_choice,
                    scheduler_name=args.scheduler_name,
                    device=args.device).generate(args.prompt, args.step_choice)