import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np

from einops import rearrange, repeat
from omegaconf import OmegaConf
from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything


def load_image(image_path, device):
    """Load and preprocess image"""
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def process_single_image(model, image_path, target_prompt, output_dir, device, 
                        source_prompt="", step=4, layer=10, 
                        guidance_scale=7.5, num_inference_steps=50, seed=42):
    """Process a single image with MasaCtrl editing"""
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Load source image
    source_image = load_image(image_path, device)
    
    # Create prompts
    prompts = [source_prompt, target_prompt]
    
    # Invert the source image
    start_code, latents_list = model.invert(source_image,
                                          source_prompt,
                                          guidance_scale=guidance_scale,
                                          num_inference_steps=num_inference_steps,
                                          return_intermediates=True)
    start_code = start_code.expand(len(prompts), -1, -1, -1)
    
    # Results of direct synthesis (without MasaCtrl)
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    image_fixed = model([target_prompt],
                       latents=start_code[-1:],
                       num_inference_steps=num_inference_steps,
                       guidance_scale=guidance_scale)
    
    # Inference with MasaCtrl
    editor = MutualSelfAttentionControl(step, layer)
    regiter_attention_editor_diffusers(model, editor)
    
    # Generate edited image
    image_masactrl = model(prompts,
                          latents=start_code,
                          guidance_scale=guidance_scale)
    
    # Prepare output images
    out_image = torch.cat([source_image * 0.5 + 0.5,
                          image_masactrl[0:1],
                          image_fixed,
                          image_masactrl[-1:]], dim=0)
    
    # Save images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_image(out_image, os.path.join(output_dir, f"{base_name}_all_step{step}_layer{layer}.png"))
    save_image(out_image[0], os.path.join(output_dir, f"{base_name}_source.png"))
    save_image(out_image[1], os.path.join(output_dir, f"{base_name}_reconstructed.png"))
    save_image(out_image[2], os.path.join(output_dir, f"{base_name}_without_masactrl.png"))
    save_image(out_image[3], os.path.join(output_dir, f"{base_name}_masactrl_edited.png"))
    
    return True


def batch_edit_images(input_dir, output_dir, target_prompt, source_prompt="",
                     model_path="CompVis/stable-diffusion-v1-4", device="cuda",
                     step=4, layer=10, guidance_scale=7.5, num_inference_steps=50,
                     seed=42):
    """Batch edit images in a directory"""
    
    # Set device
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(0)  # set the GPU device
    
    # Load model
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, 
                             beta_schedule="scaled_linear", clip_sample=False, 
                             set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Target prompt: '{target_prompt}'")
    print(f"Source prompt: '{source_prompt}'")
    print(f"Step: {step}, Layer: {layer}")
    print(f"Output directory: {output_dir}")
    
    # Process each image
    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        image_path = os.path.join(input_dir, image_file)
        
        try:
            # Use different seed for each image to avoid identical results
            current_seed = seed + i
            
            process_single_image(
                model=model,
                image_path=image_path,
                target_prompt=target_prompt,
                output_dir=output_dir,
                device=device,
                source_prompt=source_prompt,
                step=step,
                layer=layer,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=current_seed
            )
            
            print(f"✓ Processed: {image_file}")
            
        except Exception as e:
            print(f"✗ Error processing {image_file}: {str(e)}")
            continue
    
    print(f"\nBatch processing completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch MasaCtrl Image Editing")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to save edited images")
    parser.add_argument("--target_prompt", type=str, required=True,
                       help="Target prompt for editing (e.g., 'a photo of a running corgi')")
    parser.add_argument("--source_prompt", type=str, default="",
                       help="Source prompt (default: empty string)")
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4",
                       help="Hugging Face model path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--step", type=int, default=4,
                       help="MasaCtrl step parameter")
    parser.add_argument("--layer", type=int, default=10,
                       help="MasaCtrl layer parameter")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale for diffusion")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    batch_edit_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_prompt=args.target_prompt,
        source_prompt=args.source_prompt,
        model_path=args.model_path,
        device=args.device,
        step=args.step,
        layer=args.layer,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed
    )