import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from safetensors.torch import load_file
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    return np.array(image)

def image_to_latent(image, vae, device):
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    latents = vae.encode(image).latent_dist.mean
    return latents * 0.18215

def latent_to_image(latents, vae):
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    return (image[0] * 255).astype(np.uint8)

def edit_image(image_path, lora_path, scales, start_noise, output_dir, device, prompt="a photo of a person"):
    # Load models
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # Load LoRA network - determine training method and modules from lora_path
    if 'full' in lora_path:
        train_method = 'full'
    elif 'noxattn' in lora_path:
        train_method = 'noxattn'
    else:
        train_method = 'noxattn'

    network_type = "c3lier"
    if train_method == 'xattn':
        network_type = 'lierla'

    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    # Determine rank and alpha from filename
    rank = 4
    alpha = 1.0
    if 'rank4' in lora_path:
        rank = 4
    if 'rank8' in lora_path:
        rank = 8
    if 'alpha1' in lora_path:
        alpha = 1.0

    network = LoRANetwork(
        unet,
        rank=rank,
        multiplier=1.0,
        alpha=alpha,
        train_method=train_method,
    ).to(device)
    
    # Load weights based on file extension
    if lora_path.endswith('.safetensors'):
        network.load_state_dict(load_file(lora_path), strict=False)
    else:
        network.load_state_dict(torch.load(lora_path), strict=False)

    # Load and process image
    image = load_image(image_path)
    latents = image_to_latent(image, vae, device)

    # Tokenize prompt and create text embeddings
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    # Create unconditional embeddings
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    # Combine embeddings for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Edit image
    noise_scheduler = scheduler
    noise_scheduler.set_timesteps(50)
    latents = latents * noise_scheduler.init_noise_sigma
    images_list = []

    for scale in scales:
        # Reset latents for each scale
        current_latents = latents.clone()
        
        for t in tqdm(noise_scheduler.timesteps):
            if t > start_noise:
                network.set_lora_slider(scale=0)
            else:
                network.set_lora_slider(scale=scale)

            # Expand the latents if we are doing classifier-free guidance
            latent_model_input = torch.cat([current_latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                with network:
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            current_latents = noise_scheduler.step(noise_pred, t, current_latents).prev_sample

        # Decode and save image
        edited_image = latent_to_image(current_latents, vae)
        images_list.append(edited_image)

    # Save results with better naming
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for idx, img in enumerate(images_list):
        output_filename = f"{base_name}_edited_{idx}.png"
        Image.fromarray(img).save(os.path.join(output_dir, output_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Image Editor using LoRA")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save edited images")
    parser.add_argument("--prompt", type=str, default="a photo of a person", help="Text prompt for editing")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the trained LoRA weights")
    parser.add_argument("--scales", type=int, nargs='+', default=[0, 2, 4, 6], help="List of scales for editing")
    parser.add_argument("--start_noise", type=int, default=500, help="Start noise level for editing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    args = parser.parse_args()

    input_images = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in input_images:
        edit_image(image_path, args.lora_path, args.scales, args.start_noise, args.output_dir, args.device, args.prompt)