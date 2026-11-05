# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import time
import argparse
import glob
from tqdm import tqdm

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from models import *

#
# Configure this path to where you have stored the local copy of the weights:
#
SWIFTEDIT_WEIGHTS_ROOT = 'swiftedit_weights'

def to_binary(pix, threshold=0.5):
    if float(pix) > threshold:
        return 1.0
    else:
        return 0.0


@torch.no_grad()
def edit_image(
    img_path,
    src_p,
    edit_p,
    inverse_model,
    aux_model,
    ip_sb_model,
    scale_ta=1,
    scale_edit=0.2,
    scale_non_edit=1,
    clamp_rate=3.0,
    mask_threshold=0.5,
):
    """
        Save keysteps to file.
            + img_path: path to the source image.
            + src_p: Source Prompt that describes source image (could leave it empty).
            + edit_p: Edit Prompt that describes your desired changes.
    """
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 500
    final_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 999

    # Input Image
    pil_img_cond = Image.open(img_path).resize((512, 512))

    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1

    # Predict inverted noise
    latents = inverse_model.vae.encode(
        processed_image.to(inverse_model.weight_dtype)
    ).latent_dist.sample()
    latents = latents * inverse_model.vae.config.scaling_factor
    dub_latents = torch.cat([latents] * 2, dim=0)

    input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to("cuda")
    encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
        dtype=inverse_model.weight_dtype
    )

    predict_inverted_code = inverse_model.unet_inverse(
        dub_latents, mid_timestep, encoder_hidden_state
    ).sample.to("cuda", dtype=inverse_model.weight_dtype)

    # Estimate editing mask
    inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
    subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
    max_v = (subed.mean() * clamp_rate).item()
    mask12 = subed.clamp(0, max_v) / max_v
    mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to("cuda")

    # Edit images
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, scale_text_hiddenstate=scale_ta, scale_ip_fg=scale_edit, scale_ip_bg=scale_non_edit
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    res_gen_img, _ = ip_sb_model.gen_img(
        pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
    )

    # 重要：只取第二张图（编辑后的图像）
    # res_gen_img的shape通常是 [2, 3, 512, 512]，其中：
    # - 第0张：对应src_p的重建图
    # - 第1张：对应edit_p的编辑图
    if res_gen_img.shape[0] == 2:
        edited_img = res_gen_img[1:2]  # 只取第二张图，保持batch维度
        print(f"  Debug: 提取编辑图像，原shape={res_gen_img.shape}, 新shape={edited_img.shape}")
        return edited_img
    else:
        print(f"  Warning: 意外的输出shape={res_gen_img.shape}，返回全部")
        return res_gen_img


def batch_edit_images(input_dir, output_dir, src_prompt, edit_prompt):
    """批量处理图像"""
    # 初始化模型 - 完全参考infer.py
    inverse_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
    inverse_model = InverseModel(inverse_ckpt)
    aux_model = AuxiliaryModel()

    path_unet_sb = (os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5"))
    ip_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
    ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    # 处理每张图像
    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            print(f"[{i+1}/{len(image_files)}] Processing: {os.path.basename(img_path)}")
            
            start_time = time.time()
            result = edit_image(img_path, src_prompt, edit_prompt, inverse_model, aux_model, ip_sb_model)
            edit_time = time.time() - start_time
            
            # 保存结果 - 使用normalize=True自动处理范围转换
            output_name = f"edited_{os.path.splitext(os.path.basename(img_path))[0]}.png"
            output_path = os.path.join(output_dir, output_name)
            save_image(result, output_path, normalize=True)
            
            print(f"  ✓ Saved: {output_name} (time: {edit_time:.2f}s)")
            
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(img_path)}: {str(e)}")
            continue
    
    print("Batch editing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch edit images using SwiftEdit')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save edited images')
    parser.add_argument('--src_prompt', type=str, default="",
                      help='Source prompt describing the input image')
    parser.add_argument('--edit_prompt', type=str, required=True,
                      help='Edit prompt describing desired changes')
    
    args = parser.parse_args()
    
    print("🚀 SwiftEdit Batch Processing")
    print("=============================")
    print(f"📁 输入目录: {args.input_dir}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🔤 源提示: {args.src_prompt}")
    print(f"🎯 编辑提示: {args.edit_prompt}")
    print()
    
    batch_edit_images(args.input_dir, args.output_dir, args.src_prompt, args.edit_prompt)