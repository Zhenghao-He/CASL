#!/usr/bin/env python3
"""
SwiftEdit Batch Processing Script - Simple Version
批量处理文件夹中的图片

Usage:
    python batch_edit_simple.py --input_dir ./input_images --output_dir ./output_images --edit_prompt "a smiling face"
"""

import os
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from models import InverseModel, AuxiliaryModel, IPSBV2Model, tokenize_captions
from src.mask_ip_controller import MaskController

# 配置权重路径
SWIFTEDIT_WEIGHTS_ROOT = 'swiftedit_weights'

def to_binary(pix, threshold=0.5):
    if float(pix) > threshold:
        return 1.0
    else:
        return 0.0


@torch.no_grad()
def edit_image_simple(
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
    """简化的图像编辑函数"""
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 500
    final_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 999

    # 输入图像
    pil_img_cond = Image.open(img_path).resize((512, 512))
    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1

    # 预测反转噪声
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

    # 估算编辑掩码
    inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
    subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
    max_v = (subed.mean() * clamp_rate).item()
    mask12 = subed.clamp(0, max_v) / max_v
    mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to("cuda")

    # 编辑图像
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, scale_text_hiddenstate=scale_ta, scale_ip_fg=scale_edit, scale_ip_bg=scale_non_edit
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    res_gen_img, _ = ip_sb_model.gen_img(
        pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
    )

    # 简单处理：如果是batch=2，取第二张图（编辑后的）
    if res_gen_img.shape[0] == 2:
        return res_gen_img[1:2]  # 只取编辑后的图
    else:
        return res_gen_img


def tensor_to_pil_simple(tensor):
    """简化的tensor转PIL函数"""
    # 确保tensor在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 移除batch维度
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 确保是[C, H, W]格式
    if tensor.shape[0] not in [1, 3]:
        tensor = tensor.permute(2, 0, 1)
    
    # 转换为numpy
    numpy_image = tensor.detach().numpy()
    
    # 确保值在[0, 1]范围内
    numpy_image = numpy_image.clip(0, 1)
    
    # 转换为[0, 255]并转为uint8
    numpy_image = (numpy_image * 255).astype('uint8')
    
    # 转换为PIL格式[H, W, C]
    if numpy_image.shape[0] == 1:  # 灰度图
        numpy_image = numpy_image[0]
        pil_image = Image.fromarray(numpy_image, mode='L')
    elif numpy_image.shape[0] == 3:  # RGB图
        numpy_image = numpy_image.transpose(1, 2, 0)
        pil_image = Image.fromarray(numpy_image, mode='RGB')
    else:
        raise ValueError(f"Unsupported channel count: {numpy_image.shape[0]}")
    
    return pil_image


def get_image_files(directory):
    """获取目录中的所有图片文件"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            image_files.append(file_path)
    
    return sorted(image_files)


def batch_process_images(
    input_dir,
    output_dir,
    src_prompt="",
    edit_prompt="a smiling face",
):
    """批量处理图片文件夹"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print(f"❌ 在目录 {input_dir} 中未找到支持的图片文件")
        return
    
    print(f"📁 找到 {len(image_files)} 个图片文件")
    print(f"🎯 编辑提示: {edit_prompt}")
    print("=" * 50)
    
    # 加载模型
    print("⏳ 正在加载模型...")
    
    inverse_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
    inverse_model = InverseModel(inverse_ckpt)
    aux_model = AuxiliaryModel()
    
    path_unet_sb = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5")
    ip_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
    ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True)
    
    print("✅ 模型加载完成")
    print("=" * 50)
    
    # 处理每张图片
    start_time = time.time()
    successful_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] 处理: {img_file.name}")
            
            # 编辑图像
            result_tensor = edit_image_simple(
                str(img_file),
                src_prompt,
                edit_prompt,
                inverse_model,
                aux_model,
                ip_sb_model
            )
            
            # 转换为PIL并保存
            result_image = tensor_to_pil_simple(result_tensor)
            
            # 生成输出文件名
            output_filename = f"edited_{img_file.stem}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存结果
            result_image.save(output_path)
            print(f"  ✅ 已保存: {output_filename}")
            
            successful_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            continue
    
    # 统计结果
    elapsed_time = time.time() - start_time
    print("=" * 50)
    print(f"🎉 批处理完成!")
    print(f"📊 成功处理: {successful_count}/{len(image_files)} 张图片")
    print(f"⏱️ 总耗时: {elapsed_time:.1f} 秒")
    print(f"💾 结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SwiftEdit 批量图片编辑工具')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入图片目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出图片目录路径')
    parser.add_argument('--src_prompt', type=str, default="",
                       help='源图片描述提示词（可选）')
    parser.add_argument('--edit_prompt', type=str, default="a smiling face",
                       help='编辑目标提示词')
    
    args = parser.parse_args()
    
    print("🚀 SwiftEdit 批量处理 (简化版)")
    print("=" * 40)
    print(f"📁 输入目录: {args.input_dir}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🎯 编辑提示: {args.edit_prompt}")
    print()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        return
    
    batch_process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        src_prompt=args.src_prompt,
        edit_prompt=args.edit_prompt
    )


if __name__ == "__main__":
    main()