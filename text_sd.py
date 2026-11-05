# import os
# import argparse
# import numpy as np
# from tqdm import tqdm
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from diffusers import StableDiffusionPipeline, DDIMScheduler
# from losses.clip_loss import CLIPLoss


# # -------------------------
# # Utils
# # -------------------------
# def seed_everything(seed: int):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)


# def get_text_embeds(pipe: StableDiffusionPipeline, prompts, device):
#     tok = pipe.tokenizer(
#         prompts,
#         padding="max_length",
#         max_length=pipe.tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
#     with torch.no_grad():
#         cond = pipe.text_encoder(tok.input_ids.to(device))[0]

#     uncond_tok = pipe.tokenizer(
#         [""] * len(prompts),
#         padding="max_length",
#         max_length=pipe.tokenizer.model_max_length,
#         return_tensors="pt",
#     )
#     with torch.no_grad():
#         uncond = pipe.text_encoder(uncond_tok.input_ids.to(device))[0]

#     return uncond, cond


# def cfg_eps(unet, latent_in, t, uncond_emb, cond_emb, guidance_scale):
#     """
#     latent_in: [B,C,H,W] -> eps [B,C,H,W]
#     """
#     lat_in = torch.cat([latent_in, latent_in], dim=0)
#     emb = torch.cat([uncond_emb, cond_emb], dim=0)
#     eps = unet(lat_in, t, encoder_hidden_states=emb).sample
#     eps_u, eps_c = eps.chunk(2)
#     return eps_u + guidance_scale * (eps_c - eps_u)


# def ddim_predict_x0(x_t, eps, alpha_bar_t):
#     return (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)


# def ddim_asyrp_step(x_t_edit, eps_edit, eps_ref, scheduler, t, t_prev):
#     """
#     Asyrp asymmetric eta=0:
#       P uses x0_edit from (x_t_edit, eps_edit) at t
#       D uses eps_ref
#     Returns: x_prev_edit, x0_edit(t)
#     """
#     device = x_t_edit.device
#     dtype = x_t_edit.dtype
#     a_t = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)
#     a_prev = scheduler.alphas_cumprod[t_prev].to(device=device, dtype=dtype)

#     x0_edit = (x_t_edit - torch.sqrt(1 - a_t) * eps_edit) / torch.sqrt(a_t)
#     x_prev_edit = torch.sqrt(a_prev) * x0_edit + torch.sqrt(1 - a_prev) * eps_ref
#     return x_prev_edit, x0_edit


# def decode_latents_to_rgb_tensor(pipe, latents):
#     """
#     Differentiable decode for CLIP loss.
#     Return imgs in [-1,1], [B,3,H,W].
#     NOTE: This is expensive; we prevent OOM via per-step backward + detach.
#     """
#     # decode VAE in fp32 to reduce NaNs
#     pipe.vae.to(dtype=torch.float32)
#     latents = latents.to(device=pipe.vae.device, dtype=torch.float32)
#     latents = latents / pipe.vae.config.scaling_factor
#     imgs = pipe.vae.decode(latents).sample  # [-1,1]
#     imgs = imgs.nan_to_num(0.0).clamp(-1, 1)
#     return imgs


# @torch.no_grad()
# def decode_latents_to_pil(pipe, latents):
#     latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
#     latents = latents / pipe.vae.config.scaling_factor
#     imgs = pipe.vae.decode(latents).sample
#     imgs = (imgs / 2 + 0.5).clamp(0, 1)
#     imgs = imgs.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
#     imgs = imgs.detach().cpu()
#     pil = []
#     for i in range(imgs.shape[0]):
#         arr = (imgs[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
#         pil.append(Image.fromarray(arr))
#     return pil


# # -------------------------
# # Mid-block injection (h-space)
# # -------------------------
# class MidBlockInjector:
#     def __init__(self, unet: nn.Module, delta: nn.Parameter):
#         self.unet = unet
#         self.delta = delta
#         self.enabled = False
#         self.handle = None

#     def _hook(self, module, inputs, output):
#         if not self.enabled:
#             return output
#         return output + self.delta

#     def install(self):
#         if self.handle is None:
#             self.handle = self.unet.mid_block.register_forward_hook(self._hook)

#     def remove(self):
#         if self.handle is not None:
#             self.handle.remove()
#             self.handle = None


# # -------------------------
# # Stable directional CLIP loss (avoid NaN)
# # -------------------------
# def stable_clip_directional_loss(
#     clip_loss_func: CLIPLoss,
#     src_img_m11: torch.Tensor,  # [-1,1]
#     src_text: str,
#     tgt_img_m11: torch.Tensor,  # [-1,1]
#     tgt_text: str,
#     eps: float = 1e-4,
# ):
#     """
#     Compute direction cosine similarity stably:
#       d_img = f(tgt) - f(src)
#       d_txt = target_direction
#       loss = 1 - cos(d_img, d_txt) with eps
#     """
#     if clip_loss_func.target_direction is None:
#         clip_loss_func.target_direction = clip_loss_func.compute_text_direction(src_text, tgt_text)

#     src_feat = clip_loss_func.get_image_features(src_img_m11)  # [B,D], normalized
#     tgt_feat = clip_loss_func.get_image_features(tgt_img_m11)  # [B,D], normalized
#     d = tgt_feat - src_feat                                    # [B,D]

#     # cosine_similarity handles near-zero d with eps, avoids 0/0 NaN
#     cos = F.cosine_similarity(
#         d,
#         clip_loss_func.target_direction.expand_as(d),
#         dim=-1,
#         eps=eps,
#     )
#     return (1.0 - cos).mean()


# # -------------------------
# # Train: multi-step, per-step backward (gradient accumulation)
# # -------------------------
# def train_asyrp_multistep_accum_backward(args):
#     device = torch.device(args.device)
#     torch.set_float32_matmul_precision("high")

#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float16,
#         safety_checker=None,
#         requires_safety_checker=False,
#     ).to(device)
#     pipe.set_progress_bar_config(disable=True)
#     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

#     # reduce UNet activation memory
#     pipe.unet.enable_gradient_checkpointing()

#     # CLIP loss module (freeze)
#     clip_loss_func = CLIPLoss(
#         device,
#         lambda_direction=1.0,
#         lambda_patch=0.0,
#         lambda_global=0.0,
#         lambda_manifold=0.0,
#         lambda_texture=0.0,
#         clip_model=args.clip_model,
#     ).to(device)
#     for p in clip_loss_func.parameters():
#         p.requires_grad = False
#     for p in clip_loss_func.model.parameters():
#         p.requires_grad = False

#     os.makedirs(args.outdir, exist_ok=True)
#     ckpt_path = os.path.join(args.outdir, "delta.pt")

#     # CFG embeddings
#     uncond_emb, cond_emb = get_text_embeds(pipe, [args.src_prompt], device)
#     uncond_emb = uncond_emb.to(dtype=pipe.unet.dtype)
#     cond_emb = cond_emb.to(dtype=pipe.unet.dtype)

#     # delta + hook
#     C = 1280
#     delta = nn.Parameter(torch.zeros((1, C, 1, 1), device=device, dtype=pipe.unet.dtype))
#     injector = MidBlockInjector(pipe.unet, delta)
#     injector.install()

#     optim = torch.optim.Adam([delta], lr=args.lr)

#     # timesteps
#     pipe.scheduler.set_timesteps(args.steps, device=device)
#     timesteps = pipe.scheduler.timesteps  # decreasing
#     assert 0 <= args.t_edit_idx < len(timesteps) - 1
#     t0_idx = args.t0_idx if args.t0_idx >= 0 else (len(timesteps) - 1)
#     t0_idx = min(t0_idx, len(timesteps) - 1)
#     assert args.t_edit_idx <= t0_idx

#     train_indices = list(range(args.t_edit_idx, t0_idx + 1))
#     print(f"[train] steps={args.steps} t_edit_idx={args.t_edit_idx} t0_idx={t0_idx} #train_steps={len(train_indices)}")
#     print(f"[train] asym={args.asyrp_asym} edit_only_at_one_step={args.edit_only_at_one_step} target_step={args.target_step}")

#     for it in tqdm(range(args.iters), desc="train"):
#         g = torch.Generator(device=device).manual_seed(args.seed + it)

#         # init xT
#         xT = torch.randn(
#             (1, pipe.unet.config.in_channels, args.height // 8, args.width // 8),
#             generator=g,
#             device=device,
#             dtype=pipe.unet.dtype,
#         )
#         lat_ref = xT.clone()
#         lat_edit = xT.clone()

#         optim.zero_grad(set_to_none=True)

#         # stats
#         loss_l1_sum = 0.0
#         loss_clip_sum = 0.0

#         # Multi-step, but: EACH STEP BACKWARD NOW, then detach lat_edit to break BPTT
#         for idx in train_indices:
#             t = timesteps[idx]
#             t_prev = timesteps[idx + 1] if (idx + 1) < len(timesteps) else timesteps[idx]

#             # cache x_t for correct x0(t)
#             x_t_ref = lat_ref
#             x_t_edit = lat_edit

#             # ref eps (no grad)
#             with torch.no_grad():
#                 latent_in_ref = pipe.scheduler.scale_model_input(x_t_ref, t)
#                 eps_ref = cfg_eps(pipe.unet, latent_in_ref, t, uncond_emb, cond_emb, args.guidance_scale)

#             # decide if injection enabled at this step
#             if args.edit_only_at_one_step == 1:
#                 injector.enabled = (idx == args.target_step)
#             else:
#                 injector.enabled = (idx >= args.t_edit_idx)

#             # edit eps (grad can flow to delta)
#             latent_in_edit = pipe.scheduler.scale_model_input(x_t_edit, t)
#             eps_edit = cfg_eps(pipe.unet, latent_in_edit, t, uncond_emb, cond_emb, args.guidance_scale)

#             # x0 at timestep t
#             alpha_bar_t = pipe.scheduler.alphas_cumprod[t].to(device=device, dtype=pipe.unet.dtype)
#             x0_ref_t = ddim_predict_x0(x_t_ref, eps_ref, alpha_bar_t)

#             if args.asyrp_asym == 1 and injector.enabled:
#                 x_prev_edit, x0_edit_t = ddim_asyrp_step(x_t_edit, eps_edit, eps_ref, pipe.scheduler, t, t_prev)
#             else:
#                 x0_edit_t = ddim_predict_x0(x_t_edit, eps_edit, alpha_bar_t)
#                 x_prev_edit = pipe.scheduler.step(eps_edit, t, x_t_edit).prev_sample

#             # update latents: ref always standard, edit possibly asymmetric
#             with torch.no_grad():
#                 lat_ref = pipe.scheduler.step(eps_ref, t, x_t_ref).prev_sample
#             lat_edit = x_prev_edit

#             # losses
#             l1 = F.l1_loss(x0_edit_t.float(), x0_ref_t.float())

#             # directional CLIP on decoded x0 (still per-step, but we will backward immediately)
#             img_ref = decode_latents_to_rgb_tensor(pipe, x0_ref_t)
#             img_edit = decode_latents_to_rgb_tensor(pipe, x0_edit_t)

#             clip_dir = stable_clip_directional_loss(
#                 clip_loss_func,
#                 src_img_m11=img_ref,
#                 src_text=args.src_prompt,
#                 tgt_img_m11=img_edit,
#                 tgt_text=args.tgt_prompt,
#                 eps=args.clip_cos_eps,
#             )

#             if args.use_log_warp == 1:
#                 clip_dir = clip_dir.clamp(0.0, 2.0)
#                 clip_dir = -torch.log((2.0 - clip_dir + 1e-6) / 2.0)

#             step_loss = args.l1_w * l1 + args.clip_w * clip_dir

#             # regularize delta each step (small, but helps keep it in range)
#             step_loss = step_loss + args.l2 * (delta.float().pow(2).mean())

#             # backward immediately (gradient accumulation)
#             step_loss.backward()

#             # important: break time-chain to avoid holding full trajectory graph
#             lat_edit = lat_edit.detach()

#             # optional: clear some big refs early (helps python GC)
#             del img_ref, img_edit

#             loss_l1_sum += float(l1.detach())
#             loss_clip_sum += float(clip_dir.detach())

#         # apply one optimizer step per outer iter (like Asyrp)
#         torch.nn.utils.clip_grad_norm_([delta], args.grad_clip)
#         optim.step()

#         # clamp delta
#         with torch.no_grad():
#             delta.clamp_(-args.delta_clip, args.delta_clip)

#         # NaN guard
#         if torch.isnan(delta).any():
#             raise RuntimeError("delta became NaN. Suggested: reduce lr, increase clip_cos_eps, tighten delta_clip, increase l2.")

#         # periodic save
#         if (it % args.save_every == 0) or (it == args.iters - 1):
#             with torch.no_grad():
#                 # generate final image for this iter
#                 img_final = decode_latents_to_pil(pipe, lat_edit.float())[0]
#                 img_final.save(os.path.join(args.outdir, f"train_it{it:04d}_final.png"))

#         if it % args.log_every == 0:
#             denom = max(1, len(train_indices))
#             print(f"[it {it}] "
#                   f"l1={loss_l1_sum/denom:.4f} clip={loss_clip_sum/denom:.4f} "
#                   f"|delta|={float(delta.abs().mean().detach()):.5f}")

#     torch.save({"delta": delta.detach().float().cpu(), "args": vars(args)}, ckpt_path)
#     print(f"[saved] {ckpt_path}")


# # -------------------------
# # Inference
# # -------------------------
# @torch.no_grad()
# def infer_sweep(args):
#     device = torch.device(args.device)

#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float16,
#         safety_checker=None,
#         requires_safety_checker=False,
#     ).to(device)
#     pipe.set_progress_bar_config(disable=True)
#     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

#     ckpt = torch.load(args.ckpt, map_location="cpu")
#     delta_base = ckpt["delta"]  # [1,C,1,1]

#     delta = nn.Parameter(delta_base.to(device=device, dtype=pipe.unet.dtype), requires_grad=False)
#     injector = MidBlockInjector(pipe.unet, delta)
#     injector.install()

#     uncond_emb, cond_emb = get_text_embeds(pipe, [args.src_prompt], device)
#     uncond_emb = uncond_emb.to(dtype=pipe.unet.dtype)
#     cond_emb = cond_emb.to(dtype=pipe.unet.dtype)

#     os.makedirs(args.outdir, exist_ok=True)

#     pipe.scheduler.set_timesteps(args.steps, device=device)
#     timesteps = pipe.scheduler.timesteps

#     scales = args.scales
#     if isinstance(scales, str):
#         scales = [float(s) for s in scales.split(",")]

#     g = torch.Generator(device=device).manual_seed(args.seed)
#     xT = torch.randn(
#         (1, pipe.unet.config.in_channels, args.height // 8, args.width // 8),
#         generator=g,
#         device=device,
#         dtype=pipe.unet.dtype,
#     )

#     results = []
#     for s in scales:
#         lat_ref = xT.clone()
#         lat_edit = xT.clone()
#         delta.data = delta_base.to(device=device, dtype=pipe.unet.dtype) * s

#         for idx, t in enumerate(timesteps):
#             t_prev = timesteps[idx + 1] if (idx + 1) < len(timesteps) else timesteps[idx]

#             x_t_ref = lat_ref
#             x_t_edit = lat_edit

#             # ref eps
#             latent_in_ref = pipe.scheduler.scale_model_input(x_t_ref, t)
#             eps_ref = cfg_eps(pipe.unet, latent_in_ref, t, uncond_emb, cond_emb, args.guidance_scale)
#             lat_ref = pipe.scheduler.step(eps_ref, t, x_t_ref).prev_sample

#             # injection control
#             if args.edit_only_at_one_step == 1:
#                 injector.enabled = (idx == args.target_step)
#             else:
#                 injector.enabled = (idx >= args.t_edit_idx)

#             latent_in_edit = pipe.scheduler.scale_model_input(x_t_edit, t)
#             eps_edit = cfg_eps(pipe.unet, latent_in_edit, t, uncond_emb, cond_emb, args.guidance_scale)

#             if args.asyrp_asym == 1 and injector.enabled:
#                 lat_edit, _ = ddim_asyrp_step(x_t_edit, eps_edit, eps_ref, pipe.scheduler, t, t_prev)
#             else:
#                 lat_edit = pipe.scheduler.step(eps_edit, t, x_t_edit).prev_sample

#         img = decode_latents_to_pil(pipe, lat_edit.float())[0]
#         out_path = os.path.join(args.outdir, f"sweep_scale_{s:+.2f}.png")
#         img.save(out_path)
#         results.append(out_path)

#     print("[saved]")
#     for p in results:
#         print(" ", p)


# def build_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--mode", type=str, choices=["train", "infer"], required=True)
#     p.add_argument("--device", type=str, default="cuda:0")
#     p.add_argument("--outdir", type=str, default="sd15_asyrp_multistep_out")
#     p.add_argument("--seed", type=int, default=0)

#     # SD
#     p.add_argument("--height", type=int, default=512)
#     p.add_argument("--width", type=int, default=512)
#     p.add_argument("--steps", type=int, default=30)
#     p.add_argument("--guidance_scale", type=float, default=3.0)

#     # prompts
#     p.add_argument("--src_prompt", type=str, default="a photo of a person")
#     p.add_argument("--tgt_prompt", type=str, default="a photo of a smiling person")

#     # CLIP
#     p.add_argument("--clip_model", type=str, default="ViT-B/32")
#     p.add_argument("--clip_cos_eps", type=float, default=1e-4)

#     # time indices
#     p.add_argument("--t_edit_idx", type=int, default=10, help="editing starts from this timestep index (inclusive)")
#     p.add_argument("--t0_idx", type=int, default=-1, help="train ends at this index (inclusive), -1=last")

#     # inject control
#     p.add_argument("--edit_only_at_one_step", type=int, default=0)
#     p.add_argument("--target_step", type=int, default=15)

#     # asymmetric update
#     p.add_argument("--asyrp_asym", type=int, default=1)

#     # train
#     p.add_argument("--iters", type=int, default=200)
#     p.add_argument("--lr", type=float, default=3e-4)
#     p.add_argument("--l2", type=float, default=1e-3)
#     p.add_argument("--clip_w", type=float, default=3.0)
#     p.add_argument("--l1_w", type=float, default=0.0)
#     p.add_argument("--use_log_warp", type=int, default=0)

#     # stability
#     p.add_argument("--grad_clip", type=float, default=0.5)
#     p.add_argument("--delta_clip", type=float, default=0.02)

#     # logging/saving
#     p.add_argument("--save_every", type=int, default=20)
#     p.add_argument("--log_every", type=int, default=10)

#     # infer
#     p.add_argument("--ckpt", type=str, default="")
#     p.add_argument("--scales", type=str, default="-2,-1,0,1,2")

#     return p.parse_args()


# if __name__ == "__main__":
#     args = build_args()
#     seed_everything(args.seed)

#     if args.mode == "train":
#         train_asyrp_multistep_accum_backward(args)
#     else:
#         assert args.ckpt, "--ckpt is required for infer"
#         infer_sweep(args)
# text_sd.py
# SD1.5 + Asyrp-style multi-timestep training
# - two paths: ref(no hook, no_grad) vs edit(with hook)
# - multi-timestep
# - per-timestep backward (gradient accumulation)
# - negative prompt supported (uncond/cond/neg embeddings)
# - injection schedule supports "range" or "single-step"
# - Asyrp asymmetric update (P=edit, D=ref) when enabled

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler
from losses.clip_loss import CLIPLoss


# -------------------------
# Utils
# -------------------------
def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_inject_steps(spec: str):
    """
    spec examples:
      "22-25"  -> {22,23,24,25}
      "15"     -> {15}
      "22,24,25" -> {22,24,25}
      "" or "none" -> empty set
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "none", "null"):
        return set()
    if "-" in spec:
        a, b = spec.split("-")
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        return set(range(a, b + 1))
    if "," in spec:
        return set(int(x.strip()) for x in spec.split(",") if x.strip())
    return {int(spec)}


def get_text_embed(pipe: StableDiffusionPipeline, prompt: str, device):
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        emb = pipe.text_encoder(tok.input_ids.to(device))[0]  # [1, seq, dim]
    return emb


def get_three_text_embeds(pipe: StableDiffusionPipeline, pos_prompt: str, neg_prompt: str, device):
    """
    Return:
      uncond_emb: prompt=""
      cond_emb:   pos_prompt
      neg_emb:    neg_prompt
    All: [1, seq, dim]
    """
    uncond = get_text_embed(pipe, "", device)
    cond = get_text_embed(pipe, pos_prompt, device)
    neg = get_text_embed(pipe, neg_prompt, device)
    return uncond, cond, neg


def cfg_eps_posneg(unet, latent_in, t, uncond_emb, cond_emb, neg_emb, gs_pos: float, gs_neg: float):
    """
    "Positive + Negative" CFG (3-way)
    eps = eps_u + gs_pos*(eps_pos - eps_u) - gs_neg*(eps_neg - eps_u)
        = (1+gs_neg)*eps_u + gs_pos*eps_pos - gs_neg*eps_neg

    latent_in: [B,C,H,W]  (B=1 here)
    returns:   [B,C,H,W]
    """
    # batch 3
    lat_in = torch.cat([latent_in, latent_in, latent_in], dim=0)           # [3B,C,H,W]
    emb = torch.cat([uncond_emb, cond_emb, neg_emb], dim=0)                # [3,seq,dim]
    eps = unet(lat_in, t, encoder_hidden_states=emb).sample                # [3B,C,H,W]
    eps_u, eps_p, eps_n = eps.chunk(3)
    return eps_u + gs_pos * (eps_p - eps_u) - gs_neg * (eps_n - eps_u)


def ddim_predict_x0(x_t, eps, alpha_bar_t):
    return (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)


def ddim_asyrp_step(x_t_edit, eps_edit, eps_ref, scheduler, t, t_prev):
    """
    Asyrp asymmetric eta=0:
      P uses x0_edit from (x_t_edit, eps_edit) at t
      D uses eps_ref
    """
    device = x_t_edit.device
    dtype = x_t_edit.dtype
    a_t = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)
    a_prev = scheduler.alphas_cumprod[t_prev].to(device=device, dtype=dtype)

    x0_edit = (x_t_edit - torch.sqrt(1 - a_t) * eps_edit) / torch.sqrt(a_t)
    x_prev_edit = torch.sqrt(a_prev) * x0_edit + torch.sqrt(1 - a_prev) * eps_ref
    return x_prev_edit, x0_edit


def decode_latents_to_rgb_tensor(pipe, latents):
    """
    Differentiable decode for CLIP loss.
    Return imgs in [-1,1].
    """
    pipe.vae.to(dtype=torch.float32)
    latents = latents.to(device=pipe.vae.device, dtype=torch.float32)
    latents = latents / pipe.vae.config.scaling_factor
    imgs = pipe.vae.decode(latents).sample  # [-1,1]
    imgs = imgs.nan_to_num(0.0).clamp(-1, 1)
    return imgs


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    latents = latents / pipe.vae.config.scaling_factor
    imgs = pipe.vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    imgs = imgs.detach().cpu()
    pil = []
    for i in range(imgs.shape[0]):
        arr = (imgs[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        pil.append(Image.fromarray(arr))
    return pil


# -------------------------
# Mid-block injection (h-space)
# -------------------------
class MidBlockInjector:
    def __init__(self, unet: nn.Module, delta: nn.Parameter):
        self.unet = unet
        self.delta = delta
        self.enabled = False
        self.handle = None

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return output
        return output + self.delta

    def install(self):
        if self.handle is None:
            self.handle = self.unet.mid_block.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# -------------------------
# Stable directional CLIP loss (avoid NaN)
# -------------------------
def stable_clip_directional_loss(
    clip_loss_func: CLIPLoss,
    src_img_m11: torch.Tensor,  # [-1,1]
    src_text: str,
    tgt_img_m11: torch.Tensor,  # [-1,1]
    tgt_text: str,
    eps: float = 1e-4,
):
    if clip_loss_func.target_direction is None:
        clip_loss_func.target_direction = clip_loss_func.compute_text_direction(src_text, tgt_text)

    src_feat = clip_loss_func.get_image_features(src_img_m11)  # [B,D]
    tgt_feat = clip_loss_func.get_image_features(tgt_img_m11)  # [B,D]
    d = tgt_feat - src_feat

    cos = F.cosine_similarity(
        d,
        clip_loss_func.target_direction.expand_as(d),
        dim=-1,
        eps=eps,
    )
    return (1.0 - cos).mean()


# -------------------------
# Train: multi-step, per-step backward, with negative prompt CFG
# -------------------------
def train(args):
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # memory saver
    pipe.unet.enable_gradient_checkpointing()

    # CLIP loss (freeze)
    clip_loss_func = CLIPLoss(
        device,
        lambda_direction=1.0,
        lambda_patch=0.0,
        lambda_global=0.0,
        lambda_manifold=0.0,
        lambda_texture=0.0,
        clip_model=args.clip_model,
    ).to(device)
    for p in clip_loss_func.parameters():
        p.requires_grad = False
    for p in clip_loss_func.model.parameters():
        p.requires_grad = False

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "delta.pt")

    # embeddings: uncond / pos / neg
    uncond_emb, cond_emb, neg_emb = get_three_text_embeds(
        pipe, args.src_prompt, args.neg_prompt, device
    )
    uncond_emb = uncond_emb.to(dtype=pipe.unet.dtype)
    cond_emb = cond_emb.to(dtype=pipe.unet.dtype)
    neg_emb = neg_emb.to(dtype=pipe.unet.dtype)

    # delta + hook
    C = 1280
    delta = nn.Parameter(torch.zeros((1, C, 1, 1), device=device, dtype=pipe.unet.dtype))
    injector = MidBlockInjector(pipe.unet, delta)
    injector.install()

    optim = torch.optim.Adam([delta], lr=args.lr)

    # timesteps
    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    assert 0 <= args.t_edit_idx < len(timesteps) - 1
    t0_idx = args.t0_idx if args.t0_idx >= 0 else (len(timesteps) - 1)
    t0_idx = min(t0_idx, len(timesteps) - 1)
    assert args.t_edit_idx <= t0_idx
    train_indices = list(range(args.t_edit_idx, t0_idx + 1))

    inject_steps = parse_inject_steps(args.inject_steps)
    use_inject_set = (len(inject_steps) > 0)

    print(f"[train] steps={args.steps} t_edit_idx={args.t_edit_idx} t0_idx={t0_idx} #train_steps={len(train_indices)}")
    if use_inject_set:
        print(f"[train] inject_steps={sorted(list(inject_steps))} (override range)")
    else:
        print(f"[train] inject range idx>=t_edit_idx")
    print(f"[train] asym={args.asyrp_asym} gs_pos={args.guidance_scale_pos} gs_neg={args.guidance_scale_neg}")
    print(f"[train] src='{args.src_prompt}'")
    print(f"[train] neg='{args.neg_prompt}'")
    print(f"[train] tgt(loss)='{args.tgt_prompt}'")

    for it in tqdm(range(args.iters), desc="train"):
        g = torch.Generator(device=device).manual_seed(args.seed + it)

        xT = torch.randn(
            (1, pipe.unet.config.in_channels, args.height // 8, args.width // 8),
            generator=g,
            device=device,
            dtype=pipe.unet.dtype,
        )

        lat_ref = xT.clone()
        lat_edit = xT.clone()

        optim.zero_grad(set_to_none=True)

        l1_sum = 0.0
        clip_sum = 0.0

        for idx in train_indices:
            t = timesteps[idx]
            t_prev = timesteps[idx + 1] if (idx + 1) < len(timesteps) else timesteps[idx]

            x_t_ref = lat_ref
            x_t_edit = lat_edit

            # ref eps (no grad, no injection)
            injector.enabled = False
            with torch.no_grad():
                latent_in_ref = pipe.scheduler.scale_model_input(x_t_ref, t)
                eps_ref = cfg_eps_posneg(
                    pipe.unet,
                    latent_in_ref,
                    t,
                    uncond_emb,
                    cond_emb,
                    neg_emb,
                    args.guidance_scale_pos,
                    args.guidance_scale_neg,
                )

            # injection enabled?
            if use_inject_set:
                injector.enabled = (idx in inject_steps)
            else:
                injector.enabled = (idx >= args.t_edit_idx)

            latent_in_edit = pipe.scheduler.scale_model_input(x_t_edit, t)
            eps_edit = cfg_eps_posneg(
                pipe.unet,
                latent_in_edit,
                t,
                uncond_emb,
                cond_emb,
                neg_emb,
                args.guidance_scale_pos,
                args.guidance_scale_neg,
            )

            alpha_bar_t = pipe.scheduler.alphas_cumprod[t].to(device=device, dtype=pipe.unet.dtype)
            x0_ref_t = ddim_predict_x0(x_t_ref, eps_ref, alpha_bar_t)

            if args.asyrp_asym == 1 and injector.enabled:
                x_prev_edit, x0_edit_t = ddim_asyrp_step(x_t_edit, eps_edit, eps_ref, pipe.scheduler, t, t_prev)
            else:
                x0_edit_t = ddim_predict_x0(x_t_edit, eps_edit, alpha_bar_t)
                x_prev_edit = pipe.scheduler.step(eps_edit, t, x_t_edit).prev_sample

            # update latents
            with torch.no_grad():
                lat_ref = pipe.scheduler.step(eps_ref, t, x_t_ref).prev_sample
            lat_edit = x_prev_edit

            # losses
            l1 = F.l1_loss(x0_edit_t.float(), x0_ref_t.float())

            img_ref = decode_latents_to_rgb_tensor(pipe, x0_ref_t)
            img_edit = decode_latents_to_rgb_tensor(pipe, x0_edit_t)

            clip_dir = stable_clip_directional_loss(
                clip_loss_func,
                src_img_m11=img_ref,
                src_text=args.src_prompt,
                tgt_img_m11=img_edit,
                tgt_text=args.tgt_prompt,
                eps=args.clip_cos_eps,
            )

            if args.use_log_warp == 1:
                clip_dir = clip_dir.clamp(0.0, 2.0)
                clip_dir = -torch.log((2.0 - clip_dir + 1e-6) / 2.0)

            step_loss = args.l1_w * l1 + args.clip_w * clip_dir
            step_loss = step_loss + args.l2 * (delta.float().pow(2).mean())

            # per-step backward (accumulate)
            step_loss.backward()

            # break time chain
            lat_edit = lat_edit.detach()

            # cleanup
            del img_ref, img_edit

            l1_sum += float(l1.detach())
            clip_sum += float(clip_dir.detach())

        # optimizer step per outer iter
        torch.nn.utils.clip_grad_norm_([delta], args.grad_clip)
        optim.step()

        with torch.no_grad():
            delta.clamp_(-args.delta_clip, args.delta_clip)

        if torch.isnan(delta).any():
            raise RuntimeError("delta became NaN. Try: lr↓, delta_clip↓, l2↑, clip_cos_eps↑, gs_pos↓")

        if (it % args.save_every == 0) or (it == args.iters - 1):
            with torch.no_grad():
                img_final = decode_latents_to_pil(pipe, lat_edit.float())[0]
                img_final.save(os.path.join(args.outdir, f"train_it{it:04d}_final.png"))

        if it % args.log_every == 0:
            denom = max(1, len(train_indices))
            print(f"[it {it}] l1={l1_sum/denom:.4f} clip={clip_sum/denom:.4f} |delta|={float(delta.abs().mean()):.6f}")

    torch.save({"delta": delta.detach().float().cpu(), "args": vars(args)}, ckpt_path)
    print(f"[saved] {ckpt_path}")


# -------------------------
# Inference: sweep with negative CFG
# -------------------------
@torch.no_grad()
def infer_sweep(args):
    device = torch.device(args.device)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    delta_base = ckpt["delta"]

    delta = nn.Parameter(delta_base.to(device=device, dtype=pipe.unet.dtype), requires_grad=False)
    injector = MidBlockInjector(pipe.unet, delta)
    injector.install()

    uncond_emb, cond_emb, neg_emb = get_three_text_embeds(pipe, args.src_prompt, args.neg_prompt, device)
    uncond_emb = uncond_emb.to(dtype=pipe.unet.dtype)
    cond_emb = cond_emb.to(dtype=pipe.unet.dtype)
    neg_emb = neg_emb.to(dtype=pipe.unet.dtype)

    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps

    inject_steps = parse_inject_steps(args.inject_steps)
    use_inject_set = (len(inject_steps) > 0)

    scales = args.scales
    if isinstance(scales, str):
        scales = [float(s) for s in scales.split(",")]

    os.makedirs(args.outdir, exist_ok=True)

    g = torch.Generator(device=device).manual_seed(args.seed)
    xT = torch.randn(
        (1, pipe.unet.config.in_channels, args.height // 8, args.width // 8),
        generator=g,
        device=device,
        dtype=pipe.unet.dtype,
    )

    results = []
    for s in scales:
        lat_ref = xT.clone()
        lat_edit = xT.clone()
        delta.data = delta_base.to(device=device, dtype=pipe.unet.dtype) * s

        for idx, t in enumerate(timesteps):
            t_prev = timesteps[idx + 1] if (idx + 1) < len(timesteps) else timesteps[idx]

            x_t_ref = lat_ref
            x_t_edit = lat_edit

            injector.enabled = False
            latent_in_ref = pipe.scheduler.scale_model_input(x_t_ref, t)
            eps_ref = cfg_eps_posneg(
                pipe.unet, latent_in_ref, t, uncond_emb, cond_emb, neg_emb,
                args.guidance_scale_pos, args.guidance_scale_neg
            )
            lat_ref = pipe.scheduler.step(eps_ref, t, x_t_ref).prev_sample

            if use_inject_set:
                injector.enabled = (idx in inject_steps)
            else:
                injector.enabled = (idx >= args.t_edit_idx)

            latent_in_edit = pipe.scheduler.scale_model_input(x_t_edit, t)
            eps_edit = cfg_eps_posneg(
                pipe.unet, latent_in_edit, t, uncond_emb, cond_emb, neg_emb,
                args.guidance_scale_pos, args.guidance_scale_neg
            )

            if args.asyrp_asym == 1 and injector.enabled:
                lat_edit, _ = ddim_asyrp_step(x_t_edit, eps_edit, eps_ref, pipe.scheduler, t, t_prev)
            else:
                lat_edit = pipe.scheduler.step(eps_edit, t, x_t_edit).prev_sample

        img = decode_latents_to_pil(pipe, lat_edit.float())[0]
        out_path = os.path.join(args.outdir, f"sweep_scale_{s:+.2f}.png")
        img.save(out_path)
        results.append(out_path)

    print("[saved]")
    for p in results:
        print(" ", p)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train", "infer"], required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--outdir", type=str, default="sd15_negcfg_out")
    p.add_argument("--seed", type=int, default=0)

    # SD settings
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=50)

    # pos/neg CFG scales (separate)
    p.add_argument("--guidance_scale_pos", type=float, default=3.0)
    p.add_argument("--guidance_scale_neg", type=float, default=1.0)

    # prompts
    p.add_argument("--src_prompt", type=str, default="a portrait photo of a single person, centered, head and shoulders, facing camera")
    p.add_argument("--tgt_prompt", type=str, default="a portrait photo of a single smiling person, centered, head and shoulders, facing camera")
    p.add_argument(
        "--neg_prompt",
        type=str,
        default="crowd, group, many people, multiple faces, collage, tiled, pattern, repeated, duplicated, mosaic, wallpaper, people",
    )

    # CLIP
    p.add_argument("--clip_model", type=str, default="ViT-B/32")
    p.add_argument("--clip_cos_eps", type=float, default=1e-4)

    # timestep indices
    p.add_argument("--t_edit_idx", type=int, default=22, help="editing starts from this timestep index (inclusive)")
    p.add_argument("--t0_idx", type=int, default=-1, help="train ends at this index (inclusive), -1=last")

    # injection schedule override
    p.add_argument(
        "--inject_steps",
        type=str,
        default="0-25",
        help='override injection steps by index. Examples: "22-25" or "22,24,25" or "15" or "none". If set, overrides range idx>=t_edit_idx.',
    )

    # asymmetric update
    p.add_argument("--asyrp_asym", type=int, default=1)

    # train
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--clip_w", type=float, default=1.0)
    p.add_argument("--l1_w", type=float, default=0.8)
    p.add_argument("--use_log_warp", type=int, default=0)

    # stability
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--delta_clip", type=float, default=0.0075)

    # logging/saving
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)

    # infer
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--scales", type=str, default="-2,-1,0,1,2")

    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    seed_everything(args.seed)

    if args.mode == "train":
        train(args)
    else:
        assert args.ckpt, "--ckpt is required for infer"
        infer_sweep(args)
