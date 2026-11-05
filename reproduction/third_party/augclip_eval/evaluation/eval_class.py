
import os
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import math
import random

import numpy as np
from numpy import linalg as LA

from typing import List
from typing_extensions import Literal


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf

from torchvision import transforms
import evaluation.dino.vision_transformer as vits
from evaluation.segment_consistency.bisenetv2 import BiSeNetV2

from evaluation.eval_base import BaseScore


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class CLIPScore(BaseScore):
    def __init__(
        self,
        device,
        model_name,
        w = 100,   # torchMetrics 에서는 100
    ):
        super().__init__(device, model_name)
        self.w = w
        self.score = 0
        self.n_samples = 0

    def compute(self, src_images, tgt_images, tgt_text, src_text=None):
        """Computes accumulated clip score."""
        src_image_feat = self.get_image_embeddings([src_images]) # N, C
        tgt_image_feat = self.get_image_embeddings([tgt_images]) # N, C
        text_feat = self.get_text_embeddings([tgt_text])
        if src_text:
            src_feat = self.get_text_embeddings([src_text])
            text_feat = text_feat - src_feat
        
        score = ((tgt_image_feat - src_image_feat) * text_feat).sum(axis=-1)
        score = torch.maximum(score, torch.zeros_like(score))
        return score
    
         
class RefCLIPScore(BaseScore):
    def __init__(
        self,
        device,
        model_name,
        reference: List[str],
        w = 100,   # torchMetrics 에서는 100
    ):
        super().__init__(device, model_name)
        self.w = w
        self.score = 0
        self.n_samples = 0
        self.reference = self.CLIP_text_embedding(reference)
        
    def compute(self, src_images, tgt_images, text):
        """Computes accumulated clip score."""
        # return self.score / self.n_samples
        src_image_feat = self.CLIP_image_embedding(src_images) # N, C
        tgt_image_feat = self.CLIP_image_embedding(tgt_images) # N, C
        text_feat = self.CLIP_text_embedding(text)
        
        clips = [self.w * max(sim,0) for sim in (text_feat * (tgt_image_feat - src_image_feat)).sum(axis=-1)]
        refs = [max((img * self.reference).sum(axis=-1).max(),0) for img in (tgt_image_feat - src_image_feat)]
        refclips = [(2*x*y/(x+y)).item() if (x*y) else 0 for x,y in zip(clips, refs)]
        return torch.Tensor(refclips)

class CLIPR(BaseScore):
    def __init__(
        self,
        device,
        model_name,
        captions
    ):
        super().__init__(device, model_name)
        self.score = 0
        self.n_samples = 0
        self.clip_result = []
        self.tot_captions = captions
        
    def get_mismatched_caption(self, caption):
        mismatch_captions = []
        tmp_captions = [item for item in self.tot_captions if (item != caption)]
        sampled_idx = random.sample(range(len(tmp_captions)), 99)
        mismatch_captions = [tmp_captions[idx] for idx in sampled_idx]
        return mismatch_captions
    
    def make_prediction(self, images: List[str], texts: List[str]):
        result = []
        for img, caption in zip(images, texts):
            img_feat = self.CLIP_image_embedding([img])
            
            mismatched_captions = self.get_mismatched_caption(caption)
            all_texts = [caption] + mismatched_captions
            text_feat = self.CLIP_text_embedding(all_texts)
            
            logit_scale = 1 / 0.07
            logits_per_image = logit_scale * img_feat @ text_feat.T
            clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)[0, 0]
            result.append(clip_prediction)
            
        self.clip_result = self.clip_result + result
        
    def compute_r_precision(self, candidates, num_chunks=10):
        predictions = []
        for cand in candidates:
            if cand == 0:
                predictions.append(1)
            else:
                predictions.append(0)

        num_preds = len(predictions)
        chunk_size = int(num_preds / num_chunks)

        predictions = np.array(predictions)
        np.random.shuffle(predictions)

        chunks = np.zeros(num_chunks)
        for i in range(num_chunks):
            chunks[i] = np.average(predictions[i * chunk_size : (i + 1) * chunk_size])

        return np.average(chunks), np.std(chunks)
    
    def compute(self):
        R_mean, R_std = self.compute_r_precision(self.clip_result)
        return R_mean, R_std

class SegmentConsistency:
    def __init__(
            self,
            weight_path,
            device
            ):
        self.net = BiSeNetV2(171, aux_mode='eval')
        self.net.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        self.net.eval()
        self.net.to(device)
        self.device = device
        self.palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    def _seg(self, im):
        im = ttf.to_tensor(im).unsqueeze(0).to(self.device)
        org_size = im.size()[2:] # 1, 3, h, w
        new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]
        im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
        out = self.net(im)[0]
        out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
        out = out.argmax(dim=1)
        return out
    
    def seg_accuracy(self, im_src, im_trg):
        out_src, out_trg = self._seg(im_src), self._seg(im_trg)
        flat_src, flat_trg = out_src.flatten(), out_trg.flatten()
        acc = torch.sum(torch.eq(flat_src, flat_trg)) / flat_src.shape[0] * 100
        return acc
     
class DINOAttSim: # distance, lower the better
    def __init__(
        self,
        device,
        pretrained_weights,
        arch: Literal['vit_tiny', 'vit_small', 'vit_base'] = 'vit_small',
        patch_size: int = 8,
        # wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
    ):
        self.device = device
        self.arch = arch
        self.patch_size = patch_size
        self.pretrained_weights = pretrained_weights
        
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.org_atts = []
        self.tgt_atts = []
        self.n_samples = 0
        
    def _load_model(self):
        model = vits.__dict__[self.arch](patch_size=self.patch_size, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(self.device)
        
        state_dict = torch.load(self.pretrained_weights, map_location="cpu")
        if "teacher" in state_dict:
            print(f"Take key teacher in provided checkpoint dict")
            state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights found at {} and loaded with msg: {}".format(self.pretrained_weights, msg))
        
        return model
    
    def _extract_att_map(self, images):
        # imgs = [Image.open(img_path).convert('RGB') for img_path in images]
        imgs = [self.transform(img) for img in images]
        imgs = torch.stack(imgs, dim=0)
        
        w_featmap = h_featmap = 512 // self.patch_size
        
        attentions = self.model.get_last_selfattention(imgs.to(self.device))
        
        n_imgs = attentions.shape[0]
        nh = attentions.shape[1] # number of head
        attentions = attentions[:, :, 0, 1:].reshape(n_imgs, nh, -1)
        attentions = attentions.reshape(n_imgs, nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions, scale_factor=self.patch_size, mode="nearest").cpu().numpy()
        attentions = [sum(attention[i] * 1 / nh for i in range(nh)) for attention in attentions]
        attentions = [attention / LA.norm(attention) for attention in attentions]
        
        return np.stack(attentions)
    
    def compute(self, org_image, tgt_image) -> None:
        org_att_maps = self._extract_att_map([org_image]) # N, 512, 512
        tgt_att_maps = self._extract_att_map([tgt_image]) # N, 512, 512
        diff = LA.norm(org_att_maps - tgt_att_maps) 
        return diff