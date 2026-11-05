import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import clip
import numpy as np

from PIL import Image
import numpy as np
from sklearn.linear_model import SGDClassifier

class BaseScore():
    def __init__(
        self,
        device,
        model_name,
        penalty='l2',
    ):
        self.device = device
        self.model_name = model_name
        self.model, self.processor = clip.load(model_name, device=device)
        self.svc_model = SGDClassifier(penalty=penalty, loss="hinge")

    @torch.no_grad()
    def get_text_embeddings(self, texts:list, text_batch_size=30, normalize=True):
        num_text = len(texts)
        text_embeds = []
        tqdm_loader = range(0, num_text, text_batch_size)
        for i in tqdm_loader:
            text = texts[i: min(num_text, i+text_batch_size)]
            text_input = clip.tokenize(text).to(self.device) 
            text_feats = self.model.encode_text(text_input)
            if normalize:
                text_feats = F.normalize(text_feats,dim=-1)      
            text_embeds.append(text_feats)   
        
        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    @torch.no_grad()
    def get_image_embeddings(self, images, normalize=True):
        if self.processor is not None:
            images = [torch.Tensor(self.processor(image)).unsqueeze(0) for image in images]
        
        image_feats = torch.cat([self.model.encode_image(im.to(self.device)) for im in images], dim=0)
        if normalize:
            image_feats = F.normalize(image_feats, dim=-1)
        return image_feats
    
    def clip_score(self, img_paths:List[str], texts:List[str]):
        text_feat = self.get_text_embeddings(texts)
        imgs = [Image.open(im_path) for im_path in img_paths]
        img_feat = self.get_image_embeddings(imgs)
        sim = (text_feat * img_feat).sum(axis=-1)
        return sim

class AugCLIP(BaseScore):
    def __init__(
        self,
        device,
        model_name,
        penalty='l2'
    ):
        super().__init__(device, model_name, penalty)
        self.minmax = lambda x: (x - x.min()) / (x.max() - x.min())

    def cs(self, x, y):
        normalize = lambda x: F.normalize(x, p=2, dim=-1)
        check_2d = lambda x: x if len(x.shape) == 2 else x.unsqueeze(0)
        ip = lambda x, y: torch.einsum("ij, kj -> ik", x, y)
        x, y = map(check_2d, map(normalize, [x, y])) # i x k
        return ip(x, y) # i x k

    #! wx + b = 0, a: ground truth, m: manipulated
    def compute_projection_and_angle(self, gt, w, b):
        check_numpy = lambda x: x.detach().cpu().numpy() if not isinstance(x, np.ndarray) else x
        reshape_1d = lambda x: x.reshape(-1) if len(x.shape) > 1 else x
        gt, w, b = map(reshape_1d, map(check_numpy, [gt, w, b]))

        # Compute the projection of a onto the plane defined by w and b
        w_norm = np.linalg.norm(w)  # Norm of the normal vector w
        if w_norm == 0:
            return None
        
        # Compute the projection p of a onto the plane
        gt_dot_w = gt @ w.T
        projected = (gt_dot_w + b) / (w_norm ** 2) * w
        return torch.Tensor(projected).to(self.device).half()
    
    def get_rel1_weight(self, src_dist, trg_dist, **kwargs):
        src_dist = F.normalize(src_dist, dim=-1)
        trg_dist = F.normalize(trg_dist, dim=-1)
        cov_xx = src_dist @ src_dist.T
        cov_yy = trg_dist @ trg_dist.T
        cov_xy = src_dist @ trg_dist.T
    
        src_weight = torch.mean(cov_xx, dim=-1) - torch.mean(cov_xy, dim=-1)
        trg_weight = torch.mean(cov_yy, dim=-1) - torch.mean(cov_xy.T , dim=-1)
        
        assert not (torch.isnan(src_weight).any() and torch.isnan(trg_weight).any()), f"Weight contains NaN values"
        src_weight = (src_weight - src_weight.min()) / (src_weight.max() - src_weight.min())
        trg_weight = (trg_weight - trg_weight.min()) / (trg_weight.max() - trg_weight.min())
        return src_weight.detach().cpu().numpy(), trg_weight.detach().cpu().numpy()