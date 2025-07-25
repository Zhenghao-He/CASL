import torch as th
import torch.nn as nn
class DirectionDecoder(nn.Module):
    def __init__(self, channels, out_channels, mask_topk=3, is_mask=False):
        super().__init__()
        self.out_channels = out_channels
        # self.mask_logits = nn.Parameter(th.zeros(channels))
        # self.mask_logits = nn.Parameter(th.randn(channels) * 0.01 - 4.0)

        # self.mask_topk = mask_topk
        # self.is_mask = is_mask
        # self.mlp = nn.Sequential(

        #     nn.Linear(channels, hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim, out_channels)
        # )
        # from utils.l0_regularization import L0Dense
        # self.mlp = L0Dense(in_features=channels, out_features=out_channels, bias=True, weight_decay=100000., droprate_init=0.1, temperature=2./3.,
        #          lamba=10., local_rep=False)
        self.mlp = nn.Linear(channels, out_channels)

    def hard_topk_mask(self, logits, k):
        # logits: [d]
        topk = th.topk(logits, k=k)
        hard_mask = th.zeros_like(logits)
        hard_mask[topk.indices] = 1.0
        return hard_mask

    def forward(self, z):
        """
        z: Tensor of shape (H*W, C_in)
        Returns: Tensor of shape (1, C_out, H, W)
        """

        # soft_mask = th.sigmoid(self.mask_logits)
        # hard_mask = self.hard_topk_mask(logits=self.mask_logits, k=self.mask_topk)  # Assuming k=10 for hard mask
        # mask = hard_mask.detach() + soft_mask - soft_mask.detach()
        # if self.is_mask:
        #     # print("Using mask in DirectionDecoder")
        #     out = self.mlp(z * mask)          # (H*W, C_out)
        # else:
        out = self.mlp(z)                 # (H*W, C_out)
        out = out.transpose(0, 1)  # (C_out, H*W)
        out = out.unsqueeze(0)  # (1, C_out, H*W)
        # soft_mask = th.sigmoid(self.mlp.qz_loga)
        return out, None
    

class XBlock(nn.Module):
    def __init__(self, sae_ckpt_path, out_channels, mask_topk=10, is_mask=False):
        super().__init__()

        # === 1. 加载SAE模型 ===
        from models.sae.sae import Autoencoder, ACTIVATIONS_CLASSES
        state_dict = th.load(sae_ckpt_path)
        self.sae = Autoencoder.from_state_dict(state_dict)
        self.sae.eval()
        self.is_mask = is_mask
        # === 2. 构建DirectionDecoder ===
        self.decoder = DirectionDecoder(
            channels=self.sae.encoder.out_features,  
            out_channels=out_channels,
            mask_topk=mask_topk,
            is_mask=self.is_mask
        )

    def forward(self, x, t, perturb=False, topk_indices=None, scale_to_analyze=1.0):
        """
        x: Tensor of shape (B, C_in, H, W)
        Return: Tensor of shape (B, C_out, H, W)
        """
        C= x.shape[1]
        x_flat = x.squeeze(0).permute(1, 2, 0).reshape(-1, C)
        ts = t.expand(x_flat.shape[0])  # 假设 t.shape == (1, C)
        ts = ts.long()  # 确保 ts 是 LongTensor
        # import pdb; pdb.set_trace()
        with th.no_grad():
            z, info = self.sae.encode(x_flat,ts)  # (B, H*W, latent_dim)
        if perturb:
            # import pdb; pdb.set_trace()
            if self.is_mask:
                hard_mask = self.decoder.hard_topk_mask(logits=self.decoder.mask_logits, k=self.decoder.mask_topk)
                idx = th.nonzero(hard_mask).squeeze()
                z[:, idx] *= scale_to_analyze
                # import pdb; pdb.set_trace()
            else:
                # 先置零，再单独放大Top-K
                # import pdb; pdb.set_trace()
                # to_remove = th.tensor([32088, 3644])
                # remain_idx = ~th.isin(topk_indices, to_remove)
                # topk_indices = topk_indices[remain_idx]
                # topk_indices = th.tensor([32088, 23984,  3644, 30090, 20018, 20188,  5753,   992, 10111, 19893])
                # topk_indices = topk_indices[2:]
                z_sparse = th.zeros_like(z)
                z_sparse[:, topk_indices] = z[:, topk_indices] * scale_to_analyze
                z = z_sparse
                # z[:, topk_indices] *= scale_to_analyze
        delta_h, soft_mask = self.decoder(z)  # (B, C_out, H, W)
        return delta_h, soft_mask