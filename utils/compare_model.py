from models.sae.sae import Autoencoder
import torch as th
from ..diffusion_latent import Asyrp
activation = 'ReLU'  # or 'L0'
k = 10  # TopK value, only used if activation is 'TopK'
alpha = 4  # L0 regularization parameter, only used if activation is 'L0'
scale = 32  # Scale parameter for the SAE model
sae_ckpt_path = f"sae/sae_model_{activation}{k if activation == 'TopK' else None}_alpha{alpha}_scale{scale}.pth"
print(f"Loading SAE model from {sae_ckpt_path}...")
state_dict = th.load(sae_ckpt_path)
sae = Autoencoder.from_state_dict(state_dict)
sae.eval()

model = Asyrp.load_pretrained_model()
# model.setattr_edited_sae(sae_save_name) 
model.setattr_layers(nums=self.args.get_h_num, 
                mask_topk=self.args.mask_top, 
                activation=self.args.activation, 
                k=self.args.k, 
                alpha=self.args.alpha, 
                scale=self.args.scale,
                is_mask=self.args.masked_xblock)
model.to(self.device)
for i in range(self.args.get_h_num):                   
    get_h = getattr(model, f"layer_{i}")
    state = th.load(save_name)[f"{i}"]
    del state["sae.activation_state_dict"]
    get_h.load_state_dict(state)