from models.sae.sae import Autoencoder, ACTIVATIONS_CLASSES
import torch
import argparse
import os
parser = argparse.ArgumentParser(description=globals()['__doc__'])
parser.add_argument('--)
args = parser.parse_args()
sae_ckpt_path = os.path.join(
)
state_dict = torch.load(sae_ckpt_path)
sae = Autoencoder.from_state_dict(state_dict)