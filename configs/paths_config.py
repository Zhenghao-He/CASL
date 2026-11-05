import os
from pathlib import Path

# Set CASL_DATA_ROOT when datasets live outside the historical sibling layout.
DATA_ROOT = Path(os.environ.get('CASL_DATA_ROOT', Path(__file__).resolve().parents[2]))

DATASET_PATHS = {
	'FFHQ': str(DATA_ROOT / 'ffhq-dataset/images1024x1024/00000'),
	'CelebA_HQ_P2': str(DATA_ROOT / 'CelebA-HQ/data1024'),
	'AFHQ': str(DATA_ROOT / 'stargan-v2/data'),
	'LSUN': str(DATA_ROOT / 'lsun/church'),
    'IMAGENET': '/hdd1/datasets/imagenet/',
	'CUSTOM': '/hdd1/custom/',
	'CelebA_HQ_Dialog': '/hdd1/datasets/img_align_celeba/',
	'MetFACE': '/hdd1/datasets/metfaces/',
}

MODEL_PATHS = {
	'AFHQ': "pretrained/afhqdog_p2.pt",
	'FFHQ': "pretrained/ffhq_p2.pt",
    'LSUN': "pretrained/ddpm_church.pt",
    'CelebA_HQ_P2': "pretrained/celebahq_p2.pt",
	'ir_se50': 'pretrained/model_ir_se50.pth',
    'IMAGENET': "pretrained/256x256_diffusion_uncond.pt",
	'shape_predictor': "pretrained/shape_predictor_68_face_landmarks.dat.bz2",
	'MetFACE' : "pretrained/metface_p2.pt",
}
