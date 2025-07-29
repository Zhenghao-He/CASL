DATASET_PATHS = {
	'FFHQ': '/p/realai/zhenghao/LoRA_SAE/ffhq-dataset/images1024x1024/00000/',
	'CelebA_HQ_P2': '/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/data1024',
	'AFHQ': '/p/realai/zhenghao/LoRA_SAE/stargan-v2/data/',
	'LSUN':  '/hdd1/datasets/lsun/',
    'IMAGENET': '/hdd1/datasets/imagenet/',
	'CUSTOM': '/hdd1/custom/',
	'CelebA_HQ_Dialog': '/hdd1/datasets/img_align_celeba/',
	'MetFACE': '/hdd1/datasets/metfaces/',
}

MODEL_PATHS = {
	'AFHQ': "pretrained/afhqdog_p2.pt",
	'FFHQ': "pretrained/ffhq_p2.pt",
    'CelebA_HQ_P2': "pretrained/celebahq_p2.pt",
	'ir_se50': 'pretrained/model_ir_se50.pth',
    'IMAGENET': "pretrained/256x256_diffusion_uncond.pt",
	'shape_predictor': "pretrained/shape_predictor_68_face_landmarks.dat.bz2",
	'MetFACE' : "pretrained/metface_p2.pt",
}
