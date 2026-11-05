import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
from tqdm import tqdm
np.random.seed(123)

from bisenetv2 import BiSeNetV2

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)

def seg_accuracy(src_path, trg_path, save=False, out_path='test', out_name=''):
    # prepare data
    to_tensor = ToTensor(
        mean=(0.3257, 0.3690, 0.3223),
        std=(0.2112, 0.2148, 0.2115),
    )
    
    im_src, im_trg = cv2.imread(src_path)[:, :, ::-1], cv2.imread(trg_path)[:, :, ::-1]
    
    def seg(im):
        im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
        org_size = im.size()[2:]
        new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]
        im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
        out = net(im)[0]
        out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
        out = out.argmax(dim=1)
        return out

    out_src, out_trg = seg(im_src), seg(im_trg)
    flat_src, flat_trg = out_src.flatten(), out_trg.flatten()
    acc = torch.sum(torch.eq(flat_src, flat_trg)) / flat_src.shape[0] * 100
    # visualize
    if save:
        if not out_name:
            out_name = src_path.split('/')[-1].split('.')[0]
        out_src = out_src.squeeze().detach().cpu().numpy()
        pred_src = palette[out_src]
        out_trg = out_trg.squeeze().detach().cpu().numpy()
        pred_trg = palette[out_trg]
        cv2.imwrite(f'./output/{out_path}/{out_name}_src.jpg', pred_src)
        cv2.imwrite(f'./output/{out_path}/{out_name}_trg.jpg', pred_trg)
        
    return acc
        
if __name__=="__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', type=str, default='/home/BiSeNet/model_final_v2_coco.pth',) # BiSeNetv2 model path
    parse.add_argument('--src-path', type=str, default='',) # source dataset directory
    parse.add_argument('--trg-path', type=str, default='',) # target dataset directory
    parse.add_argument('--out-name', type=str, default='',) # output json file name
    parse.add_argument('--seg-save', type=bool, default=False,) 
    args = parse.parse_args()

    net = BiSeNetV2(171, aux_mode='eval')
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
    net.eval()
    net.cuda()

    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    assert args.src_path and args.trg_path
    acc_dict = dict()
    src_img_list = sorted(os.listdir(args.src_path))
    os.makedirs(f'./output/{args.out_name}', exist_ok=True)

    for src in tqdm(src_img_list):
        src_img_path = os.path.join(args.src_path, src)
        trg_img_path = os.path.join(args.trg_path, src) # assume that src & trg names are same
        
    
    # print(acc_dict)
    with open(f'./output/{args.out_name}.json', 'w') as f:
        json.dump(acc_dict, f, indent=2)