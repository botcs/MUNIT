#!/usr/bin/env python
# coding: utf-8

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
from torch.nn.functional import interpolate
import torchvision.utils as vutils
import sys
import torch
import os
from glob import glob
import tqdm
import imagesize
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image dir")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

transform = transforms.Compose([transforms.Resize(new_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

style_rand = None
if opts.num_style > 0:
    style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())


def process_input_image(input_path):
    with torch.no_grad():
        # input_fname = os.path.basename(input_path)
        input_fname = input_path[len(opts.input):]
        image = Variable(transform(Image.open(input_path).convert('RGB')).unsqueeze(0).cuda())
        width, height = imagesize.get(input_path)
        style_image =  None

        # Start testing
        content, _ = encode(image)

        if opts.trainer == 'MUNIT':
            if opts.num_style < 0:
                for j in range(abs(opts.num_style)):
                    style = Variable(torch.randn(1, style_dim, 1, 1).cuda())
                    outputs = decode(content, style)
                    outputs = (outputs + 1) / 2.
                    outputs = interpolate(outputs, [height, width], mode="bilinear")
                    path = os.path.join(opts.output_folder, "random-style", f"{j:03d}", input_fname)
                    dirname = os.path.dirname(path)
                    os.makedirs(dirname, exist_ok=True)
                    vutils.save_image(outputs.data, path, padding=0, normalize=True)
            else:
                style = style_rand
                for j in range(opts.num_style):
                    s = style[j].unsqueeze(0)
                    outputs = decode(content, s)
                    outputs = (outputs + 1) / 2.
                    outputs = interpolate(outputs, [height, width])
                    path = os.path.join(opts.output_folder, f"{j:03d}", input_fname)
                    dirname = os.path.dirname(path)
                    os.makedirs(dirname, exist_ok=True)
                    vutils.save_image(outputs.data, path, padding=0, normalize=True)


for img_path in tqdm.tqdm(glob(f"{opts.input}/**/*.jpg", recursive=True) + glob(f"{opts.input}/**/*.png", recursive=True)):
    process_input_image(img_path)




