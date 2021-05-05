# -*- coding: utf-8 -*-
# author: pinakinathc

import os, tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from network import PVSE
from options import parser
from model import train, evaluate, testing

if __name__ == '__main__':
    opt = parser.parse_args() # get parameters

    model = PVSE(opt) # Model defined
    iterations = 0

    if torch.cuda.is_available(): # False if no cuda
        multi_gpu = torch.cuda.device_count() > 1
        model = torch.nn.DataParallel(model).cuda() if multi_gpu else model.cuda()
        torch.backends.cudnn.benchmark = True

    if os.path.exists(os.path.join(opt.curr_dir, opt.ckpt)):
        checkpoint = torch.load(opt.ckpt)
        model.load_state_dict(checkpoint['model'])
        iterations = checkpoint['iterations']
        print ('Model loaded from iteration: ', iterations)

    log_writer = SummaryWriter(opt.log_file) # Keep logs

    if opt.mode == 'train':
        train(opt, model, log_writer, iterations=iterations)
    else:
        testing(opt, model)
