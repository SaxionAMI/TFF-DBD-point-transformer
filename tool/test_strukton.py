import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize

from util.saxion_dataset import SaxionDataset

random.seed(123)
np.random.seed(123)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def main():
    global args, logger
    args = get_parser()

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        #logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        # What is this state_dict stuff???
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        #logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)

def data_load():
    ds = SaxionDataset(args.dset_path, mode="val")
    coord, feat, label = ds[args.leave_out]
    idx_data = np.arange(label.shape[0])
    return coord, feat, label, idx_data

def test(model, criterion, names):
    #logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()

    coord, feat, label, idx_data = data_load()
    pred = torch.zeros((label.size, args.classes)).cuda()

    coord = torch.FloatTensor(coord).cuda(non_blocking=True)
    feat = torch.FloatTensor(feat).cuda(non_blocking=True)
    offset = torch.IntTensor(np.cumsum(idx_data.size)).cuda(non_blocking=True)
    with torch.no_grad():
        pred = model([coord, feat, offset])  # (n, k)
    torch.cuda.empty_cache()
    #loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
    pred = pred.max(1)[1].data.cpu().numpy()
    
    acc = np.sum(label[0] == pred)/pred.size
    print(acc)

if __name__ == '__main__':
    main()
