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
from util.common_util import AverageMeter, intersectionAndUnionGPU, check_makedirs
from util.voxelize import voxelize
from util.data_util import collate_fn

from util.saxion_dataset_val import SaxionDatasetVal

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

    ds = SaxionDatasetVal(args.dset_path, npoints=4096, idx=args.leave_out)
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    test(model, criterion, ds.get_labels(), ds_loader)

def test(model, criterion, labels, ds_loader):
    #logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()

    preds = np.array([]) # All predictions, for each point
    for i, (coord, feat, target, offset) in enumerate(ds_loader):  # (n, 3), (n, c), (n), (b)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        pred = torch.zeros((target.shape[0], args.classes)).cuda()

        with torch.no_grad():
            pred = model([coord, feat, offset])  # (n, k)

        torch.cuda.empty_cache()
        #loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
        pred = pred.max(1)[1].data.cpu().numpy()
        offset = offset.data.cpu().numpy()
        offset = np.insert(offset, 0, 0)
        offset = offset[:-1] # Remove last element
        preds = np.hstack([preds, pred[offset]]) if preds.size else pred[offset]

        #print("pred: {:d}".format(pred[0]))
        #if i==100:
        #    break

    iou = np.zeros(args.classes)
    #labels = labels[0:101*32]
    for i in np.arange(args.classes):
        num = np.sum(np.logical_and(preds==i,labels==i))
        den = np.sum(np.logical_or(preds==i,labels==i))
        iou[i] = num/(den + 1e-10)
        print("Class {:d}: {:02f}".format(i,iou[i]))

    print("mIoU: {:.2f}".format(np.mean(iou)))

    with open("lo_{:02d}.pkl".format(args.leave_out), "wb") as f:
        pickle.dump([preds, labels], f)

if __name__ == '__main__':
    main()
