"""Video clip order prediction."""
import os
import math
import itertools
import argparse
import time
import random
import logging
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
import adabound
from datasets.ucf101 import UCF101VCOPDataset, UCF101VFCOPDataset, UCF101VCOPDataset_color
from models.c3d import C3D
from models.r3d import R3DNet
#from models.r21d import R2Plus1DNet
from models.r21d import R2Plus1DNet
#from models.i3d import InceptionI3d
from models.vcopn import VCOPN, VCOPN_GCN, VCOPN_GCN_R, VCOPN_GCN_R_Eight, VCOPN_GATN_R, VCOPN_GCN_randomedge, TCG_FourClip,VCOPN_GCN_R3D_R21D
from models.TCG import TCG_triple_R3D_R21D_FCA2
from lib.NCEAverage import NCEAverage, NCEAverage_ori
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion, NCESoftmaxLoss
from lib.utils import AverageMeter#, adjust_learning_rate
import ast
import warnings
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
warnings.filterwarnings("ignore")
def order_class_index(order):
    """Return the index of the order in its full permutation.
    
    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    return classes.index(tuple(order.tolist()))


def train(args, model,criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()
    running_loss = 0.0
    correct = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_contrast_frame = AverageMeter()
    losses_contrast_clip = AverageMeter()
    losses_contrast = AverageMeter()
    losses_order = AverageMeter()

    end = time.time()
    #torch.cuda.empty_cache()
    for i, data in enumerate(train_dataloader, 1):
        data_time.update(time.time() - end)
        # get inputs
        #tuple_clips, tuple_orders, tuple_clips_random, tuple_orders_random, index = data
        tuple_clips, tuple_orders, index = data
        bsz = tuple_clips.size(0)
        inputs = tuple_clips.to(device)
        #inputs_random = tuple_clips_random.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        index = index.to(device)
        # zero the parameter gradients
        #optimizer.zero_grad()
        # forward and backward
        contrast_loss_1, contrast_loss_2, contrast_loss_3,contrast_loss_clip,outputs = model(inputs,tuple_orders) # return logits here
        loss_contrast_frame=(contrast_loss_1.sum()+contrast_loss_2.sum()+contrast_loss_3.sum())/(3*args.bs)
        loss_contrast_clip=contrast_loss_clip.sum()/args.bs
        loss_order = criterion(outputs, targets)
        loss=args.weight_contrast_frame*loss_contrast_frame+args.weight_contrast_clip*loss_contrast_clip+args.weight_order*loss_order
        #loss=args.weight_order*loss_order
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        #losses_contrast.update(args.weight_contrast*loss_contrast.item(), bsz)
        losses_contrast_frame.update(args.weight_contrast_frame*loss_contrast_frame.item(), bsz)
        losses_contrast_clip.update(args.weight_contrast_clip*loss_contrast_clip.item(), bsz)
        losses_order.update(args.weight_order*loss_order.item(), bsz)
        losses.update(loss.item(), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        # compute loss and acc
        running_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print statistics and write summary every N batch
    
        # print info
        if i % 20 == 0:
            log_str=('Train: [{0}/{1}][{2}/{3}]  '
                #'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                #'DT {data_time.val:.3f} ({data_time.avg:.3f}) '
                'loss_contrast_frame {loss_contrast_frame.val:.3f} ({loss_contrast_frame.avg:.3f}) '
                'loss_contrast_clip {loss_contrast_clip.val:.3f} ({loss_contrast_clip.avg:.3f}) '
                'loss_order {loss_order.val:.3f} ({loss_order.avg:.3f}) '
                'loss {loss.val:.3f} ({loss.avg:.3f})'
                .format(epoch, args.epochs, i , len(train_dataloader), loss_contrast_frame=losses_contrast_frame, loss_contrast_clip=losses_contrast_clip,loss_order=losses_order, loss=losses))
            logging.info(log_str)

        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            avg_acc = correct / (args.pf * args.bs)
            logging.info('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch-1)*len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0
    # summary params and grads per eopch
    #for name, param in model.named_parameters():
    #    writer.add_histogram('params/{}'.format(name), param, epoch)
    #    writer.add_histogram('grads/{}'.format(name), param.grad, epoch)
    avg_loss = running_loss / len(train_dataloader)
    return avg_loss

def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        tuple_clips, tuple_orders,_ = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        _,_,_,_,outputs = model(inputs,tuple_orders) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    #writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    #writer.add_scalar('val/Accuracy', avg_acc, epoch)
    logging.info('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss, avg_acc


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    logging.info('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Contrast Graph')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d/i3d/r3d50/s3d')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=8, help='interval')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, default='log', help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=24, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=32, help='seed for initializing training.')
    parser.add_argument('--softmax', type=ast.literal_eval, default=True)
    parser.add_argument('--nce_k', type=int, default=512)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=512, help='dim of feat for inner product')
    parser.add_argument('--weight_contrast_frame', type=float, default=1)
    parser.add_argument('--weight_contrast_clip', type=float, default=1)
    parser.add_argument('--weight_order', type=float, default=1)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--output_dir', type=str, default='log/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    torch.backends.cudnn.benchmark = True
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    if args.model == 'c3d':
        base = C3D(with_classifier=False).to(device)
    elif args.model == 'r3d':
        base = R3DNet(layer_sizes=(3, 4, 6, 3), with_classifier=False).to(device)
    elif args.model == 'r21d':   
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False).to(device)
    elif args.model == 'i3d':   
        base = InceptionI3d(final_endpoint='Logits', with_classifier=False).to(device)
    elif args.model == 'r3d50':   
        base = resnet50(sample_size=112,sample_duration=16,with_classifier=False, return_conv=False).to(device)
    elif args.model == 's3d':   
        base = S3D(gating = True, with_classifier=False, return_conv=False).to(device)

    tcg = TCG_triple_R3D_R21D_FCA2(base_network=base, feature_size=512, tuple_len=args.tl).cuda()

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            pretrained_weights = torch.load(args.ckpt)['model']
            tcg.load_state_dict({k.replace('module.',''):v for k,v in pretrained_weights.items()},strict=True)
            #tcg.load_state_dict(torch.load(args.ckpt))
            log_dir = os.path.dirname(args.ckpt)
        else:
            if args.desp:
                exp_name = 'UCF101_R3D50_TCG112_{}_cl{}_it{}_tl{}_{}_{}'.format(args.model, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = 'UCF101_R3D50_TCG112_{}_cl{}_it{}_tl{}_{}'.format(args.model, args.cl, args.it, args.tl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
        writer = SummaryWriter(log_dir)
        if torch.cuda.device_count() > 1:
            tcg = torch.nn.DataParallel(tcg, device_ids=[0,1,2,3,4,5,6,7]).cuda()
        log_format = '%(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
        fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),  # smaller edge to 128
            transforms.RandomCrop(112),
            #transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])
        color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        #train_dataset = UCF101VCOPDataset_color('data/ucf101', args.cl, args.it, args.tl, True, train_transforms,color_jitter_=color_jitter)
        train_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, True, train_transforms)
        # split val for 800 videos
        train_dataset, val_dataset = random_split(train_dataset,(len(train_dataset)-800, 800))
        logging.info('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

        if args.ckpt:
            pass
        else:
            # save graph and clips_order samples
            for data in train_dataloader:
                #tuple_clips, tuple_orders, tuple_clips_random, tuple_orders_random,idx = data
                tuple_clips, tuple_orders,idx = data
                for i in range(args.tl):
                    writer.add_video('train/tuple_clips', tuple_clips[:, i, :, :, :, :], i, fps=8)
                    writer.add_text('train/tuple_orders', str(tuple_orders[:, i].tolist()), i)
                tuple_clips = tuple_clips.to(device)
                #writer.add_graph(tcg, tuple_clips)
                break
            # save init params at step 0
            for name, param in tcg.named_parameters():
                writer.add_histogram('params/{}'.format(name), param, 0)

        n_data = train_dataset.__len__()
        
        torch.backends.cudnn.benchmark = True

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(tcg.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        #optimizer = optim.Adam(tcg.parameters(), lr=args.lr, weight_decay=args.wd)
        #optimizer = adabound.AdaBound(tcg.parameters(), lr=args.lr, final_lr=0.1, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=50, factor=0.1)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)

        prev_best_val_loss = float('inf')
        prev_best_val_acc = 0.0
        prev_best_loss_model_path = None
        prev_best_acc_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train_loss = train(args, tcg, criterion, optimizer, device, train_dataloader, writer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            val_loss, val_acc = validate(args, tcg, criterion, device, val_dataloader, writer, epoch)
            scheduler.step(val_loss)         
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            # save model every 20 epoches
            if epoch % 10 == 0:
                state = {
                'opt': args,
                'model': tcg.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                }
                torch.save(state, os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
                del state
            # save model for the best val
            if val_loss < prev_best_val_loss:
                state = {
                'opt': args,
                'model': tcg.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                }
                model_path = os.path.join(log_dir, 'best_loss_model_{}.pt'.format(epoch))
                torch.save(state, model_path)
                prev_best_val_loss = val_loss
                if prev_best_loss_model_path:
                    os.remove(prev_best_loss_model_path)
                prev_best_loss_model_path = model_path
                del state
            if val_acc > prev_best_val_acc:
                state = {
                'opt': args,
                'model': tcg.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                }
                model_path = os.path.join(log_dir, 'best_acc_model_{}.pt'.format(epoch))
                torch.save(state, model_path)
                prev_best_val_acc = val_acc
                if prev_best_acc_model_path:
                    os.remove(prev_best_acc_model_path)
                prev_best_acc_model_path = model_path
                del state
            #torch.cuda.empty_cache()
            #scheduler.step()

    elif args.mode == 'test':  ########### Test #############
        tcg.load_state_dict(torch.load(args.ckpt))
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])
        test_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, False, test_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, tcg, criterion, device, test_dataloader)
