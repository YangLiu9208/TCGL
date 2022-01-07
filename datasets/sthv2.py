"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile
#import sh
import numpy as np
#import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from datasets.data_parser import WebmDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets.transforms_video import *
from PIL import Image
from decord import VideoReader
from decord import cpu,gpu

#ffprobe = sh.ffprobe.bake('-v', 'error', '-show_entries',
#                          'format=start_time,duration')

#FRAMERATE = 12  # default value


class SthV2Dataset_train(Dataset):
    """K400 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, json_file_input, json_file_labels,clip_len, train=True, transforms_=None, test_sample_num=10, is_test=False):
        self.dataset_object = WebmDataset(json_file_input, json_file_labels,
                                          root_dir, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root_dir
        self.clip_len = clip_len
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        item = self.json_data[idx]

        #videodata = VideoReader(item.path,ctx=cpu(0)) 
        videodata = skvideo.io.vread(item.path)
        length, height, width, channel = videodata.shape
        #length, height, width, channel = videodata.shape
        #length = len(videodata)
        #height = videodata[0].shape[0]
        #width = videodata[0].shape[1]
        #channel = videodata[0].shape[2]
        class_idx=self.classes_dict[item.label]
        # random select a clip for train
        if self.train:
            if length<self.clip_len and length>=self.clip_len/2:
                videodata=torch.from_numpy(videodata)
                videodata=torch.cat([videodata,videodata],0)
                clip_start = random.randint(0, 2*length - self.clip_len)  
                clip = videodata[clip_start: clip_start + self.clip_len]
            elif length<self.clip_len and length<self.clip_len/2:
                videodata=torch.from_numpy(videodata)
                videodata=torch.cat([videodata,videodata,videodata],0)
                clip_start = random.randint(0, 3*length - self.clip_len)  
                clip = videodata[clip_start: clip_start + self.clip_len]
            else:
                clip_start = random.randint(0, length - self.clip_len)      
                clip = videodata[clip_start: clip_start + self.clip_len]
            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                #frequency
                clip_mean=torch.mean(clip, 1, keepdim=True)
                clip=clip-clip_mean
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            if length<self.clip_len and length>=self.clip_len/2:
                videodata=torch.from_numpy(videodata)
                videodata=torch.cat([videodata,videodata],0)
                for i in np.linspace(self.clip_len/2, 2*length-self.clip_len/2, self.test_sample_num):
                    clip_start = int(i - self.clip_len/2)
                    clip = videodata[clip_start: clip_start + self.clip_len]
                    if self.transforms_:
                        trans_clip = []
                        # fix seed, apply the sample `random transformation` for all frames in the clip 
                        seed = random.random()
                        for frame in clip:
                            random.seed(seed)
                            frame = self.toPIL(frame) # PIL image
                            frame = self.transforms_(frame) # tensor [C x H x W]
                            trans_clip.append(frame)
                        # (T x C X H x W) to (C X T x H x W)
                        clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                        #frequency
                        clip_mean=torch.mean(clip, 1, keepdim=True)
                        clip=clip-clip_mean
                    else:
                        clip = torch.tensor(clip)
                    all_clips.append(clip)
                    all_idx.append(torch.tensor(int(class_idx)))

                return torch.stack(all_clips), torch.tensor(int(class_idx))
            elif length<self.clip_len and length<self.clip_len/2:
                videodata=torch.from_numpy(videodata)
                videodata=torch.cat([videodata,videodata,videodata],0)
                for i in np.linspace(self.clip_len/2, 3*length-self.clip_len/2, self.test_sample_num):
                    clip_start = int(i - self.clip_len/2)
                    clip = videodata[clip_start: clip_start + self.clip_len]
                    if self.transforms_:
                        trans_clip = []
                        # fix seed, apply the sample `random transformation` for all frames in the clip 
                        seed = random.random()
                        for frame in clip:
                            random.seed(seed)
                            frame = self.toPIL(frame) # PIL image
                            frame = self.transforms_(frame) # tensor [C x H x W]
                            trans_clip.append(frame)
                        # (T x C X H x W) to (C X T x H x W)
                        clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                        #frequency
                        clip_mean=torch.mean(clip, 1, keepdim=True)
                        clip=clip-clip_mean
                    else:
                        clip = torch.tensor(clip)
                    all_clips.append(clip)
                    all_idx.append(torch.tensor(int(class_idx)))

                return torch.stack(all_clips), torch.tensor(int(class_idx))
            else:
                clip_start = random.randint(0, length - self.clip_len)      
                clip = videodata[clip_start: clip_start + self.clip_len]
                for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                    clip_start = int(i - self.clip_len/2)
                    clip = videodata[clip_start: clip_start + self.clip_len]
                    if self.transforms_:
                        trans_clip = []
                        # fix seed, apply the sample `random transformation` for all frames in the clip 
                        seed = random.random()
                        for frame in clip:
                            random.seed(seed)
                            frame = self.toPIL(frame) # PIL image
                            frame = self.transforms_(frame) # tensor [C x H x W]
                            trans_clip.append(frame)
                        # (T x C X H x W) to (C X T x H x W)
                        clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                        #frequency
                        clip_mean=torch.mean(clip, 1, keepdim=True)
                        clip=clip-clip_mean
                    else:
                        clip = torch.tensor(clip)
                    all_clips.append(clip)
                    all_idx.append(torch.tensor(int(class_idx)))

                return torch.stack(all_clips), torch.tensor(int(class_idx))
    def __len__(self):
        return len(self.json_data)

class SthV2Dataset_val(Dataset):
    """K400 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, json_file_input, json_file_labels,clip_len, val=True, transforms_=None, test_sample_num=10, is_test=False):
        self.dataset_object = WebmDataset(json_file_input, json_file_labels,
                                          root_dir, is_test=is_test)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root_dir
        self.clip_len = clip_len
        self.val = val
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        item = self.json_data[idx]

        videodata = skvideo.io.vread(item.path)
        length, height, width, channel = videodata.shape
        #length, height, width, channel = videodata.shape
        #length = len(videodata)
        #height = videodata[0].shape[0]
        #width = videodata[0].shape[1]
        #channel = videodata[0].shape[2]
        class_idx=self.classes_dict[item.label]
        # random select a clip for train
        if self.val:
            if length<self.clip_len and length>=self.clip_len/2:
                videodata=torch.from_numpy(videodata)
                videodata=torch.cat([videodata,videodata],0)
                clip_start = random.randint(0, 2*length - self.clip_len)  
                clip = videodata[clip_start: clip_start + self.clip_len]
            elif length>=self.clip_len/3 and length<self.clip_len/2:
                videodata=torch.from_numpy(videodata)
                videodata=torch.cat([videodata,videodata,videodata],0)
                clip_start = random.randint(0, 3*length - self.clip_len)  
                clip = videodata[clip_start: clip_start + self.clip_len]
            elif length>=self.clip_len/4 and length<self.clip_len/3:
                videodata=torch.from_numpy(videodata)
                videodata=torch.cat([videodata,videodata,videodata,videodata],0)
                clip_start = random.randint(0, 4*length - self.clip_len)  
                clip = videodata[clip_start: clip_start + self.clip_len]
            else:
                clip_start = random.randint(0, length - self.clip_len)      
                clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                #frequency
                clip_mean=torch.mean(clip, 1, keepdim=True)
                clip=clip-clip_mean
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                clip_start = int(i - self.clip_len/2)
                clip = videodata[clip_start: clip_start + self.clip_len]
                if self.transforms_:
                    trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip 
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame) # PIL image
                        frame = self.transforms_(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                    #frequency
                    clip_mean=torch.mean(clip, 1, keepdim=True)
                    clip=clip-clip_mean
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            return torch.stack(all_clips), torch.tensor(int(class_idx))
    def __len__(self):
        return len(self.json_data)

class Sthv2VCOPDataset_train(Dataset):
    """Sthv2 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        if self.train:
            vcop_train_split_name = 'vcop_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_train_split_path = os.path.join("/data0/liuyang/Something-something-v2/", vcop_train_split_name)
            self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
        else:
            vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_test_split_path = os.path.join("/data0/liuyang/Something-something-v2/", vcop_test_split_name)
            self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx].split()[0]
        else:
            videoname = self.test_split[idx].split()[0]
        
        filename = os.path.join(self.root_dir, videoname)#.replace('\\', '/')
        videodata = skvideo.io.vread(filename)
        #videodata = VideoReader(filename,ctx=cpu(0)) 
        length, height, width, channel = videodata.shape
        #length = len(videodata)
        #height = videodata[0].shape[0]
        #width = videodata[0].shape[1]
        #channel = videodata[0].shape[2]

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, abs(length - self.tuple_total_frames))
        else:
            random.seed(idx)
            tuple_start = random.randint(0, abs(length - self.tuple_total_frames))

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(tuple_order), idx

class Sthv2VCOPDataset_val(Dataset):
    """Sthv2 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval, tuple_len, val=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.val = val
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        if self.val:
            vcop_train_split_name = 'vcop_val_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_train_split_path = os.path.join("/data0/liuyang/Something-something-v2/", vcop_train_split_name)
            self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
        else:
            vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_test_split_path = os.path.join("/data0/liuyang/Something-something-v2/", vcop_test_split_name)
            self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]

    def __len__(self):
        if self.val:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.val:
            videoname = self.train_split[idx].split()[0]
        else:
            videoname = self.test_split[idx].split()[0]
        
        filename = os.path.join(self.root_dir, videoname)#.replace('\\', '/')
        videodata = skvideo.io.vread(filename)
        #videodata = VideoReader(filename,ctx=cpu(0)) 
        length, height, width, channel = videodata.shape
        #length = len(videodata)
        #height = videodata[0].shape[0]
        #width = videodata[0].shape[1]
        #channel = videodata[0].shape[2]

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.val:
            tuple_start = random.randint(0, abs(length - self.tuple_total_frames))
        else:
            random.seed(idx)
            tuple_start = random.randint(0, abs(length - self.tuple_total_frames))

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.val:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(tuple_order), idx

def gen_sthv2_vcop_splits(root_dir, json_file_input_train, json_file_input_val, json_file_labels, clip_len, interval, tuple_len):
    """Generate split files for different configs."""
    dataset_object_train = WebmDataset(json_file_input_train, json_file_labels,
                                          root_dir, is_test=False)
    dataset_object_val = WebmDataset(json_file_input_val, json_file_labels,
                                          root_dir, is_test=False)
    json_data_train = dataset_object_train.json_data
    json_data_val = dataset_object_val.json_data

    vcop_train_split_name = 'vcop_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    vcop_train_split_path = os.path.join(root_dir, vcop_train_split_name).replace('\\', '/')
    vcop_val_split_name = 'vcop_val_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    vcop_val_split_path = os.path.join(root_dir, vcop_val_split_name).replace('\\', '/')
    #vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    #vcop_test_split_path = os.path.join(root_dir, vcop_test_split_name).replace('\\', '/')
    # minimum length of video to extract one tuple
    min_video_len = clip_len * tuple_len + interval * (tuple_len - 1)

    file_write_obj_train = open(vcop_train_split_path, 'w')
    file_write_obj_val = open(vcop_val_split_path, 'w')
    for i in range(0,len(json_data_train)):
        videodata = skvideo.io.vread(json_data_train[i].path)
        length, height, width, channel = videodata.shape
        if length>=min_video_len:
            file_write_obj_train.writelines(json_data_train[i].path)
            file_write_obj_train.write('\n')
            print(json_data_train[i].path)
    file_write_obj_train.close()
    for i in range(0,len(json_data_val)):
        videodata = skvideo.io.vread(json_data_val[i].path)
        length, height, width, channel = videodata.shape
        if length>=min_video_len:
            file_write_obj_val.writelines(json_data_val[i].path)
            file_write_obj_val.write('\n')
            print(json_data_val[i].path)
    file_write_obj_val.close()
    #val_split = pd.read_csv(os.path.join(root_dir, 'split', 'vallist.txt').replace('\\', '/'), header=None, sep=' ')[0]
    #val_split = val_split[val_split.apply(_video_longer_enough_val)]
    #val_split.to_csv(vcop_val_split_path, index=None)

    #test_split = pd.read_csv(os.path.join(root_dir, 'split', 'testlist.txt').replace('\\', '/'), header=None, sep=' ')[0]
    #test_split = test_split[test_split.apply(_video_longer_enough_test)]
    #test_split.to_csv(vcop_test_split_path, index=None)


def gen_sthv2_trainval_splits(root_dir, json_file_input_train, json_file_input_val, json_file_input_test,json_file_labels, clip_len):
    """Generate split files for different configs."""
    dataset_object_train = WebmDataset(json_file_input_train, json_file_labels,
                                          root_dir, is_test=False)
    dataset_object_val = WebmDataset(json_file_input_val, json_file_labels,
                                          root_dir, is_test=False)
    dataset_object_test = WebmDataset(json_file_input_test, json_file_labels,
                                          root_dir, is_test=True)
    json_data_train = dataset_object_train.json_data
    json_data_val = dataset_object_val.json_data
    json_data_test = dataset_object_test.json_data

    vcop_train_split_name = 'train_clip{}.txt'.format(clip_len)
    vcop_train_split_path = os.path.join(root_dir, vcop_train_split_name).replace('\\', '/')
    vcop_val_split_name = 'val_clip{}.txt'.format(clip_len)
    vcop_val_split_path = os.path.join(root_dir, vcop_val_split_name).replace('\\', '/')
    vcop_test_split_name = 'test_clip{}.txt'.format(clip_len)
    vcop_test_split_path = os.path.join(root_dir, vcop_test_split_name).replace('\\', '/')
    #vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    #vcop_test_split_path = os.path.join(root_dir, vcop_test_split_name).replace('\\', '/')
    # minimum length of video to extract one tuple
    min_video_len = clip_len

    file_write_obj_train = open(vcop_train_split_path, 'w')
    file_write_obj_val = open(vcop_val_split_path, 'w')
    file_write_obj_test = open(vcop_test_split_path, 'w')
    for i in range(0,len(json_data_train)):
        videodata = skvideo.io.vread(json_data_train[i].path)
        length, height, width, channel = videodata.shape
        if length>=min_video_len:
            file_write_obj_train.writelines(json_data_train[i].path)
            file_write_obj_train.write('\n')
            print(json_data_train[i].path)
    file_write_obj_train.close()
    for i in range(0,len(json_data_val)):
        videodata = skvideo.io.vread(json_data_val[i].path)
        length, height, width, channel = videodata.shape
        if length>=min_video_len:
            file_write_obj_val.writelines(json_data_val[i].path)
            file_write_obj_val.write('\n')
            print(json_data_val[i].path)
    file_write_obj_val.close()
    for i in range(0,len(json_data_test)):
        videodata = skvideo.io.vread(json_data_test[i].path)
        length, height, width, channel = videodata.shape
        if length>=min_video_len:
            file_write_obj_test.writelines(json_data_test[i].path)
            file_write_obj_test.write('\n')
            print(json_data_test[i].path)
    file_write_obj_test.close()

if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ucf101_stats()
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 16, 2)
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 32, 3)
    #gen_sthv2_vcop_splits("/data0/liuyang/Something-something-v2/20bn-something-something-v2",
    #                     "/data0/liuyang/Something-something-v2/something-something-v2-train.json",
    #                      "/data0/liuyang/Something-something-v2/something-something-v2-validation.json",
    #                     "/data0/liuyang/Something-something-v2/something-something-v2-labels.json", 16, 8, 3)

    gen_sthv2_trainval_splits("E:/Something-something-v2/20bn-something-something-v2",
                         "E:/Something-something-v2/annotations/something-something-v2-train.json",
                          "E:/Something-something-v2/annotations/something-something-v2-validation.json",
                           "E:/Something-something-v2/annotations/something-something-v2-test.json",
                         "E:/Something-something-v2/annotations/something-something-v2-labels.json", 16)

    # train_transforms = transforms.Compose([
    #     transforms.Resize((128, 171)),
    #     transforms.RandomCrop(112),
    #     transforms.ToTensor()])
    # # train_dataset = UCF101FOPDataset('../data/ucf101', 8, 3, True, train_transforms)
    # # train_dataset = UCF101VCOPDataset('../data/ucf101', 16, 8, 3, True, train_transforms)
    # train_dataset = UCF101Dataset('../data/ucf101', 16, False, train_transforms)
    # # train_dataset = UCF101RetrievalDataset('../data/ucf101', 16, 10, True, train_transforms)    
    # train_dataloader = DataLoader(train_dataset, batch_size=8)

    # for i, data in enumerate(train_dataloader):
    #     clips, idxs = data
    #     # for i in range(10):
    #     #     filename = os.path.join('{}.mp4'.format(i))
    #     #     skvideo.io.vwrite(filename, clips[0][i])
    #     print(clips.shape)
    #     print(idxs)
    #     exit()
    # pass

#if __name__ == '__main__':
#    train_transforms = transforms.Compose([
#        transforms.Resize((128, 171)),
#        transforms.RandomCrop(112),
#        transforms.ToTensor()
#    ])
#    train_dataset = SthV2Dataset_train(root_dir="E:/Something-something-v2/20bn-something-something-v2",
#                         json_file_input="E:/Something-something-v2/annotations/something-something-v2-train.json",
#                         json_file_labels="E:/Something-something-v2//annotations/something-something-v2-labels.json",
#                         clip_len=16,
#                         train=True,
#                         transforms_=train_transforms,
#                         )
#    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,
#                                num_workers=4, pin_memory=True)
