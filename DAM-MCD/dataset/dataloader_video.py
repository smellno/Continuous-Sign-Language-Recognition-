import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# 用warnings模块中的simplefilter函数来配置警告过滤器，将特定类别的警告消息设置为忽略。在这种情况下，它将忽略类别为FutureWarning的警告消息

import numpy as np
import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")
# 将上一级目录添加到Python模块搜索路径（sys.path）。这是为了使Python能够搜索并导入上一级目录中的模块。通常情况下，当您的Python文件位于项目的子目录中时，您可能需要将项目的根目录添加到模块搜索路径，以便能够导入项目的其他模块。
global kernel_sizes 
# 在Python中，global关键字用于在函数内部声明一个全局变量，以便在整个模块中都能访问和修改该变量。


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            print(fi['folder'])
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])  
        elif self.dataset == 'CSL':
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.jpg")
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride

        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict
from DPC_main2 import extract_images_and_group

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation
from torch.cuda.amp import autocast as autocast

# 导入PyTorch配置文件
os.environ['TORCH_CUDA_ALLOC_CONF'] = '/root/tf-logs/CorrNet/pytorch_env'

import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
import re

warnings.simplefilter(action='ignore', category=FutureWarning)
# 用warnings模块中的simplefilter函数来配置警告过滤器，将特定类别的警告消息设置为忽略。在这种情况下，它将忽略类别为FutureWarning的警告消息

import numpy as np
import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


# 将上一级目录添加到Python模块搜索路径（sys.path）。这是为了使Python能够搜索并导入上一级目录中的模块。通常情况下，当您的Python文件位于项目的子目录中时，您可能需要将项目的根目录添加到模块搜索路径，以便能够导入项目的其他模块。
# 在Python中，global关键字用于在函数内部声明一个全局变量，以便在整个模块中都能访问和修改该变量。

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014-T', drop_ratio=1, num_gloss=-1, mode="train",
                 transform_mode=True,
                 datatype="video", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval  # not implemented for read_features()
        self.image_scale = image_scale  # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"/root/tf-logs/CorrNet/preprocess/{dataset}/{mode}_info.npy",
                                   allow_pickle=True).item()
        self.data_aug = self.transform()
        self.data_aug1 = self.transform1()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        elif self.data_type == "video":
            input_data, label, fi, input_data1, label1 = self.read_video(idx)
            vid_agu = video_augmentation.RandomCrop(self.input_size)
            input_data = vid_agu(input_data)
            img_resize = extract_images_and_group(input_data)
            print('img_resize.shape', img_resize.shape)

            if self.normalize(input_data, label) is not None:
                input_data, label = self.normalize(input_data, label)
            else:
                print('input_data, label none')

            if self.normalize1(img_resize, label1) is not None:
                img_resize, label = self.normalize1(img_resize, label1)
                print('img_resize.size()', img_resize.size())
            else:
                print('input_data, label none')

            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx][
                'original_info'], img_resize, torch.LongTensor(label1)
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            img_folder = os.path.join((self.prefix + "/features/fullFrame-256x256px/" + fi['folder'])[:-8])
            print('img_folder', img_folder)
        elif self.dataset == 'CSL':
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.jpg")
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])
        try:
            image_files1 = []
            files = os.listdir(img_folder)
            image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if len(image_files) == 0:
                print('image_files none')
                return None

            sorted_list = sorted(image_files, key=lambda x: int(x.split('.')[0]))
            for image_file in sorted_list:
                image_path = os.path.join(img_folder, image_file)
                image_files1.append(image_path)
        except Exception as e:
            print(f"Error processing image {img_folder}: {str(e)}")
        # img_list = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        # print(len(img_list))
        image_files2 = image_files1[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]

        # print(img_list)
        # print(len(img_list))
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        label_list1 = label_list.copy()
        print('label_list', label_list)
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in image_files2], label_list

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        if video is None:
            print("normalize video is None")
        else:
            video = torch.tensor(video)
            video = video.float() / 127.5 - 1
            # video = video.float() / 255
            return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def normalize1(self, video, label, file_id=None):
        video, label = self.data_aug1(video, label, file_id)
        if video is None:
            print("normalize video is None")
        else:
            video = torch.tensor(video)
            video = video.float() / 127.5 - 1
            # video = video.float() / 255
            return video, label

    def transform1(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    #     def collate_fn(batch):

    #         batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
    #         video, label, info = list(zip(*batch))

    #         left_pad = 0
    #         last_stride = 1
    #         total_stride = 1
    #         global kernel_sizes
    #         for layer_idx, ks in enumerate(kernel_sizes):
    #             if ks[0] == 'K':
    #                 left_pad = left_pad * last_stride
    #                 left_pad += int((int(ks[1]) - 1) / 2)
    #             elif ks[0] == 'P':
    #                 last_stride = int(ks[1])
    #                 total_stride = total_stride * last_stride
    #         video1 = np.array(video[0])
    #         if len(video1.shape) > 3:
    #             max_len = len(video[0])
    #             video_length = torch.LongTensor(
    #                 [np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad for vid in video])
    #             # video_length = video_length.to(torch.float32)  # 将输入数据转换为 float32（单精度浮点数）

    #             right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    #             max_len = max_len + left_pad + right_pad
    #             # video = np.array(video)
    #             # video = torch.from_numpy(video)
    #             # padded_video = [torch.cat(
    #             #     (
    #             #         vid[0][None].expand(left_pad, -1, -1, -1),
    #             #         vid,
    #             #         vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
    #             #     )
    #             #     , dim=0)
    #             #     for vid in video]
    #             vide = []
    #             for vid in video:
    #                 try:
    #                     if vid is not None and vid[0] is not None:
    #                         vid = np.array(vid)
    #                         print('vid.shape', vid.shape)
    #                         vid_tensor = torch.from_numpy(vid)
    #                         # 现在可以在vid_tensor上执行expand操作
    #                         padded_video = [torch.cat(
    #                             (
    #                                 vid_tensor[0][None].expand(left_pad, -1, -1, -1),
    #                                 vid_tensor,
    #                                 vid_tensor[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
    #                             )
    #                             , dim=0)]
    #                         padded_video = torch.stack(padded_video)

    #                         vide.append(padded_video)
    #                 except Exception as e:
    #                     print(f"Error processing image")

    #             padded_video = torch.cat(vide, dim=0)
    #             print('padded_video.shape', padded_video.shape)
    #         else:
    #             print('video[0].shape',video1.shape)
    #             max_len = len(video[0])
    #             video_length = torch.LongTensor([len(vid) for vid in video])
    #             vide = []
    #             for vid in video:
    #                 try:
    #                     if vid is not None and vid[0] is not None:
    #                         vid = np.array(vid)
    #                         print('vid.shape', vid.shape)
    #                         vid_tensor = torch.from_numpy(vid)
    #                         # 现在可以在vid_tensor上执行expand操作
    #                         padded_video = [torch.cat(
    #                             (
    #                                 vid,
    #                                 vid[-1][None].expand(max_len - len(vid), -1),
    #                             )
    #                             , dim=0)]
    #                         padded_video = torch.stack(padded_video)

    #                         vide.append(padded_video)
    #                 except Exception as e:
    #                     print(f"Error processing image")

    #             padded_video = torch.cat(vide, dim=0).permute(0, 2, 1)
    #             print('padded_video.shape', padded_video.shape)
    #         label_length = torch.LongTensor([len(lab) for lab in label])
    #         if max(label_length) == 0:
    #             return padded_video, video_length, [], [], info
    #         else:
    #             padded_label = []
    #             for lab in label:
    #                 padded_label.extend(lab)
    #             padded_label = torch.LongTensor(padded_label)
    #             return padded_video, video_length, padded_label, label_length, info

    def collate_fn(batch):

        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info, video_resize, label1 = list(zip(*batch))
        video = np.array(video, dtype=object)
        video_resize = np.array(video_resize, dtype=object)
        print('video.shapes', video.shape)
        print('video.shapes', video.shape)
        print('video_resize.shapes', video_resize.shape)
        print('video_resize.shapes', video_resize.shape)

        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride
                left_pad += int((int(ks[1]) - 1) / 2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride

        video1 = np.array(video[0])
        video2 = np.array(video[1])

        print('video1.shapes', video1.shape)
        print('video2.shapes', video2.shape)

        size_x = video1.shape[0]
        size_y = video2.shape[0]

        if len(video1.shape) > 3:
            if size_x > size_y:
                max_len = len(video[0])
            elif size_x == size_y:
                max_len = len(video[0])
            else:
                max_len = len(video[1])
            video_length = torch.LongTensor(
                [np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad for vid in video])
            # video_length = video_length.to(torch.float32)  # 将输入数据转换为 float32（单精度浮点数）

            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            vide = []
            for vid in video:
                try:
                    if vid is not None and vid[0] is not None:
                        vid = np.array(vid)
                        vid_tensor = torch.from_numpy(vid)
                        # 现在可以在vid_tensor上执行expand操作
                        padded_video = [torch.cat(
                            (
                                vid_tensor[0][None].expand(left_pad, -1, -1, -1),
                                vid_tensor,
                                vid_tensor[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                            )
                            , dim=0)]
                        padded_video = torch.stack(padded_video)
                        print('padded_video.size()', padded_video.size())
                        print('padded_video.size()', padded_video.size())

                        vide.append(padded_video)
                        print(len(vide))
                except Exception as e:
                    print(f"Error processing image")
            padded_video = torch.cat(vide, dim=0)
            print('padded_video.size()', padded_video.size())

        else:
            print('video[0].shape', video1.shape)
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            vide = []
            for vid in video:
                try:
                    if vid is not None and vid[0] is not None:
                        vid = np.array(vid)
                        print('vid.shape', vid.shape)
                        vid_tensor = torch.from_numpy(vid)
                        # 现在可以在vid_tensor上执行expand操作
                        padded_video = [torch.cat(
                            (
                                vid,
                                vid[-1][None].expand(max_len - len(vid), -1),
                            )
                            , dim=0)]
                        padded_video = torch.stack(padded_video)

                        vide.append(padded_video)
                except Exception as e:
                    print(f"Error processing image")
            padded_video = torch.cat(vide, dim=0).permute(0, 2, 1)

        video_resize1 = np.array(video_resize[0])
        video_resize2 = np.array(video_resize[1])
        # print('video_resize1', video_resize1)
        # print('video_resize2', video_resize2)

        print('video_resize1.shapes', video_resize1.shape)
        print('video_resize2.shapes', video_resize2.shape)
        size_x1 = video_resize1.shape[0]
        size_y1 = video_resize2.shape[0]
        if len(video_resize1.shape) > 3:
            if size_x1 > size_y1:
                max_len = len(video_resize[0])
            elif size_x1 == size_y1:
                max_len = len(video_resize[0])
            else:
                max_len = len(video_resize[1])

            video_length1 = torch.LongTensor(
                [np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad for vid in video_resize])
            # video_length = video_length.to(torch.float32)  # 将输入数据转换为 float32（单精度浮点数）

            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            vide1 = []
            for vid in video_resize:
                try:
                    if vid is not None and vid[0] is not None:
                        vid = np.array(vid)
                        print('vid.shape', vid.shape)
                        vid_tensor = torch.from_numpy(vid)
                        # 现在可以在vid_tensor上执行expand操作
                        padded_video1 = [torch.cat(
                            (
                                vid_tensor[0][None].expand(left_pad, -1, -1, -1),
                                vid_tensor,
                                vid_tensor[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                            )
                            , dim=0)]
                        padded_video1 = torch.stack(padded_video1)

                        vide1.append(padded_video1)
                except Exception as e:
                    print(f"Error processing image")

            padded_video1 = torch.cat(vide1, dim=0)
        else:
            print('video_resize1.shape<=3')
            max_len = len(video_resize[0])
            video_length1 = torch.LongTensor([len(vid) for vid in video_resize])
            vide1 = []
            for vid in video_resize:
                try:
                    if vid is not None and vid[0] is not None:
                        vid = np.array(vid)
                        print('vid.shape', vid.shape)
                        vid_tensor = torch.from_numpy(vid)
                        # 现在可以在vid_tensor上执行expand操作
                        padded_video1 = [torch.cat(
                            (
                                vid,
                                vid[-1][None].expand(max_len - len(vid), -1),
                            )
                            , dim=0)]
                        padded_video1 = torch.stack(padded_video1)

                        vide1.append(padded_video1)
                except Exception as e:
                    print(f"Error processing image")

            padded_video1 = torch.cat(vide1, dim=0).permute(0, 2, 1)

        # 合并两个张量

        print('padded_video.shape', padded_video.shape)
        print('padded_video1.shape', padded_video1.shape)
        combined_tensor = torch.cat([padded_video, padded_video1], dim=1)

        # 查看合并后张量的形状
        print("Combined Tensor Shape:", combined_tensor.shape)

        video_length1 = torch.LongTensor(
            [np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad for vid in video_resize])
        label_length = torch.LongTensor([len(lab) for lab in label])
        label_length1 = torch.LongTensor([len(lab) for lab in label1])

        if max(label_length) == 0:
            return padded_video, video_length, [], [], info, video_length1
        else:
            padded_label1 = []
            for lab in label1:
                padded_label1.extend(lab)
            padded_label1 = torch.LongTensor(padded_label1)

            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)

            print('video_length1', video_length1)
            print('video_length', video_length)

            return combined_tensor, video_length, padded_label, label_length, info, video_length1


if __name__ == "__main__":
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open('/root/tf-logs/CorrNet/configs/baseline.yaml', 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"/root/tf-logs/CorrNet/configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    prefix = '/root/autodl-tmp/phoenix-2014-T.v3/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
    gloss_dict = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()
    feeder = BaseFeeder(prefix, gloss_dict, dataset='phoenix2014-T', drop_ratio=1, num_gloss=-1, mode="train",
                        transform_mode=True,
                        datatype="video", frame_interval=1, image_scale=1.0, kernel_size=['K5', 'P2', 'K5', 'P2'],
                        input_size=224)

    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=feeder.collate_fn
    )
    for data in dataloader:
        print("data[0].shape", data[0].shape)
        print("video_length", data[1])
        print("data[5].shape", data[5].shape)
        print("video_length1", data[6])

    prefix = '/home/xsj/CorrNet/dataset/phoenix2014-T/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
'/home/xsj/CorrNet/configs/baseline.yaml', 'r'