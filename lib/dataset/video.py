import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import torch
import torchaudio
import torchvision
import torchvision.transforms as T
import torchaudio.transforms as AT
import torch.nn.functional as F
from collections import Counter
import random

def vis_MDM(X, X_MOT, save_folder, id):

    [b, m, n, t, c, h, w] = X.size()
    [b, m, n, c, h, w] = X_MOT.size()

    for jb in range(b):
        for jm in range(m):
            for jn in range(n):

                file_ID = f"{id}_b{jb}_m{jm}_n{jn}"
                T.ToPILImage()(X_MOT[jb,jm,jn]).save(f'{save_folder}/{file_ID}_MDM.png', mode='png')
                T.ToPILImage()(X[jb, jm, jn, t//2]).save(f'{save_folder}/{file_ID}_X.png', mode='png')

                T.ToPILImage()(X_MOT[jb,jm,jn, 0:1]).save(f'{save_folder}/{file_ID}_MDM_0.png', mode='png')
                T.ToPILImage()(X_MOT[jb,jm,jn, 1:2]).save(f'{save_folder}/{file_ID}_MDM_1.png', mode='png')
                T.ToPILImage()(X_MOT[jb,jm,jn, 2:3]).save(f'{save_folder}/{file_ID}_MDM_2.png', mode='png')

def load_image( directory, idx, image_tmpl):
    try:
        X = T.ToTensor()(Image.open(f"{directory}/{image_tmpl.format(idx)}").convert('RGB'))
    except Exception as e:
        print('error loading image yy:', f"{directory}/{image_tmpl.format(idx)}")
        exit()
    return X

def scales_pairs(scales):
    pairs = []
    for i1 in range(len(scales)):
        for i2 in range(len(scales)):
            if (abs(i1 - i2) > 1): continue
            pairs.append((scales[i1], scales[i2]))
    return pairs

def augmentation_video( X , scales , img_output_size, param , crop = "random"):
    pairs = scales_pairs(scales)
    (C, H, W) = X.size()[-3:]
    #print("X.size()", X.size())
    S = min(H, W)
    #print("pairs", pairs)

    new_crop_size = random.choice(pairs)
    #print("new_crop_size", new_crop_size)

    new_crop_size = ( int(S * new_crop_size[0]),  int(S * new_crop_size[1]) )
    if crop == "random":
        transform_list = [ T.RandomCrop(new_crop_size), T.Resize((img_output_size,img_output_size))]
    else:
        transform_list = [T.CenterCrop(new_crop_size), T.Resize((img_output_size, img_output_size))]

    if param["RandomHorizontalFlip"]: transform_list.append(T.RandomHorizontalFlip(p=0.5))
    if param["ColorJitter"]: transform_list.append(T.ColorJitter(saturation=(0.75, 1.25), brightness=(0.8, 1.2),contrast=(0.75, 1.25), hue=0.05))
    if param["RandomGrayscale"] > 0: transform_list.append(T.RandomGrayscale(p=float(param["RandomGrayscale"])))
    if param["GaussianBlur"]: transform_list.append(T.GaussianBlur(9, (0.5, 2)))
    transforms = torch.nn.Sequential(*transform_list)
    X = transforms(X.view((-1, C, H, W))).view(X.size()[:-3] + (C, img_output_size,img_output_size))
    return X

def validation_transform( X  , scales , img_output_size, crop = "central"):

    (C, H, W) = X.size()[-3:]
    S = min(H, W)

    if crop == "central":
        new_crop_size = (S, S)
    elif crop == "full":
        pass
    else:
        #pairs = scales_pairs(scales[0:2])
        pairs = scales_pairs(scales)
        new_crop_size = random.choice(pairs)
        new_crop_size = (int(S * new_crop_size[0]), int(S * new_crop_size[1]))
        #print("new_crop_size", new_crop_size)



    if crop == "central":
        transform_list = [T.CenterCrop(new_crop_size), T.Resize((img_output_size, img_output_size))]
    elif crop == "random":
        transform_list = [T.CenterCrop(new_crop_size), T.Resize((img_output_size, img_output_size))]#[T.RandomCrop(new_crop_size), T.Resize((img_output_size, img_output_size))]
    elif crop == "full":
        transform_list = [ T.Resize((img_output_size, img_output_size))]
    else:
        print("validation_transform ERROR: unspecified crop", crop)

    transforms = torch.nn.Sequential(*transform_list)
    X = transforms(X.view((-1, C, H, W))).view( X.size()[:-3] + (C, img_output_size,img_output_size))
    return X

def sample_indices_training( size, num_segments, k_frames ):
    tick = size / float(num_segments)
    indices = np.array([int(tick * k) + randint(tick+1)  for k in range(num_segments)])
    indices[indices < k_frames // 2] = k_frames // 2
    indices[indices > size - 1 - k_frames // 2] = size - (k_frames // 2) - 1
    return indices  + 1

def sample_indices_validation(size, num_segments, k_frames ):
    tick = size  / float(num_segments)
    indices = np.array([int(tick / 2.0 + tick * i) for i in range(num_segments)])
    indices[indices < k_frames//2] = k_frames//2
    indices[indices > size -1 - k_frames // 2] = size - (k_frames // 2) - 1
    return indices  + 1

def get_video_x(record, args,  mode_train_val):


    num_segments, k_frames  =  args["model"]["video_segments"] , args["dataset"]["k_frames"]

    size = record["imagefolder_size"]

    if mode_train_val == "training":
        indices = sample_indices_training(size, num_segments, k_frames)
    elif mode_train_val == "validation":
        indices = sample_indices_validation(size, num_segments, k_frames)
    else:
        indices = sample_indices_training(size, num_segments, k_frames)

    image_tmpl = args["dataset"]["frames_tmpl"]
    X = torch.stack([torch.stack([load_image(record["imagefolder"], indices[s] + i - k_frames // 2, image_tmpl) for i in range(k_frames)]) for s in range(num_segments)])

    scales = args["data_augmentation"]["video_augmentation"]["scales"]
    img_output_size = args["data_augmentation"]["video_image_param"]["output_size"]
    video_augmentation_param = args["data_augmentation"]["video_augmentation"]

    if mode_train_val == "training":
        X = augmentation_video(X, scales, img_output_size, video_augmentation_param)
    elif mode_train_val == "validation":
        X = validation_transform(X,  scales, img_output_size, crop = "central")
    else:
        #multisampling validation:
        scales= [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94,0.93, 0.92, 0.91, 0.9]
        video_augmentation_param["RandomHorizontalFlip"]: False
        video_augmentation_param["ColorJitter"] = False
        video_augmentation_param["RandomGrayscale"] = 0
        X = augmentation_video(X,  scales, img_output_size, video_augmentation_param, crop = "random")#"central" random


    return X









