

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

from re import search



def random_shift_waveform(waveform, sr, t_start = 0.5, t_end = 0.5):
    s = random.randint(0,t_start * sr)
    f = waveform.size()[1]- random.randint(0,t_end * sr)
    #print("sf", s, f, waveform.size())
    if f - s > 4800:
        waveform = waveform[:,s:f]
    return waveform

def shift_waveform(waveform, sr, t_start = 0.5, t_end = 0.5):
    s = int(t_start * sr)
    f = waveform.size()[1]- int(t_end * sr)

    waveform = waveform[:,s:f]
    return waveform

def wafeform_to_split(args,S):
    n = args["num_segments"]
    m = args["m_segments"]

    x = int(m*S//(m*n + 1 - n))
    y = int(x//m)
    se = [[i*x -i*y,i*x -i*y + x] for i in range(n)]
    #print("se", se)
    return se

def wafeform_to_spectrogram_torch(wafeform, sr,  args):
    n_mels = args["n_mels"]
    [num_channels, H, W] = args["spec_size"]  # 3
    window_sizes = args["window_sizes"]  # [25, 50, 100]
    hop_sizes = args["hop_sizes"]  # [10, 25, 50]

    eps = args["eps"]  # 1e-6
    S = wafeform.size()[-1]
    se = wafeform_to_split(args,S)
    S = torch.zeros((len(se),num_channels, H, W))
    for j,[s,e] in enumerate(se):
        for i in range(num_channels):
            window_length = int(round(window_sizes[i] * sr / 1000))
            hop_length = int(round(hop_sizes[i] * sr / 1000))
            try:
                X = torch.unsqueeze(torchaudio.transforms.MelSpectrogram(sample_rate=sr,  n_fft=4800, win_length=window_length, hop_length=hop_length, n_mels=n_mels)(wafeform[s:e]), 0)
                #print("s:e",  s, e, sr)
            except:
                print("s:e", s,e , sr)
                print("wafeform:", wafeform.size())
                exit()

            X = np.log(X + eps)
            S[j][i] = torchvision.transforms.Resize((H, W))(X)
    return S


def augmentation_audio(waveform, sr, param):

    if 'random_shift_waveform' in param:

        t_start = param["random_shift_waveform"][0]
        t_end = param["random_shift_waveform"][1]
        waveform = random_shift_waveform(waveform, sr, t_start=t_start, t_end=t_end)

    return waveform



def get_audio_x( record, args , mode_train_val):

    audio_augmentation_param, audio_img_param = args["data_augmentation"]["audio_augmentation"], args["data_augmentation"]["audio_image_param"]
    file_audio = record["audio_file"]
    waveform, sr = torchaudio.load(file_audio)

    ## data augmentation by cuting/shifting waveform
    if mode_train_val == "training":
        if audio_augmentation_param["status"]:
            #print("audio_augmentation_param", audio_augmentation_param["status"])
            waveform = augmentation_audio(waveform, sr, audio_augmentation_param)


    X = torch.squeeze(wafeform_to_spectrogram_torch(waveform[0], sr, audio_img_param),0)

    if args["model"]["audio_segments"] == 2:
        if waveform.size()[0] > 1:
            X1 = torch.squeeze(wafeform_to_spectrogram_torch(waveform[1], sr, audio_img_param),0)
            X = torch.stack([X,X1])
            #print("stereo ", waveform.size())
        else:
            X = torch.stack([X, X])
            #print("mono ", waveform.size())
    else:
        X = torch.stack([X])
        #print("X stereo", X.size())

    return X




