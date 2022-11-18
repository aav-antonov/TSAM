


import os
import torch

import torchvision.transforms as T

from collections import Counter

from lib.dataset.audio import get_audio_x
from lib.dataset.video import  get_video_x


import time

class MmDataSet(torch.utils.data.Dataset):
    def __init__(self, input_file,  args ={}, mode_train_val = "training", n_sample= 1 ):

        self.input_file = input_file
        self.args = args
        self.n_sample = n_sample

        param_dataset = args["dataset"]
        self.path_frame_folders = f'{param_dataset["data_dir"]}/{param_dataset["dir_frames"]}'
        self.path_audio_folder = f'{param_dataset["data_dir"]}/{param_dataset["dir_audios"]}'

        self.mode_train_val = mode_train_val

        self.parse_input_file()

        #self.__getitem__(0)
        #exit()



    def __getitem__(self, index):

        record = self.records[index]
        x_video, x_audio = "None", "None"
        ID = record["ID"]
        y = int(record["label"])

        if self.args["model"]["video_segments"] > 0:
            x_video = torch.stack([get_video_x(record, self.args, self.mode_train_val) for _ in range(self.n_sample)])

        if self.args["model"]["audio_segments"] > 0:
            x_audio = torch.stack([get_audio_x(record, self.args, self.mode_train_val) for _ in range(self.n_sample)])

        return  ID,   y , x_video, x_audio


    def __len__(self):
        return len(self.records)


    def parse_input_file(self):
        self.records = []
        error_not_exist, error_no_frames, error_no_wav = {}, {}, {}
        stat_labels = Counter()

        lines = [x.strip().split(' ') for x in open(self.input_file)]

        for i, item in enumerate(lines):
            ID = item[0]
            label = int(item[1])

            indicator, ifolder, ifolder_size, wavefile = self.check(label, ID, error_not_exist, error_no_frames, error_no_wav)

            if indicator:
                line_data = {}
                line_data["imagefolder"] = ifolder
                line_data["imagefolder_size"] = ifolder_size
                line_data["audio_file"] = wavefile

                line_data["ID"] = ID
                line_data["label"] = label
                stat_labels[label] += 1

                self.records.append(line_data)

        print("stat labels", len(stat_labels))
        print("error_not_exist", len(error_not_exist))
        print("error_no_frames", len(error_no_frames))
        print("error_no_wav", len(error_no_wav))
        print("self.records:", len(self.records))


    def check(self, label, ID, error_not_exist, error_no_frames, error_no_wav):

        ifolder, ifolder_size, wavefile = "", "", ""

        """ Check folder with frames exist and has minimum number of frames"""

        if self.args["model"]["video_segments"] > 0:

            ifolder = f"{self.path_frame_folders}/{ID}"

            if not os.path.isdir(ifolder):
                error_not_exist[ID] = 1
                return False , ifolder , ifolder_size , wavefile

            ifolder_size = int(len(os.listdir(ifolder)))
            if ifolder_size < 3:
                error_no_frames[ID] = ifolder_size
                return False , ifolder , ifolder_size, wavefile

        """ Check wavefile exist """

        if self.args["model"]["audio_segments"] > 0:
            wavefile = f"{self.path_audio_folder}/{ID}.wav"
            if not os.path.isfile(wavefile):
                error_no_wav[ID] = 1
                return False , ifolder , ifolder_size, wavefile

        return True, ifolder, ifolder_size, wavefile





def GetDataSet(args, mode_train_val="training",n_sample= 1):

    param_dataset = args["dataset"]

    if mode_train_val == "training":
        file_list = f'{param_dataset["data_dir"]}/{param_dataset["file_train_list"]}'
    else:
        file_list = f'{param_dataset["data_dir"]}/{param_dataset["file_val_list"]}'

    MMD = MmDataSet(file_list, args=args, mode_train_val=mode_train_val,n_sample= n_sample )

    return MMD

def GetDataLoaders(MMD, args,  mode_train_val="training"):

    shuffle = False
    if mode_train_val == "training": shuffle = True

    data_loader = torch.utils.data.DataLoader(
        MMD,
        batch_size=args["training_param"]["batch_size"],
        num_workers=args["training_param"]["num_workers"],
        shuffle=shuffle, pin_memory=False, drop_last=False
    )

    return data_loader




def visualise_x(x, ID, folder):

    (n_batches, n_samples, n_segment, c, h, w) = x.size()

    for j_batch in range(n_batches):

        folder_id = f"{folder}/{ID[j_batch]}"
        if not os.path.exists(folder_id): os.mkdir(folder_id)

        for j_sample in range(n_samples):
            for j_seg in range(n_segment):
                file_ID = f"sample_{j_sample}_segment_{j_seg}"
                T.ToPILImage()(x[j_batch,j_sample,j_seg ]).save(f'{folder_id}/{file_ID}.png', mode='png')

def test_input( train_loader, model_input, device, folder):

    if not os.path.exists(folder): os.mkdir(folder)

    print(f'test_input, see output at: {folder}\n')

    count_IDs = 0
    for i, (ID, y, x_video, x_audio) in enumerate(train_loader):

        if x_video[0] != "None": print("x_video", x_video.size())
        if x_audio[0] != "None": print("x_audio", x_audio.size())

        if x_video[0] != "None": x_video = x_video.to(device)
        if x_audio[0] != "None": x_audio = x_audio.to(device)

        x = model_input(x_video, x_audio)

        visualise_x(x, ID, folder)
        count_IDs += len(ID)

        if count_IDs > 10: break




