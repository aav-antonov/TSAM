import shutil
import json
import os
import torch
import torch.nn as nn
from collections import Counter




def adjust_args_in(args, args_in):



    #if args["dataset"]["name"] ==  "adcumen":
    #    args = adcumen_model_segment_config(args)

    args["dataset"]["k_frames"] = 1
    if args["TSM"]["motion"]:
        args["dataset"]["k_frames"] = args["TSM"]["motion_param"]["k_frames"]

    args["video_segments"] = args["TSM"]["video_segments"]
    args["audio_segments"] = args["TSM"]["audio_segments"]
    args["segment_frames"] = args["dataset"]["k_frames"]


    return  args




def adcumen_model_segment_config(args):

    clip_length = args["adcumen"]["clip_length"]
    clip_fps = args["adcumen"]["clip_fps"]
    args["TSM"]["video_segments"] = clip_length * clip_fps
    return  args






