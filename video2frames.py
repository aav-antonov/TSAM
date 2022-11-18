import os
import argparse
import json

from lib.utils.utils import loadarg , create_clean_DIR




def run_frames(FID, param,  error_conversion):
    fps =  param["fps"]
    path_frame_folder = param["path_frame_folder"]
    frames_tmpl  = param["frames_tmpl"]

    frames_tmpl = frames_tmpl.rstrip(".jpg")
    frames_tmpl = frames_tmpl.strip("}")
    frames_tmpl = frames_tmpl.strip("{")
    frames_tmpl = frames_tmpl.strip(":")
    frames_tmpl = "%" + frames_tmpl


    print("run_frames: ", len(FID),  path_frame_folder, "frames_tmpl:", frames_tmpl)
    #exit()
    print("run_frames fps: ", fps)

    for i, [id, file_video] in enumerate(FID):
        #print(i, id, file_mp4)

        create_clean_DIR(f"{path_frame_folder}/{id}")
        if fps > 0:
            cmd_frames = f"ffmpeg  -i {file_video}  -vf \"scale=-1:256,fps={fps}\" -q:v 0 \"{path_frame_folder}/{id}/{frames_tmpl}.jpg\" "
        else:
            cmd_frames = f"ffmpeg  -i {file_video}  -vf \"scale=-1:256\" -q:v 0 \"{path_frame_folder}/{id}/{frames_tmpl}.jpg\" "
#-loglevel panic
        #print(cmd_frames)
        #exit()
        try:
            os.system(cmd_frames)
        except:
            error_conversion[id] = "video"
            print(f"An exception occurred {id}\n")


def run_audio(HHV_split, param, error_conversion):

    path_data_out_audios =  param["path_audio_folder"]


    for i, [id, file_mp4] in enumerate(HHV_split):

        cmd_audio = f"ffmpeg -loglevel panic -i {file_mp4}   {path_data_out_audios}/{id}.wav"
        print("cmd_audio ", cmd_audio )

        try:
            os.system(cmd_audio)
        except:
            error_conversion[id] = "audio"
            print(f"An exception occurred {id}\n")


def split_data(data, n):
    """Yield successive n-sized chunks from data"""
    for i in range(0, len(data), n):
        yield data[i:i + n]

def multirun_video(FID, param, THREAD_N):

    SIZE_SPLIT = len(FID) // THREAD_N
    splits = list(split_data(FID, SIZE_SPLIT))

    for i, split in enumerate(splits):
        print(i, len(split), split[0])

    import multiprocessing
    manager = multiprocessing.Manager()
    error_conversion = manager.dict()

    threads = []
    for i, split in enumerate(splits):
        thread = multiprocessing.Process(target=run_frames, args=(split, param, error_conversion))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return error_conversion


def multirun_audio(FID, param, THREAD_N ):

    SIZE_SPLIT = len(FID) // THREAD_N
    splits = list(split_data(FID, SIZE_SPLIT))

    for i, split in enumerate(splits):
        print(i, len(split), split[0])

    path_data_out_audios = param["path_audio_folder"]
    create_clean_DIR(f"{path_data_out_audios}")

    import multiprocessing
    manager = multiprocessing.Manager()
    error_conversion = manager.dict()

    threads = []
    for i, split in enumerate(splits):
        thread = multiprocessing.Process(target=run_audio, args=(split,  param, error_conversion))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return error_conversion




def main():

    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)  # path to json config file, example see in "config/example.json"
    parser.add_argument("--fps", type=int)  # optional, frame per second, if not specified use default fps from video
    parser.add_argument("--audio", action='store_true')  # --audio optional, to convert audio, if not specified-> video is converted

    THREAD_N = 10 # specify the number of threads to be used for framing

    args_command_line = parser.parse_args()
    if args_command_line.data:
        args = loadarg(args_command_line.data)
    else:
        print("Usage: python video2frame.py --config <config.json> --fps <10 (optional)> --SIZE_SPLIT <1000 (optional)>")
        print("Please see example of config file in <config/example.json>.")
        exit()

    path_video_folder = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_videos"]}'
    path_frame_folder = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_frames"]}'
    path_audio_folder = f'{args["dataset"]["data_dir"]}/{args["dataset"]["dir_audios"]}'

    list_video_files = os.listdir(path_video_folder)
    FID = []
    for file_video in list_video_files:
        file_id = file_video.rstrip(".mp4")
        FID.append([file_id, f"{path_video_folder}/{file_video}"])

    param = {}
    param["path_frame_folder"] = path_frame_folder
    param["path_audio_folder"] = path_audio_folder
    param["frames_tmpl"] = args["dataset"]["frames_tmpl"]

    param["fps"] = 10
    if args_command_line.fps:
        param["fps"] = args_command_line.fps


    if args_command_line.audio:

        error_conversion = multirun_audio(FID, param, THREAD_N)

        if len(error_conversion) > 0:
            file_error_conversion = f'{args["dataset"]["data_dir"]}/audio_conversion_error'
            with open(file_error_conversion, 'w') as f:
                json.dump(error_conversion, f, indent=4)

            print(f"Attention: there are {len(error_conversion)} errors in audio conversion , see details in file: {file_error_conversion}")

    else:

        try:
            os.mkdir(param["path_frame_folder"])
        except OSError:
            print("Dir %s exist" % param["path_frame_folder"])

        error_conversion = multirun_video(FID, param, THREAD_N)

        if len(error_conversion) > 0:
            file_error_conversion = f'{args["dataset"]["data_dir"]}/video_conversion_error'
            with open(file_error_conversion, 'w') as f:
                json.dump(error_conversion, f, indent=4)

            print(f"Attention: there are {len(error_conversion)} errors in video conversion to frames, see details in file: {file_error_conversion}")


if __name__ == '__main__':
    main()