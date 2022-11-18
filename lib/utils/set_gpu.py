import torch


def set_model_DataParallel( args, model ):

    DataParallel = False
    cuda_ids = args["cuda_ids"]
    if len(cuda_ids) > 1:

        print(f"Let's use {cuda_ids} GPUs out of {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model, device_ids=cuda_ids , output_device=cuda_ids[0], dim=0)
        DataParallel = True

    return model, DataParallel


def set_cuda_device(args_command_line, args):
    #args_command_line.cuda_ids = "0_4_5" set 3 gpu to use : 0,4,5, must be in increacing order
    if args_command_line.device:
        cuda_device_ids = list(map(int, args_command_line.device.split(",")))  # gpu to be used
        args["cuda_ids"] = cuda_device_ids
    else:
        cuda_device_ids = [0]  # set here default GPU ids , normally [0,1,2,..,16]
        args["cuda_ids"] = [0]

    device = torch.device(f"cuda:{cuda_device_ids[0]}" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
    device_id =cuda_device_ids[0]
    return device , device_id , args


