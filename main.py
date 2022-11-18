import argparse

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from lib.model.prepare_input import  x_input
from lib.model.backbone import BackBone
from lib.model.model import VCM
from lib.model.policy import gradient_policy

from lib.dataset.dataset import GetDataSet, GetDataLoaders, test_input
from lib.utils.set_gpu import set_model_DataParallel, set_cuda_device
from lib.utils.utils import loadarg,  AverageMeter , get_num_class
from lib.utils.set_folders import check_rootfolders , get_file_results
from lib.utils.saveloadmodels import checkpoint_acc , save_timepoint , load_model , fine_tune
from lib.utils.report import report_model_param
from lib.running.run import run_epoch , run_MultiClip
from lib.utils.saveloadmodels import  WeightsData , CheckWeightsExistDownLoadElse



def main():

    global device, device_id

    global args, best_score, history_score
    best_score, history_score = {} , {}
    best_score["balanced"], best_score["unbalanced"] = 0.00, 0.00

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)  # path to json config file with dataset, example see in "config/example.json"
    parser.add_argument("--config", type=str) # path to json config file of the model, example see in "config/example.json"
    parser.add_argument("--resume", type=str) # path to checkpoint folder with model to resume"
    parser.add_argument("--fine_tune", type=str)
    parser.add_argument("--device", type=str) # string to with cuda ids to be used, i.e. 0_2_6 (gpu 0, 2 and 6 will be used)
    parser.add_argument("--run_id", type=str) # id for this run, if not specified id = f"{date}_run"
    parser.add_argument("--validate", type=str) # path to checkpoint folder with model to validate"

    args_command_line = parser.parse_args()
    args = {**loadarg(args_command_line.data), **loadarg(args_command_line.config)}

    """get number of classes from file: <file_train_list>, see config/example.json"""
    args["model"]["num_class"] = get_num_class(args)
    print("args.num_class", args["model"]["num_class"])

    if args_command_line.run_id:
        args["run_id"] = args_command_line.run_id


    """set_model"""
    model_input = x_input(args)
    model_BackBone = BackBone(args,model_input.config_segments)
    model = VCM(args,model_input,model_BackBone)
    
    

    """set_cuda_device"""
    device, device_id , args = set_cuda_device(args_command_line,args)
    model, DataParallel = set_model_DataParallel(args, model)
    model.to(device)

    print("device, device_id", device, device_id)
    print("DataParallel", DataParallel)

    """set loss and optimizer"""
    optimizer = torch.optim.SGD(model.parameters(), args["optimizer_param"]["lr"], momentum=args["optimizer_param"]["momentum"], weight_decay=args["optimizer_param"]["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    cudnn.benchmark = True


    """set DataLoaders"""
    TrainDataSet = GetDataSet(args, mode_train_val="training")
    ValidDataSet = GetDataSet(args, mode_train_val="validation")
    TrainDataLoader = GetDataLoaders(TrainDataSet, args, mode_train_val="training")
    ValidDataLoader = GetDataLoaders(ValidDataSet, args, mode_train_val="validation")
    DataLoaders = [TrainDataLoader, ValidDataLoader]

    """Adjust args """
    args["start_epoch"], args["last_epoch"] = 0, args["training_param"]["epochs"]
    args["dataset"]["k_frames"] = 1


    """On validate: run MultiClip validation"""
    if args_command_line.validate:
        print("args_command_line.validate", args_command_line.validate)
        
        if args_command_line.validate == "kinetics400":
            model_pretrained_param = WeightsData()
            [file_chkpoint, http_link_chkpoint] = model_pretrained_param["resnet50_kinetics400"]
            print("file_chkpoint", file_chkpoint)
            print("http_link_chkpoint", http_link_chkpoint)
            CheckWeightsExistDownLoadElse(file_chkpoint, http_link_chkpoint)
            (model, optimizer, data_state) = load_model(file_chkpoint, model, optimizer, DataParallel=DataParallel, Filter_layers={})
        elif args_command_line.validate == "something_v1":
            model_pretrained_param = WeightsData()
            [file_chkpoint, http_link_chkpoint] = model_pretrained_param["resnet50_something_v1"]
            CheckWeightsExistDownLoadElse(file_chkpoint, http_link_chkpoint)
            (model, optimizer, data_state) = load_model(file_chkpoint, model, optimizer, DataParallel=DataParallel, Filter_layers={})
        else:
            (model, optimizer, data_state) = load_model(args_command_line.validate, model, optimizer,DataParallel=DataParallel, Filter_layers={})
            
        print("Data_state:", data_state)
        run_MultiClip(args, ValidDataSet, model, device, MultiClip = 10)
        exit()

    """On Fine_tune"""
    if args_command_line.fine_tune:
        model = fine_tune(args_command_line.fine_tune, model, DataParallel=DataParallel)

    """On Resume"""
    if args_command_line.resume:
        (model, optimizer, data_state) = load_model(args_command_line.resume, model, optimizer, DataParallel=DataParallel,Filter_layers={})

        #print("On Resume epoch:", args.start_epoch)
        print("On Resume data_state:", data_state)
        epoch = data_state['epoch']
        print("On Resume epoch:", epoch , "last_epoch:", args["last_epoch"])
        print('Results output_folder: ', args["output_folder"])

        if  args["last_epoch"] - epoch < 5:
            args["start_epoch"] = epoch +1
            args["last_epoch"] = epoch +5
            print('Extending with 5 epochs')

    """Create output folders"""
    if "output_folder" not in args:
        check_rootfolders(args)
    print('Results would be stored at: ', args["output_folder"])


    """ TRAINING """
    for epoch in range(args["start_epoch"], args["last_epoch"]):

        """print model args"""
        report_model_param(args)

        """save_epoch if specified in args"""
        if 'save_epoch' in args:
            if epoch in args["save_epoch"]:
                save_timepoint(f"epoch_{epoch}", args["checkpoint"],  model, optimizer, DataParallel)

        """adjust learning rate with epoch """
        gradient_policy(args, epoch, optimizer)

        """RUN epoch """
        run_data = [best_score, history_score, epoch]
        run_epoch(args, run_data, DataLoaders, model, device, optimizer, criterion, DataParallel = DataParallel)


if __name__ == '__main__':
    main()
