
import os
import torch

from pathlib import Path
from urllib.request import urlopen

def WeightsData():

    try:
        os.mkdir("net_weigths")
    except OSError:
        print("folder exists: net_weigths")

    model_pretrained_param = {}
    model_pretrained_param["resnet50_miil_21k"] = ["net_weigths/resnet50_miil_21k.pth", "https://bitbucket.org/AlexeyAntonov74/resnet50_pretrained_weights/raw/cc2bc64184212a0c0b710ab27f78b54ec7c06adb/resnet50_miil_21k.pth"]
    model_pretrained_param["resnet50_kinetics400"] = ["net_weigths/resnet50_kinetics400.ckpt.pth.tar","https://bitbucket.org/AlexeyAntonov74/resnet50_pretrained_weights/raw/cc2bc64184212a0c0b710ab27f78b54ec7c06adb/resnet50_kinetics400.ckpt.pth.tar"]
    model_pretrained_param["resnet50_something_v1"] = ["net_weigths/resnet50_something_v1.tar","https://bitbucket.org/AlexeyAntonov74/resnet50_pretrained_weights/raw/cc2bc64184212a0c0b710ab27f78b54ec7c06adb/resnet50_something_v1.tar"]

    return model_pretrained_param


def CheckWeightsExistDownLoadElse(file_weights,file_url):

    file_weights_Path = Path(file_weights)
    if file_weights_Path.is_file():
        pass
    else:
        
        try:
            # Download from URL
            with urlopen(file_url) as file:
                content = file.read()

            # Save to file
            with open(file_weights, 'wb') as download:
                download.write(content)
        
        except:
            print(f'Error: cant retrieve file with pretrained weights: {file_url}')
        





def print_model_layers(model, DataParallel=False):
    if DataParallel:
        for name in model.module.model_layers:
            print(name)
    else:
        for name in model.model_layers:
            print(name)


def get_model_state(model, DataParallel=False):
    model_state = {}
    if DataParallel:
        for name in model.module.model_layers:
            print("get_model_state model_layers:", name)
            model_state[name] = model.module.model_layers[name].state_dict()
    else:
        for name in model.model_layers:
            model_state[name] = model.model_layers[name].state_dict()
    return model_state


def load_model(path_checkpoint,  model, optimizer, DataParallel=False, Filter_layers = None):

    if os.path.isfile(path_checkpoint):
        print(f"=> loading checkpoint {path_checkpoint}\n")
        checkpoint = torch.load(path_checkpoint)

        data_state = checkpoint['data_state']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_state = checkpoint['model_state']
        print(f"DataParallel: {DataParallel}\n")
        #print(f"model: {model.module.model_layers}\n")
        if DataParallel:
            for name in model.module.model_layers:

                if name in Filter_layers: continue
                model.module.model_layers[name].load_state_dict(model_state[name])
        else:
            for name in model.model_layers:
                if name in Filter_layers: continue
                model.model_layers[name].load_state_dict(model_state[name])

    else:
        print(f"=> no checkpoint found at {path_checkpoint}")

    return (model, optimizer, data_state)

def save_timepoint(ID, path_checkpoint, model, optimizer,  DataParallel=False , data_state = None):
    filename = f'{path_checkpoint}/{ID}.ckpt.pth.tar'
    model_state = get_model_state(model,DataParallel = DataParallel)
    torch.save({
        'data_state': data_state,
        'model_state': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)



def checkpoint_loss( ID, path_checkpoint, epoch, best_score, current_score,  model, optimizer, DataParallel = False ):

    is_best_score = current_score < best_score
    best_score = min(current_score, best_score)
    data_state = {'epoch': epoch + 1, f'best_score {ID}': best_score}

    print("checkpoint_loss", data_state)

    save_timepoint(ID, path_checkpoint, model, optimizer, data_state=data_state, DataParallel=DataParallel)


    return best_score, is_best_score

def checkpoint_acc( ID, path_checkpoint, epoch, best_score, current_score,  model, optimizer, DataParallel = False ):

    is_best_score = current_score > best_score
    best_score = max(current_score, best_score)
    data_state = {'epoch': epoch + 1, f'best_score {ID}': best_score}

    print("checkpoint_acc", ID, data_state)
    save_timepoint(ID, path_checkpoint, model, optimizer, data_state=data_state, DataParallel=DataParallel)

    if is_best_score:
        ID_best = ID + "_best"
        save_timepoint(ID_best, path_checkpoint, model, optimizer, data_state=data_state, DataParallel=DataParallel)

    return best_score, is_best_score



def fine_tune(path_fine_tune, model, DataParallel=False):

    if os.path.isfile(path_fine_tune):
        print(f"=> loading checkpoint {path_fine_tune}\n")
        checkpoint = torch.load(path_fine_tune)

        data_state = checkpoint['data_state']
        model_state = checkpoint['model_state']

        if DataParallel:
            for name in model.module.model_layers:
                if name == "last_fc": continue
                model.module.model_layers[name].load_state_dict(model_state[name])

        else:
            for name in model.model_layers:
                if name == "last_fc": continue
                model.model_layers[name].load_state_dict(model_state[name])
    else:
        print(f"=> no checkpoint found at {path_fine_tune}")

    return model
