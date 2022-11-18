

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def freeze_layers(net, FilterPos, DataParallel = False):
    if DataParallel:
        for param_name, param in net.module.named_parameters():
            param.requires_grad = False
            for layer_id in FilterPos:
                if layer_id in param_name:
                    param.requires_grad = True
    else:
        for param_name, param in net.named_parameters():
            param.requires_grad = False
            for layer_id in FilterPos:
                if layer_id in param_name:
                    print("fffreeze_layers ", layer_id)
                    param.requires_grad = True


def unfreeze_layers(net, FilterNeg, DataParallel = False):
    if DataParallel:
        for param_name, param in net.module.named_parameters():
            param.requires_grad = True
            for layer_id in FilterNeg:
                if layer_id in param_name:
                    param.requires_grad = False
    else:
        for param_name, param in net.named_parameters():
            param.requires_grad = True
            for layer_id in FilterNeg:
                if layer_id in param_name:
                    print("unfreeze_layers ", layer_id)
                    param.requires_grad = False

def count_free_layers(net,DataParallel = False):

    if DataParallel:
        count_free, count_freeze = 0 , 0
        for param in net.module.parameters():
            if param.requires_grad:
                count_free +=1
            else:
                count_freeze +=1
    else:
        count_free, count_freeze = 0 , 0
        for param in net.parameters():
            if param.requires_grad:
                count_free +=1
            else:
                count_freeze +=1

        if count_freeze < 10:
            for name, param in net.named_parameters():
                if not param.requires_grad:
                    print("freeze", name)
    return count_free, count_freeze


def freeze_policy(args, net, epoch, optimizer, DataParallel=False):
    FilterFreezeA, FilterFreezeB = {}, {}
    for i in range (0,len(args.net_optim_policy["freeze_layers"]),2):
        layer_id = args.net_optim_policy["freeze_layers"][i]
        epoch_i = args.net_optim_policy["freeze_layers"][i+1]

        if epoch == epoch_i:
            FilterFreezeA[layer_id] = 1

    for i in range(0, len(args.net_optim_policy["unfreeze_layers"]), 2):
        layer_id = args.net_optim_policy["unfreeze_layers"][i]
        epoch_start = args.net_optim_policy["unfreeze_layers"][i + 1]

        if epoch < epoch_start:
            FilterFreezeB[layer_id] = 1

    if len(FilterFreezeA) > 0:
        freeze_layers(net, FilterFreezeA, DataParallel=DataParallel)
    else:
        unfreeze_layers(net, FilterFreezeB, DataParallel=DataParallel)




def gradient_policy(args, epoch, optimizer):
    #exit()
    lr_decay = args["optimizer_param"]["lr_decay"]
    print("lr_decay", lr_decay, len(lr_decay))
    for i in range(0,len(lr_decay),2):

        if epoch >= lr_decay[i+1]:
            optimizer.param_groups[-1]['lr'] = lr_decay[i] * args["optimizer_param"]["lr"]
    print("run_epoch lr", optimizer.param_groups[-1]['lr'], "epoch:", epoch)



def get_optim_policy(args, net, epoch, optimizer, DataParallel=False):

    freeze_policy(args, net, epoch, optimizer, DataParallel)
    gradient_policy(args, epoch, optimizer)

    if args.net_optim_policy["freeze_BN_layers"] < epoch:
        freeze_BN_layers(net, DataParallel)

    if args.net_optim_policy["freeze_BLOCK_layers_soft"] < epoch:
        freeze_BLOCK_layers_soft(net, epoch, DataParallel)

    count_free, count_freeze = count_free_layers(net, DataParallel=DataParallel)
    print("count_free_layers, count_freeze_layers", count_free, count_freeze)

    print("args.net_optim_policy", args.net_optim_policy)
    print("run_epoch lr", optimizer.param_groups[-1]['lr'], "epoch:", epoch)





def freeze_BN_layers(net,  DataParallel = False):

    print("freeze_BN_layers")

    if DataParallel:
        for param_name, param in net.module.named_parameters():
            if "norm" in param_name: param.requires_grad = False
            if "bn" in param_name: param.requires_grad = False
    else:
        for param_name, param in net.named_parameters():
            if "norm"  in param_name: param.requires_grad = False
            if "bn"  in param_name: param.requires_grad = False


def freeze_BLOCK_layers_hard(net, epoch,  DataParallel = False):

    print("freeze_BLOCK_layers_hard", epoch)

    if DataParallel:
        for param_name, param in net.module.named_parameters():
            param.requires_grad = False
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        print("param_name", param_name)
                        param.requires_grad = True
    else:
        for param_name, param in net.named_parameters():
            param.requires_grad = False
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        print("param_name", param_name)
                        param.requires_grad = True


def freeze_BLOCK_layers_soft(net, epoch,  DataParallel = False):

    print("freeze_BLOCK_layers_soft", epoch)

    if DataParallel:
        for param_name, param in net.module.named_parameters():
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        #print("freeze_BLOCK_layers_soft", param_name)
                        param.requires_grad = False
    else:
        for param_name, param in net.named_parameters():
            for s in range(4):
                if epoch % 4 == s:
                    if f"layer{s + 1}" in param_name:
                        #print("param_name", param_name)
                        param.requires_grad = False



def restart_fc_layers(net, DataParallel = False):

    print("restart_fc_layers")

    if DataParallel:

        if len(net.module.num_class) >= 1:
            normal_(net.module.model.last_fc_0.weight, 0, 0.001)
            constant_(net.module.model.last_fc_0.bias, 0)
        if len(net.module.num_class) >= 2:
            normal_(net.module.model.last_fc_1.weight, 0, 0.001)
            constant_(net.module.model.last_fc_1.bias, 0)
        if len(self.num_class) >= 3:
            normal_(net.module.model.last_fc_2.weight, 0, 0.001)
            constant_(net.module.model.last_fc_2.bias, 0)

    else:
        if len(net.num_class) >= 1:
            normal_(net.model.last_fc_0.weight, 0, 0.001)
            constant_(net.model.last_fc_0.bias, 0)
        if len(net.num_class) >= 2:
            normal_(net.model.last_fc_1.weight, 0, 0.001)
            constant_(net.model.last_fc_1.bias, 0)
        if len(self.num_class) >= 3:
            normal_(net.model.last_fc_2.weight, 0, 0.001)
            constant_(net.model.last_fc_2.bias, 0)


if __name__ == '__main__':


    print('Test passed.')




