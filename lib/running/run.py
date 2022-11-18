
import json
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score , accuracy_score
import warnings
from sklearn.exceptions import DataConversionWarning
from lib.dataset.dataset import GetDataSet, GetDataLoaders, test_input
from lib.utils.utils import   AverageMeter , get_num_class
from lib.utils.set_folders import  get_file_results
from lib.utils.saveloadmodels import checkpoint_acc , save_timepoint , load_model , fine_tune


""" TRAINING epoch"""
def run_epoch(args, run_data, DataLoaders, model, device, optimizer, criterion, DataParallel = False):

    [best_score, history_score, epoch] = run_data

    [TrainDataLoader, ValidDataLoader] = DataLoaders

    """train and validate rounds"""
    train_score  = run_step(TrainDataLoader, model, device, criterion, optimizer = optimizer)
    print(f'Training: epoch: {epoch} accuracy_balanced: {round(float(train_score["balanced"]) * 100, 2)} accuracy: {round(float(train_score["unbalanced"]) * 100, 2)}')

    val_score = run_step(ValidDataLoader, model, device, criterion, epoch,training=False)
    print(f'Validation: epoch: {epoch} accuracy_balanced: {round(float(val_score["balanced"]) * 100, 2)} accuracy: {round(float(val_score["unbalanced"]) * 100, 2)}')

    """checkpoint: saves model to args["checkpoint"]"""
    best_score["unbalanced"], is_best_score = checkpoint_acc("unbalanced", args["checkpoint"],epoch, best_score["unbalanced"], val_score["unbalanced"],  model, optimizer, DataParallel)
    best_score["balanced"], is_best_score = checkpoint_acc("balanced", args["checkpoint"], epoch, best_score["balanced"], val_score["balanced"], model, optimizer, DataParallel)

    """saving results to file_results at args.results_folder"""
    history_score[epoch] = {"training": train_score, "valid": val_score}
    result = {'best_score': best_score,  'history_score': history_score}
    file_results = get_file_results(args)
    with open(file_results, 'w') as f:
        json.dump(result, f, indent=4)

""" TRAINING/VALIDATION round"""
def run_step( loader, model, device, criterion, optimizer = None, training=True):

    score = {}
    score["balanced"], score["unbalanced"] = 0.00, 0.00
    losses_avg, accuracy_avg = AverageMeter(), AverageMeter()
    y_pred, y_true = [], []

    # switch to train/eval mode
    if training:
        model.train()
        lr = optimizer.param_groups[-1]['lr']
        print(f'training leaning rate: {float(lr)} ')
    else:
        model.eval()


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        with tqdm(total=len(loader)) as t:
            for i, (ID,  y , x_video, x_audio) in enumerate(loader):

                if i == 0:
                    if x_video[0] != "None": print("x_video", x_video.size())
                    if x_audio[0] != "None": print("x_audio", x_audio.size())

                if x_video[0] != "None": x_video = x_video.to(device)
                if x_audio[0] != "None": x_audio = x_audio.to(device)
                y = y.long().to(device)

                output  = model(x_video, x_audio)

                loss = criterion(output, y)
                losses_avg.update(loss.item(), y.size(0))

                _, y_pred_max = torch.max(output.data.cpu(), 1)
                y_pred_new = y_pred_max.detach().cpu().tolist()
                y_true_new = y.cpu().tolist()
                y_pred.extend(y_pred_new)
                y_true.extend(y_true_new)

                acc = accuracy_score(y_true_new, y_pred_new)
                accuracy_avg.update(acc, y.size(0))

                if training:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                t.set_postfix( loss='{:05.5f}'.format(float(losses_avg.avg)),
                               accuracy = '{:05.3f}'.format(float(accuracy_avg.avg)*100)
                       )
                t.update()

    score["unbalanced"] = accuracy_score(y_true, y_pred)
    score["balanced"] = balanced_accuracy_score(y_true, y_pred)

    return score

""" MultiClip Test/Validation Rounds"""
def run_MultiClip( args, ValidDataSet , model, device, MultiClip = 10):

    print("run_MultiClip", MultiClip)

    ValidDataSet.mode_train_val = "MultiClip"
    loader = GetDataLoaders(ValidDataSet, args, mode_train_val="MultiClip")

    # switch to train/eval mode
    model.eval()

    acc_MultiClip = {}
    for m in range(MultiClip):
        output_list = []
        y_list = []
        with tqdm(total=len(loader)) as t:
            for i, (ID,  y , x_video, x_audio) in enumerate(loader):

                if i == 0:
                    if x_video[0] != "None": print("x_video", x_video.size())
                    if x_audio[0] != "None": print("x_audio", x_audio.size())

                if x_video[0] != "None": x_video = x_video.to(device)
                if x_audio[0] != "None": x_audio = x_audio.to(device)

                y_list.append(y)
                output  = model(x_video, x_audio).data.cpu()
                output_list.append(output)

                t.update()

        output_stack = torch.cat(output_list, dim=0)
        y_stack = torch.cat(y_list, dim=0)
        if m == 0:
            output_stack_add =  output_stack #torch.square(RELU(output_stack))
        else:
            output_stack_add = torch.add(output_stack_add, output_stack,alpha=1) #torch.square(RELU(output_stack))

        _, y_pred_max = torch.max(output_stack_add, 1)
        acc = accuracy_score(y_stack, y_pred_max)
        acc_MultiClip[m] = acc
        print("acc", m, acc)

    acc_max = max(acc_MultiClip, key=acc_MultiClip.get)
    print("Best accuracy:", acc_MultiClip[acc_max])







