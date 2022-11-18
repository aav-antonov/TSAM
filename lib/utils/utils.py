import json
import os


from collections import Counter
from sklearn.metrics import balanced_accuracy_score , accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def saveargs(args,json_path):
    with open(json_path, 'w') as f:
         json.dump( args, f, indent=4)


def loadarg(json_path):
    with open(json_path) as f:
        args = json.load(f)
    return args




def create_clean_DIR(path):
 try:
    os.mkdir(path)
 except OSError:
    path_ids = os.listdir(path)
    print ("Dir %s exist" % path)
    count_rm = 0
    for fileA in path_ids:
     if(len(fileA) > 2):
      #print(fileA)
      if os.path.isdir(path + "/" + fileA):
          path_ids2 = os.listdir(path + "/" + fileA)
          for fileB in path_ids2:
              if os.path.isdir(f"{path}/{fileA}/{fileB}"):continue
              os.remove(f"{path}/{fileA}/{fileB}")
              count_rm += 1
          continue
      os.remove(path + "/" + fileA)
      count_rm +=1
    print ("Dir" ,  path, "cleaned, files removed = " , count_rm)
 else:
    print ("Successfully created the directory %s " % path)

def get_num_class(args):
    file_train_list = f'{args["dataset"]["data_dir"]}/{args["dataset"]["file_train_list"]}'
    lines = [x.strip().split(' ') for x in open(file_train_list)]
    stat = Counter()
    for item in lines: stat[item[1]] += 1
    num_class= len(stat)
    return num_class



def check_true_pred(y_pred,y_true):
    s_pred = set(y_pred)
    s_true = set(y_true)
    print("check_true_pred Pred - True:", s_pred.difference(s_true))
    print("check_true_pred True - Pred:", s_true.difference(s_pred))


def get_label_class(value, threthholds):
    for j,vth in enumerate(threthholds):
        if value < vth:
            return j
    return len(threthholds)


def get_labels(Y_true_max, Y_pred_max, threthholds):
    Y_true_label, Y_pred_label = [], []
    for i, ID in enumerate(Y_true_max):
        Y_true_label.append(get_label_class(Y_true_max[ID], threthholds))
        Y_pred_label.append(get_label_class(Y_pred_max[ID], threthholds))
    return Y_true_label, Y_pred_label

def estimate_accuracy(Y_true_max, Y_pred_max, v):

    threthholds = [v]

    Y_true_label, Y_pred_label = get_labels(Y_true_max, Y_pred_max, threthholds)
    print("sum:", sum(Y_true_label), sum(Y_pred_label))
    score = {}
    score["unbalanced"] = accuracy_score(Y_true_label, Y_pred_label)
    score["balanced"] = balanced_accuracy_score(Y_true_label, Y_pred_label)
    return score








