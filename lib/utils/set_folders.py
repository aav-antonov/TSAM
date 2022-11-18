import os
from datetime import date

from lib.utils.utils import saveargs


def args_store_name(args):

    if "run_id" in args:
        store_name = f'{args["run_id"]}'
    else:
        run_id = str(date.today().strftime("%d_%m_%Y"))
        store_name = f'{run_id}'

    store_name += f'_{args["dataset"]["name"]}'

    return store_name

def get_file_results(args):
    file_results = f'{args["output_folder"]}/result.json'
    return file_results

def define_rootfolders(args):
    store_name = args_store_name(args)
    args["output_folder"] = f'{args["results_folder"]}/{store_name}'
    args["checkpoint"] = f'{args["output_folder"]}/checkpoints'
    return args


def check_rootfolders(args):

    args = define_rootfolders(args)

    """Create root_folder and output_folder """
    folders_util = [args["results_folder"], args["output_folder"], args["checkpoint"]]

    for folder in folders_util:
        if os.path.exists(folder):
            print('folder exist ', folder)
        else:
            print('creating folder ' + folder)
            os.mkdir(folder)

    """saving args to checkpoint folder"""
    saveargs(args, f'{args["output_folder"]}/args.json')





