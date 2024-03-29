import logging
from argparse import ArgumentParser
import pandas as pd
import torch


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--log_dir', default='/home/coopers/log', help='Log dir [default: ]')
    parser.add_argument('--notes', default='', help='Experiments notes [default: None]')
    parser.add_argument('--model_path',
                        # default='C:/Users/sharon/Documents/Research/ObjectCompletion3D/model/')
                        default='/home/coopers/models/')
    parser.add_argument('--train_path',
                        # default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\train\\gt\\02691156\\')
                        default='/home/coopers/data/train/gt/02691156/')
    parser.add_argument('--val_path',
                        # default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\val\\gt\\02691156\\')
                        default='/home/coopers/data/val/gt/02691156/')
    parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run [default: 100]')
    parser.add_argument('--bins', type=int, default=10, help='resolution of main cube [default: 10]')
    parser.add_argument('--samples_per_face', type=int, default=100, help='number of samples per voxel [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 1]')
    parser.add_argument('--threshold', default=0.5, help='cube probability threshold')
    parser.add_argument('--optimizer', default="sgd", help='should be either adam or SGD [default: sgd]')
    parser.add_argument('--lr', default=0.1, type=float, help='optimizer learning rate')
    parser.add_argument('--min_lr', default=0.001, type=float, help='minimum optimizer learning rate [default: 0.001]')
    parser.add_argument('--step_size', default=200, type=float, help='step_size epochs for learning rate wrapper')
    parser.add_argument('--gamma', default=0.1, type=float, help='Decays the learning rate of each parameter'
                                                                 ' group by gamma every step_size epochs')
    parser.add_argument('--momentum', default=0.09, type=float, help='cube probability threshold')
    parser.add_argument('--cd_coeff', type=float, default=1.)
    parser.add_argument('--fn_coeff', type=float, default=1.)
    parser.add_argument('--bce_coeff', type=float, default=1.)
    parser.add_argument('--reg_start_iter', type=int, default=150)

    return parser.parse_args()


def update_tracking(
        id, data=None, csv_file="./tracking.csv", nround=6,
        drop_broken_runs=False, field_mark_done="end_time"):
    """
    Tracking function for keep track of model parameters and
    CV scores. `integer` forces the value to be an int.
    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except FileNotFoundError:
        df = pd.DataFrame()
    if drop_broken_runs:
        try:
            df = df.dropna(subset=[field_mark_done])
        except KeyError:
            logging.warning("No loss column found in tracking file")
    if id in df.index:
        # append dictionary to existing row
        for field, value in data.items():
            df.loc[id, field] = value
    else:
        df = df.append(pd.DataFrame(data=data, index=[id]))
    df.to_csv(csv_file)


def detach_dict(d):
    state_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.cpu().detach().numpy()
    return state_dict