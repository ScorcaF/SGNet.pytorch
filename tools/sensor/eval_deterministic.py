import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
sys.path.append(os.getcwd())

from configs.sensor.sensor import parse_sgnet_args as parse_args
import lib.utils as utl
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.sensor_train_utils import train, test


def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = args.save_dir + '/test
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)
    model = model.to(device)
    if osp.isfile(args.checkpoint):

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint


    criterion = rmse_loss().to(device)
    
    if os.path.exists(f'{save_dir}/results.csv'):
      results = pd.read_csv(f'{save_dir}/results.csv')
      print('Loading results')
    else:
      results = pd.DataFrame()
    for test_split in ['test_wheel', 'test_controller']:
        
        test_gen = utl.build_data_loader(args, 'test', batch_size = 1)
        print("Number of test samples:", test_gen.__len__())
        

        # test
        test_loss, ADE, FDE, = test(model, test_gen, criterion, device)

        results = results.append({
                    'enc_steps' : args.enc_steps
                    'dec_steps': args.dec_steps,
                    'seed': args.seed,
                    'ADE': ADE,
                    'FDE': FDE}, ignore_index=True)
        
    results.to_csv(f'{save_dir}/results.csv', index = False)
        

if __name__ == '__main__':
    main(parse_args())
