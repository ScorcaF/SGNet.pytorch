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
import pandas as pd

sys.path.append(os.getcwd())
from configs.sensor.sensor import parse_sgnet_args as parse_args
import lib.utils as utl
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.sensor_train_utils import train, test



def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = args.save_dir + \
                str(args.seed) + \
                '_hist' + \
                str(args.enc_steps) + \
                '_horz' + \
                str(args.dec_steps)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                           min_lr=1e-10, verbose=1)
    model = model.to(device)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        args.start_epoch += checkpoint['epoch']
        del checkpoint


    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train', batch_size = args.batch_size)
    val_gen = utl.build_data_loader(args, 'val', batch_size = args.batch_size)
    print("Number of train samples:", train_gen.__len__() * args.batch_size)
    print("Number of val samples:", val_gen.__len__()* args.batch_size)
    # train
    min_loss = 1e6
    min_ADE = 10e5
    min_FDE = 10e5
    best_model = None
    best_model_metric = None

    if os.path.exists(f'{save_dir}/results_{args.dec_steps}_{args.seed}.csv'):
      results = pd.read_csv(f'{save_dir}/results_{args.dec_steps}_{args.seed}.csv')
      print('Loading results')
    else:
      results = pd.DataFrame()
    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):
        train_goal_loss, train_dec_loss, total_train_loss = train(model, train_gen, criterion, optimizer, device)
        

        print('Train Epoch: {} \t Goal loss: {:.4f}\t Decoder loss: {:.4f}\t Total: {:.4f}'.format(
                epoch, train_goal_loss, train_dec_loss, total_train_loss))

        # val
        val_loss, ADE, FDE = test(model, val_gen, criterion, device)
        results = results.append({
                    'epoch': epoch,
                    'train_goal_loss': train_goal_loss,
                    'train_dec_loss': train_dec_loss,
                    'total_train_loss': total_train_loss,
                    'val_loss': val_loss,
                    'ADE': ADE,
                    'FDE': FDE}, ignore_index=True)
        results.to_csv(f'{save_dir}/results_{args.dec_steps}_{args.seed}.csv', index = False)
        if ADE < min_ADE:
          min_ADE = ADE
          torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            save_dir + '.pth')

          print('Model saved to ', save_dir + '.pth')
        # lr_scheduler.step(val_loss)




if __name__ == '__main__':
    main(parse_args())
