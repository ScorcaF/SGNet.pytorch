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

from lib.utils.eval_utils import eval_sensor


def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.
    count = 0
    total_goal_loss = 0
    total_dec_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            # first_history_index = data['first_history_index']
            # assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            # TODO: improve reshaping 
            input_traj = data['input_x'].to(device)[None, :, :]
            # input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)[None, None, :, :]

            # target_bbox_st = data['target_y_st'].to(device)

            all_goal_traj, all_dec_traj = model(input_traj.float())

            goal_loss = criterion(all_goal_traj, target_traj)
            dec_loss = criterion(all_dec_traj, target_traj)

            train_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += dec_loss.item()* batch_size


            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= count
    total_dec_loss /= count

    
    return total_goal_loss, total_dec_loss, total_goal_loss + total_dec_loss

# def val(model, val_gen, criterion, device):
#     total_goal_loss = 0
#     total_dec_loss = 0
#     count = 0
#     model.eval()
#     loader = tqdm(val_gen, total=len(val_gen))
#     with torch.set_grad_enabled(False):
#         for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
#             first_history_index = data['first_history_index']
#             assert torch.unique(first_history_index).shape[0] == 1
#             batch_size = data['input_x'].shape[0]
#             count += batch_size
            
#             input_traj = data['input_x'].to(device)
#             input_bbox_st = data['input_x_st'].to(device)
#             target_traj = data['target_y'].to(device)
#             # target_bbox_st = data['target_y_st'].to(device)

#             all_goal_traj, all_dec_traj = model(input_traj, first_history_index[0])


#             goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
#             dec_loss = criterion(all_dec_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

#             total_goal_loss += goal_loss.item()* batch_size
#             total_dec_loss += dec_loss.item()* batch_size

#     val_loss = total_goal_loss/count + total_dec_loss/count
#     return val_loss  

def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_dec_loss = 0
    ADE = 0
    FDE = 0 
    count = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):

            # first_history_index = data['first_history_index']
            # assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            # TODO: improve reshaping 
            input_traj = data['input_x'].to(device)[None, :, :]
            # input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)[None, None, :, :]

            all_goal_traj, all_dec_traj = model(input_traj.float())
            goal_loss = criterion(all_goal_traj, target_traj)
            dec_loss = criterion(all_dec_traj, target_traj)

            train_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += dec_loss.item()* batch_size

            all_dec_traj_np = all_dec_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            # Decoder
            batch_ADE, batch_FDE,  =\
                eval_sensor(input_traj_np, target_traj_np, all_dec_traj_np)

            ADE += batch_ADE
            FDE += batch_FDE            
    ADE/= count
    FDE /= count
    

    test_loss = total_goal_loss/count + total_dec_loss/count

    print("ADE: %4f;   FDE: %4f\n" % (ADE, FDE))
    return test_loss, ADE, FDE
