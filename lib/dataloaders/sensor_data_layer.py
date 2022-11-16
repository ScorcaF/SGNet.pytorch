import os
import sys

import numpy as np
import torch
from torch.utils import data
import dill
import json
import random
import torch
from torch.utils.data import TensorDataset
import sys

import pandas as pd
import glob

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class SENSORDataLayer(data.Dataset):

    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.batch_size = args.batch_size

        if split == 'train':
            datasets = [r'/content/drive/MyDrive/driving_data/scorca_curved_1.csv',
                        r'/content/drive/MyDrive/driving_data/yacometti_curved_1.csv',
                        r'/content/drive/MyDrive/controller_data/train.csv']

        elif split == 'val':
            datasets = [r'/content/drive/MyDrive/driving_data/scorca_curved_2.csv',
                        r'/content/drive/MyDrive/driving_data/yacometti_curved_2.csv',
                        r'/content/drive/MyDrive/controller_data/val.csv']

        elif split == 'test_all':
            datasets = [r'/content/drive/MyDrive/driving_data/scorca_curved_3.csv',
                        r'/content/drive/MyDrive/driving_data/yacometti_curved_3.csv',
                        r'/content/drive/MyDrive/driving_data/scorca_straight_1.csv',
                        r'/content/drive/MyDrive/driving_data/yacometti_straight_1.csv',
                        ]
        elif split == 'test_scorca':
            datasets = [r'/content/drive/MyDrive/driving_data/scorca_curved_3.csv',
                        r'/content/drive/MyDrive/controller_data/test.csv',
                        r'/content/drive/MyDrive/driving_data/scorca_straight_1.csv', 
                        ]

        elif split == 'test_yacometti':   
            datasets = [r'/content/drive/MyDrive/driving_data/yacometti_curved_3.csv',
                        r'/content/drive/MyDrive/driving_data/yacometti_straight_1.csv',
            ]
       
        window_generator = WindowGenerator(
                                          prediction_horizon=args.dec_steps,
                                          history_horizon=args.enc_steps -1
                                          )
        X, Y = [], []
        for dataset in datasets:
            X_partial, Y_partial = window_generator.make_timeseries_dataset_from_csv(dataset)
            X.append(X_partial)
            Y.append(Y_partial)
        X = np.vstack(X)
        Y = np.vstack(Y)
        self.dataset = TensorDataset(torch.tensor(X),
                                    torch.tensor(Y))
        self.length = len(Y)
         
        # self.dataset = 
        # self.len_dict = {}
        # for index in range(len(self.dataset)):
        #     first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name,timestep = self.dataset.__getitem__(index)
        #     if first_history_index not in self.len_dict:
        #         self.len_dict[first_history_index] = []
        #     self.len_dict[first_history_index].append(index)
        # self.shuffle_dataset()

    # def shuffle_dataset(self):
    #     self._init_inputs()

    # def _init_inputs(self):
    #     '''
    #     shuffle the data based on its length
    #     '''
    #     self.inputs = []
    #     for length in self.len_dict:
    #         indices = self.len_dict[length]
    #         random.shuffle(indices)
    #         self.inputs.extend(list(chunks(self.len_dict[length], self.batch_size)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # first_history_index, x_t, y_t, x_st_t, y_st_t, scene_name, timestep = self.dataset.__getitem__(index)
        ret = {}
        x_t, y_t = self.dataset.__getitem__(index)
        # all_t = torch.cat((x_t[:,:2], y_t),dim=0)
        # y_t = self.get_target(all_t, 0, self.args.enc_steps, self.args.enc_steps, self.args.dec_steps)
        # ret['first_history_index'] = first_history_index
        ret['input_x'] = x_t
        # ret['input_x_st'] = x_st_t
        ret['target_y'] = y_t
        # ret['target_y_st'] = y_st_t
        # ret['scene_name'] = scene_name
        # ret['timestep'] = timestep
        return ret


class WindowGenerator:
    def __init__(self, prediction_horizon: int, history_horizon: int):
        self.prediction_horizon: int = prediction_horizon
        self.history_horizon: int = history_horizon


    def make_timeseries_dataset_from_csv(self, filename: str):
        
        df = pd.read_csv(filename)
        if filename == r'/content/drive/MyDrive/driving_data/yacometti_curved_1.csv':
          df = df[:3000]

        df_observations = self.preprocess(df)
        
        # TODO: remove
        if 'curved' in filename or 'straight'in filename:
          df_observations['yawAngle'] = df_observations['headingAngle']

        
        initial_observations_list, timeseries_list = self.make_timeseries_dataset(df_observations)

        return initial_observations_list, timeseries_list
    
    
    def make_timeseries_dataset(self, df: pd.DataFrame):
        X = []
        Y = []

        for t in range(len(df) - self.prediction_horizon - 1):

            if t >= self.history_horizon: 
              x = df.iloc[t - self.history_horizon: t + 1 , :].to_numpy().reshape(self.history_horizon +1, -1)
              y = df.iloc[t + 1: t + self.prediction_horizon + 1, :2].to_numpy()

            # else:
            #   x = df.iloc[t, :].to_numpy().reshape(1, -1)
            #   y = df.iloc[t + 1: t + self.prediction_horizon + 1, :2].to_numpy()

              X.append(x)
              Y.append(y)
        return np.array(X), np.array(Y)
      

    def preprocess(self, df: pd.DataFrame):

        columns_to_remove = ['timestamp', 'gear', 'RPM', 'fXbody', 'sideSlipeAngle', 'wheelAngularVel', 'groundtruth']
        if set(columns_to_remove) < set(df.columns):
          df_observations = df.drop(columns=columns_to_remove).reset_index(drop=True)
        elif set(['Travel_distance_X (m)', 'Travel_distance_Y (m)']) < set(df.columns):
          columns_to_remove = ['Travel_distance_X (m)', 'Travel_distance_Y (m)']
          df_observations = df.drop(columns=columns_to_remove).reset_index(drop=True)

        df_observations = df_observations.drop(columns=['steering', 'throttle', 'brake'])
        ordered_columns = [
            'LongGPS', 'LatGPS',
            'LongVel', 'LatVel',
            'LongAcc', 'LatAcc',
            'yawAngle', 'yawRate',
            'lDistLane', 'rDistLane',
            'lCurvLane', 'rCurvLane',
            'lCurvDevLane', 'rCurvDevLane',
            'headingAngle',
        ]

        df_observations = df_observations.reindex(columns=ordered_columns)

        return df_observations


