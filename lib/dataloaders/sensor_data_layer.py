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

        prediction_horizon = 75
        dataset = r'/content/scorca_straight/driver_scorca_straight_1'
        window_generator = WindowGenerator(prediction_horizon)
        X, Y = window_generator.make_timeseries_dataset_from_csv(dataset)
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
    def __init__(self, prediction_horizon: int):
        self.prediction_horizon: int = prediction_horizon

    def make_timeseries_dataset_from_csv(self, filename: str):
      
        df_vehicle, df_camera = self.load(filename)
        df_observations_commands = self.preprocess(df_vehicle, df_camera)
        initial_observations_list, timeseries_list = self.make_timeseries_dataset(df_observations_commands)
        return initial_observations_list, timeseries_list

    def make_timeseries_dataset(self, df: pd.DataFrame):
        X = []
        Y = []

        for t in range(len(df) - self.prediction_horizon - 1):
            x = df.iloc[t, :6].to_numpy().reshape(1, -1)
            y = df.iloc[t + 1: t + self.prediction_horizon + 1, :2].to_numpy()

            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def load(self, filename: str):
        for csv in glob.glob(f'{filename}*'):
            if csv.split('_')[-2] == 'sim':
                df_vehicle = pd.read_csv(csv)
            elif csv.split('_')[-2] == 'sensor':
                df_camera = pd.read_csv(csv)
        return df_vehicle, df_camera

    def time_str_to_float(self, time):
        time = time.split(' ')
        time = time[0]
        return float(time)

    def rescale_signal(self, signal):
        '''Rescaling from [-1, 1] range
        to [0, 1]'''
        signal = (signal + 1) / 2
        return signal

    def preprocess(self, df_vehicle: pd.DataFrame, df_camera: pd.DataFrame):
        df_camera['Time'] = df_camera['Time'].map(self.time_str_to_float)
        df_vehicle['Time'] = df_vehicle['Time'].map(self.time_str_to_float)

        df = df_vehicle.merge(df_camera, on='Time')
        df = df.drop_duplicates(subset='Time')
        df.reset_index(drop=True, inplace=True)

        df['Brake Cmd (%)'] = df['Brake Cmd (%)'].map(self.rescale_signal)
        df['Acceleration Cmd (%)'] = df['Acceleration Cmd (%)'].map(self.rescale_signal)

        for signal in ['Acceleration Cmd (%)', 'Brake Cmd (%)', 'Steering Cmd (deg)']:
            for i, value in enumerate(df[signal]):
                if value == 0.5:
                    df.loc[i, signal] = 0
                else:
                    break

        df_observations = df.drop(columns=['Yaw Acceleration (rad/s^2)',
                                           'Distraction (X)', 'Engine Torque (Nm)', 'Engine Speed (deg/s)',
                                           'Gear (1)', 'Wind X (m/s)', 'Wind Y (m/s)', 'Lateral Slip FL (deg)',
                                           'Lateral Slip FR (deg)', 'Lateral Slip RL (deg)',
                                           'Lateral Slip RR (deg)', 'Strength Left (1)', 'Heading Angle Right (rad)',
                                           'Strength Right (1)'])
        df_observations.reset_index(drop=True, inplace=True)

        # Add travel distances
        # last_index = df_observations.shape[0] - 1
        # df_zeros = pd.DataFrame(0, index=np.arange(1), columns=['Travel_distance_X (m)', 'Travel_distance_Y (m)'])
        # df_offsets = abs(df_observations.loc[1:, ['Position_X (m)', 'Position_Y (m)']].reset_index(drop=True) \
        #                  - df_observations.loc[:last_index, ['Position_X (m)', 'Position_Y (m)']].reset_index(
        #     drop=True))
        # df_offsets = df_offsets.rename(columns={'Position_X (m)': 'Travel_distance_X (m)',
        #                                         'Position_Y (m)': 'Travel_distance_Y (m)'
        #                                         })
        # df_offsets = df_offsets.drop(index=last_index)
        # df_offsets = pd.concat((df_zeros, df_offsets)).reset_index(drop=True)
        # df_offsets['Time'] = df_observations['Time']
        # df_observations = df_observations.merge(df_offsets, on='Time')

        df_observations = df_observations.drop(columns=['Time', 'Steering Cmd (deg)', 'Acceleration Cmd (%)', 'Brake Cmd (%)'])
        ordered_columns = [
            'Position_X (m)', 'Position_Y (m)',
            'Velocity_X (m/s)', 'Velocity_Y (m/s)',
            'Acceleration_X (m/s^2)', 'Acceleration_Y (g)',
            'Yaw Angle (rad)', 'Yaw Rate (rad/s)',
            'Lateral Offset Left (m)', 'Lateral Offset Right (m)',
            'Curvature Left (1)', 'Curvature Right (1)',
            'Curvature Derivative Left (1)', 'Curvature Derivative Right (1)',
            'Heading Angle Left (rad)',
            # 'Travel_distance_X (m)', 'Travel_distance_Y (m)',
        ]
        df_observations['Position_Y (m)'] *= -1

        df_observations = df_observations.reindex(columns=ordered_columns)
        # TODO: remove
        # this is an adjustment that will work (if it will) only on straight roads: need to remap from xy to longlat
        df_observations['Yaw Angle (rad)'] *= -1
        df_observations['Velocity_Y (m/s)'] *= -1
        df_observations['Acceleration_Y (g)'] *= -1  # g?
        
        return df_observations
