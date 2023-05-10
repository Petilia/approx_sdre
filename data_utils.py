import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_to_traj_and_control(data):
    X = data[:, 0:6]
    y = data[:, 6:]
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    dataset = TensorDataset(X, y)
    return dataset

def split_tracks(data_path="sdreDataset.csv",
                         LEN_TRACK=101):
    df = pd.read_csv(data_path, names=range(1, 10))
    N_TRACKS = int(df.shape[0] / LEN_TRACK)

    tracks = []
    for track_index in range(N_TRACKS):
        cur_track = df.values[track_index*LEN_TRACK : (track_index+1)*LEN_TRACK]
        tracks.append(cur_track)

    train_tracks, test_tracks = train_test_split(
            tracks, test_size=0.2, random_state=42)
    return train_tracks, test_tracks

def get_datasets(data_path="sdreDataset.csv",
                         LEN_TRACK=101):
    
    train_tracks, test_tracks = split_tracks(data_path, LEN_TRACK)

    train_data = np.vstack((train_tracks))
    test_data = np.vstack((test_tracks))

    train_dataset = split_data_to_traj_and_control(train_data)
    test_dataset = split_data_to_traj_and_control(test_data)
    return train_dataset, test_dataset

