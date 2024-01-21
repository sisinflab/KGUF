import pickle

import pandas as pd
import torch
from torch_sparse import SparseTensor
import numpy as np
from collections import Counter

def LoadEdgeFeatures(item_features_path: str):
    print('loading item features')
    with open(item_features_path, 'rb') as file:
        item_features = pickle.load(file)
    print(f'item features loaded from \'{item_features_path}\'')
    return  item_features
