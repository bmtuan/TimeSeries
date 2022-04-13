import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import trange, tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
import copy
import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
