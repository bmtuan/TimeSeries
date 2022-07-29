from LSTMAttention import AttentionLSTM, MultiTaskLSTM, ClusteringMultiTaskLSTM
from torch.optim import Adam, SGD
import torch.nn as nn
from model_utils import *
from torch.optim.lr_scheduler import StepLR
from utils import *
import argparse
from datetime import datetime
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("-in_seq","--input_seq_len",help="input sequence length",type=int,default=10)
parser.add_argument("-out_seq","--output_seq_len",help="output sequence length",type=int,default=10)
parser.add_argument("-path","--model_path",help="path of save model",type=str,default="checkpoint/")
parser.add_argument("-hsize","--hidden_size",help="hidden size",type=int,default=32)
parser.add_argument("-num_lay","--number_layer",help="number of LSTM layer",type=int,default=1)
args = parser.parse_args()
dir = 'data/multitask_train'
model_path = 'checkpoint/ClusteringMultiTaskLSTM_0707/best.pth'
cluster = [1 ,2 ,1, 1, 2, 0, 2, 0]
(
    train_df,
    valid_df,
    test_df,
    train_iterator,
    valid_iterator,
    test_iterator
) = prepare_multitask(
    input_path=dir,
    input_len=args.input_seq_len,
    output_len=args.output_seq_len,
)
model = ClusteringMultiTaskLSTM(
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        number_layer=args.number_layer,
        input_size=len(train_df.columns),
        output_size=len(train_df.columns),
        hidden_size=args.hidden_size,
        cluster=cluster,
        device=device,
    )

model.to(device)

model.load_state_dict(copyStateDict(torch.load(model_path)))

mutil_predict(model, test_iterator, 8)