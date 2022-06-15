from LSTMAttention import AttentionLSTM
from LSTM import LSTM
import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from model_utils import train
from torch.optim.lr_scheduler import StepLR
from utils import *
import argparse
from datetime import datetime
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("-in","--input_path",help="path of file csv",type=str,default="data/train/sensor_1.csv")
parser.add_argument("-ep", "--epochs", help="Number of training epochs", type=int, default=200)
parser.add_argument("-in_seq","--input_seq_len",help="input sequence length",type=int,default=60)
parser.add_argument("-out_seq","--output_seq_len",help="output sequence length",type=int,default=60)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-4)
parser.add_argument("-path","--model_path",help="path of save model",type=str,default="checkpoint/")
parser.add_argument("-hsize","--hidden_size",help="hidden size",type=int,default=32)
parser.add_argument("-ksize","--kernel_size",help="hidden size",type=int,default=3)
parser.add_argument("-embed_size","--embed_size",help="hidden size",type=int,default=8)
parser.add_argument("-num_lay","--number_layer",help="number of LSTM layer",type=int,default=1)
args = parser.parse_args()
dir = 'data/train'
lengths = [30, 45, 60]
for path in os.listdir(dir):
  if 'sensor_7.csv' in path:
   for length in lengths:     
    wandb.init()

    (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator
    ) = prepare_2(
        # input_path=args.input_path,
        input_path=os.path.join(dir, path),
        # input_len=args.input_seq_len,
        input_len=length,
        output_len=args.output_seq_len,
    )

    model = AttentionLSTM(
            # input_seq_len=args.input_seq_len,
            input_seq_len=length,
            output_seq_len=args.output_seq_len,
            number_layer=args.number_layer,
            input_size=len(train_df.columns),
            output_size=len(train_df.columns),
            hidden_size=args.hidden_size,
            device=device,
            kernel_size=args.kernel_size,
            embed_size=args.embed_size
        )
    
    model.to(device)
    now = datetime.now()

    str_now = now.strftime("%m_%d_%Y_%H_%M")
    name = f'{length}_AttentionLSTM_{path[:-4]}'
    model_path = os.path.join(args.model_path, name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dict_infor = vars(args)
    save_train_info(dict_infor, model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    train(
        model=model,
        train_iterator=train_iterator,
        valid_iterator=valid_iterator,
        num_epochs=args.epochs,
        criterion=criterion,
        model_path=model_path,
        optimizer=optimizer,
        scheduler=scheduler,
    )