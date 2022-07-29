from LSTMAttention import AttentionLSTM, MultiTaskLSTM
# from Multitask_LSTM import MultiTaskLSTM
from torch.optim import Adam, SGD
import torch.nn as nn
from model_utils import train
from torch.optim.lr_scheduler import StepLR
from utils import *
import argparse
from datetime import datetime
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("-ep", "--epochs", help="Number of training epochs", type=int, default=300)
parser.add_argument("-in_seq","--input_seq_len",help="input sequence length",type=int,default=10)
parser.add_argument("-out_seq","--output_seq_len",help="output sequence length",type=int,default=10)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("-path","--model_path",help="path of save model",type=str,default="checkpoint/")
parser.add_argument("-hsize","--hidden_size",help="hidden size",type=int,default=32)
parser.add_argument("-num_lay","--number_layer",help="number of LSTM layer",type=int,default=1)
args = parser.parse_args()
dir = 'data/multitask_train'
wandb.init()
# for i in range(8):
# if i == 7: 
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
    # task=i
)
# print(train_df)
model = MultiTaskLSTM(
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        number_layer=args.number_layer,
        input_size=len(train_df.columns),
        output_size=len(train_df.columns),
        hidden_size=args.hidden_size,
        device=device,
    )

model.to(device)
now = datetime.now()

# str_now = now.strftime("%m_%d_%Y_%H_%M")
# name = f'Single_Task_{i+1}'
name = 'MultiTaskLSTM_0507'
model_path = os.path.join(args.model_path, name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

dict_infor = vars(args)
save_train_info(dict_infor, model_path)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()
scheduler = StepLR(optimizer, step_size=25, gamma=0.9)
# print(train_df)
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