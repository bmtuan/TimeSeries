from LSTMAttention import AttentionLSTM
from LSTM import LSTM
import torch
import torch.nn as nn
from model_utils import *
from torch.optim.lr_scheduler import StepLR
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-in","--input_path",help="path of file csv",type=str,default="data/train/sensor_7.csv")
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
dir = '/media/aimenext/disk1/tuanbm/TimeSerires/model/data/train'
list_path = [
    # '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/60_AttentionLSTM_sensor_1/best.pth',
    # '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/45_AttentionLSTM_sensor_1/best.pth',
    # '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/30_AttentionLSTM_sensor_1/best.pth'
    # '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/60_AttentionLSTM_sensor_14/best.pth'
    # '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/60_AttentionLSTM_sensor_7/best.pth',
    # '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/45_AttentionLSTM_sensor_7/best.pth',
    # '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/30_AttentionLSTM_sensor_7/best.pth',
    '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/60_AttentionLSTM_sensor_22/best.pth',
    '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/45_AttentionLSTM_sensor_22/best.pth',
    '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/30_AttentionLSTM_sensor_22/best.pth'
    ]
input_lengths = [60, 45, 30]
for path in os.listdir(dir):
  if 'sensor_22.csv' in path:    
    print(path)
    list_model = []
    (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator
    ) = prepare_2(
        # input_path=args.input_path,
        input_path=os.path.join(dir,path),
        input_len=args.input_seq_len,
        output_len=args.output_seq_len,
    )
    for model_path in list_path:
        model = AttentionLSTM(
                input_seq_len=args.input_seq_len,
                output_seq_len=args.output_seq_len,
                number_layer=args.number_layer,
                input_size=len(train_df.columns),
                output_size=len(train_df.columns),
                hidden_size=args.hidden_size,
                device=device,
                kernel_size=args.kernel_size,
                embed_size=args.embed_size
            ) 
        # model = LSTM(
        #         input_seq_len=args.input_seq_len,
        #         output_seq_len=args.output_seq_len,
        #         number_layer=args.number_layer,
        #         input_size=len(train_df.columns),
        #         output_size=len(train_df.columns),
        #         hidden_size=args.hidden_size,
        #         device=device,
        #         kernel_size=args.kernel_size,
        #         embed_size=args.embed_size
        #     )
        model.to(device)
        model.load_state_dict(copyStateDict(torch.load(model_path)))
        # predict(
        # model=model,
        # iterator=test_iterator)
        # inference_cyclical(
        #     list_model=model,
        #     test_df=test_df,
        #     input_length=args.input_seq_len,
        #     output_length=args.output_seq_len,
        #     name='123'
        # )
        list_model.append(model)
    # test_mutil_model(list_model, test_iterator)
    # test_error_model(list_model, test_iterator)
    inference_ensemble(
        list_model,
        test_df,
        input_lengths,
        60,
        '123'
        )
    inference_ensemble_final(
        list_model,
        test_df,
        input_lengths,
        60,
        '123'
        )
    # predict(
    #     model=model,
    #     iterator=test_iterator
    # )
    # inference_cyclical(
    #     list_model=model,
    #     test_df=test_df,
    #     input_length=args.input_seq_len,
    #     output_length=args.output_seq_len,
    #     name='123'
    #     )