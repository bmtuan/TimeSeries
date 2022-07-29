# from LSTMAttention import AttentionLSTM
# import torch
from model_utils import *
from utils import *
import argparse

parser = argparse.ArgumentParser()
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

for path in os.listdir(dir):
  if 'sensor_12.csv' in path:    
    print(path)
    list_model = []
    folders = os.listdir(os.path.join(args.model_path, path[:-4]))
    input_lengths = []
    for folder in folders:
        input_lengths.append(int(folder.split('_')[0]))
        
    train_df, valid_df, test_df, train_iterator, valid_iterator, test_iterator = prepare(
        input_path=os.path.join(dir,path),
        input_len=args.input_seq_len,
        output_len=args.output_seq_len,
    )
    
    list_model = load_list_model(
        path=os.path.join(args.model_path, path[:-4]),
        input_lengths=input_lengths,
        output_length=args.output_seq_len,
        number_layer=args.number_layer,
        input_size=len(train_df.columns),
        output_size=len(train_df.columns),
        hidden_size=args.hidden_size,
        device=device
    )
    print('test batch')
    test_mean_ensemble_model(
        list_model=list_model,
        iterator=test_iterator
    )
    print('inference_cyclical_ensemble')
    inference_cyclical_ensemble(
        list_model=list_model,
        test_df=test_df,
        input_lengths=input_lengths,
        output_length=args.output_seq_len,
        )
    print('inference_ensemble_final')
    inference_ensemble_final(
        list_model=list_model,
        test_df=test_df,
        input_lengths=input_lengths,
        output_length=args.output_seq_len,
        mape_threshold=5
        )
