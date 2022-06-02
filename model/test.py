from LSTMmodel import *
from utils import *
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from dataset import *

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--confidence",help="Confidence score to predict",type=float,default=0.5)
    parser.add_argument("-in","--input_path",help="path of file csv",type=str,default="data/train/sensor_3.csv")
    parser.add_argument("-ep", "--epochs", help="Number of training epochs", type=int, default=200)
    parser.add_argument("-in_seq","--input_seq_len",help="input sequence length",type=int,default=10)
    parser.add_argument("-out_seq","--output_seq_len",help="output sequence length",type=int,default=10)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-4)
    parser.add_argument("-coef", "--coefficient_loss", help="coefficent loss", type=float, default=15e-2)
    parser.add_argument("-path","--model_path",help="path of save model",type=str,default="checkpoint/")
    parser.add_argument("-syn_thresh","--synthetic_threshold",help="synthetic_threshold",type=float,default=0.3)
    parser.add_argument("-hsize","--hidden_size",help="hidden size",type=int,default=32)
    parser.add_argument("-syn_seq","--synthetic_seq_len",help="synthetic sequence length",type=int,default=5)
    parser.add_argument("-num_lay","--number_layer",help="number of LSTM layer",type=int,default=2)
    parser.add_argument("-mode","--mode",help="normal or turn synthetic",type=str,default='normal')
    parser.add_argument("-att","--att",help="attention decoder or not",type=bool,default=True)
    args = parser.parse_args()

    (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator,
        sc_train,
        sc_val,
        sc_test,
    ) = prepare(
        input_path=args.input_path,
        synthetic_threshold=args.synthetic_threshold,
        synthetic_sequence_length=args.synthetic_seq_len,
        input_len=args.input_seq_len,
        output_len=args.output_seq_len,
        mode=args.mode
    )
    
    model = LSTM(
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        confidence=args.confidence,
        number_layer=args.number_layer,
        input_size=len(train_df.columns),
        output_size=len(train_df.columns),
        hidden_size=args.hidden_size,
        mode=args.mode,
        att=args.att
    )

    model.to(device)
    
    # test phase
    # model.load_state_dict(copyStateDict(torch.load('/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/_06_01_2022_23_32/97_0.0338.pth')))
    model.load_state_dict(copyStateDict(torch.load('/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/_06_01_2022_23_33/84_0.0851.pth')))
    # print('test batch')

    # model.predict(iterator=test_iterator, sc_test=sc_test)

    # inference
    model.inference_cyclical(test_df=test_df,
                            input_length=args.input_seq_len,
                            output_length=args.output_seq_len,
                            sc_test=sc_test,
                            name='input_path')
    
    model.inference_confidence(test_df=test_df,
                            input_length=args.input_seq_len,
                            output_length=args.output_seq_len,
                            sc_test=sc_test,
                            confidence=0,
                            name='input_path')
