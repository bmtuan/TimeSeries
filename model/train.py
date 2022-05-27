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
    parser.add_argument("-in","--input_path",help="path of file csv",type=str,default="data/final_sensor.csv")
    parser.add_argument("-ep", "--epochs", help="Number of training epochs", type=int, default=100)
    parser.add_argument("-in_seq","--input_seq_len",help="input sequence length",type=int,default=5)
    parser.add_argument("-out_seq","--output_seq_len",help="output sequence length",type=int,default=1)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-4)
    parser.add_argument("-coef", "--coefficient_loss", help="coefficent loss", type=float, default=1e-1)
    parser.add_argument("-path","--model_path",help="path of save model",type=str,default="checkpoint/LSTM")
    parser.add_argument("-syn_thresh","--synthetic_threshold",help="synthetic_threshold",type=float,default=0.3)
    parser.add_argument("-hsize","--hidden_size",help="hidden size",type=int,default=64)
    parser.add_argument("-syn_seq","--synthetic_seq_len",help="synthetic sequence length",type=int,default=5)
    parser.add_argument("-num_lay","--number_layer",help="number of LSTM layer",type=int,default=1)
    parser.add_argument("-mode","--mode",help="normal or turn synthetic",type=str,default='bce')
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
        mode='turn_on'
    )
    
    model = LSTM(
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        confidence=args.confidence,
        number_layer=args.number_layer,
        input_size=len(train_df.columns),
        output_size=len(train_df.columns),
        hidden_size=args.hidden_size,
        mode=args.mode
    )

    model.to(device)
    now = datetime.now()

    str_now = now.strftime("%m_%d_%Y_%H_%M")
    model_path = args.model_path + "_" + str_now
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    dict_infor = vars(args)
    save_train_info(dict_infor, model_path)
    
    # train phase
    model.train(
        model_path=model_path,
        train_iterator=train_iterator,
        valid_iterator=valid_iterator,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        coefficient=args.coefficient_loss,
    )

    # test phase
    # model.load_state_dict(copyStateDict(torch.load('/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/LSTM_05_24_2022_14_29/88_0.0066683387981723635.pth')))
    # print('test batch')
    model.predict(iterator=test_iterator, sc_test=sc_test, confidence=args.confidence, name='test_batch.png')

    # inference
    # import time
    # print('inference...')
    # list_confidence = [0.2, 0.25, 0.3, 0.35 , 0.4, 0.45 , 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    # s = []
    # mape = []

    # for c in list_confidence:
    #     st = time.time()
    #     print('confidence: ', c)
    #     out1, out2 = model.eval_realtime_2(test_df=test_df,
    #                           input_length=args.input_seq_len,
    #                           output_length=args.output_seq_len,
    #                           confidence=c,
    #                           sc_test=sc_test,
    #                           synthetic_threshold=args.synthetic_threshold,
    #                           synthetic_seq_len=args.synthetic_seq_len)
    #     print('take: ', time.time() - st)
    #     s.append(out1)
    #     mape.append(out2)

    # dicts = {'saving': s, 'mape': mape}

    # df_result = pd.DataFrame(dicts)
    # df_result.to_csv('medium_inference2.csv')
