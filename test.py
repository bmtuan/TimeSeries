from LSTMmodel import *
from RNNmodel import *
from GRUmodel import *
from utils import *
from init import *
from dataset import *

def prepare_test(input_path, synthetic_threshold, synthetic_sequence_length, input_len, output_len):
    df = pd.read_csv(input_path)
    ignore_colum = ['datetime','Unnamed: 0','NO2','SO2','PM1_0','CO']   
    # ignore_colum = ['datetime','Unnamed: 0','CO','NO2','SO2','humidity','temp','PM10', 'PM1_0']
    for column in ignore_colum:
        if column in df.columns:
            df.drop(column, axis=1,inplace=True)
            
    pm2_5 = df['PM2_5'].values
    turn_on = cal_synthetic_turn_on(
        synthetic_threshold, synthetic_sequence_length, pm2_5)
    df['turn_on'] = turn_on
    test_df, sc_test = get_test_data(df)
    
    test_dataset = PMDataset(
        test_df, input_len=input_len, output_len=output_len)
    # use drop_last to get rid of last batch
    test_iterator = DataLoader(test_dataset, batch_size=32,
                               shuffle=False, drop_last=True)

    return test_df, test_iterator, sc_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--confidence', help='Confidence score to predict', type=float, default=0.7)
    parser.add_argument('-ep', '--epochs',
                        help='Number of training epochs', type=int, default=50)
    parser.add_argument('-in_seq', '--input_seq_len',
                        help='input sequence length', type=int, default=180)
    parser.add_argument('-out_seq', '--output_seq_len',
                        help='output sequence length', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate',
                        help='learning rate', type=float, default=0.00001)
    parser.add_argument('-coef', '--coefficient_loss',
                        help='coefficent loss', type=float, default=0.005)
    parser.add_argument('-path', '--model_path',
                        help='path of save model', type=str, default='checkpoint/medianffLSTM.pth')
    parser.add_argument('-syn_thresh', '--synthetic_threshold',
                        help='synthetic_threshold', type=float, default=0.45)
    parser.add_argument('-syn_seq', '--synthetic_seq_len',
                        help='synthetic sequence length', type=int, default=4)
    args = parser.parse_args()

    input_paths = [
        'data/test_envitus_1.csv', 
        'data/test_envitus_2.csv'
        ]
    
    for index,input_path in enumerate(input_paths):
        test_df, test_iterator, sc_test = prepare_test(
            input_path=input_path,
            synthetic_threshold=args.synthetic_threshold,
            synthetic_sequence_length=args.synthetic_seq_len,
            input_len=args.input_seq_len,
            output_len=args.output_seq_len)
            
        model = LSTM(input_seq_len=args.input_seq_len,
                            output_seq_len=args.output_seq_len,
                            confidence=args.confidence,
                            number_layer=2,
                            input_size=5,   
                            hidden_size=64
                            )
        model.to(device)
        # print(model)
        # test phase
        model.load_state_dict(copyStateDict(torch.load(args.model_path)))
        # print('test batch')
        print(args.model_path)
        model.predict(iterator=test_iterator,
                    sc_test=sc_test,
                    confidence=args.confidence)

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
        #                         input_length=args.input_seq_len,
        #                         output_length=args.output_seq_len,
        #                         confidence=c,
        #                         sc_test=sc_test,
        #                         synthetic_threshold=args.synthetic_threshold,
        #                         synthetic_seq_len=args.synthetic_seq_len)
        #     print('take: ', time.time() - st)
        #     s.append(out1)
        #     mape.append(out2)
        
        # dicts = {'saving': s, 'mape': mape} 
        
        # df_result = pd.DataFrame(dicts)
        # df_result.to_csv(f'result/smallLSTM_inference2_data_{index}.csv')
        
        # s = []
        # mape = []
        
        # for c in list_confidence:
        #     st = time.time()
        #     print('confidence: ', c)
        #     out1, out2 = model.eval_realtime(test_df=test_df,
        #                         input_length=args.input_seq_len,
        #                         output_length=args.output_seq_len,
        #                         confidence=c,
        #                         sc_test=sc_test,
        #                         synthetic_threshold=args.synthetic_threshold,
        #                         synthetic_seq_len=args.synthetic_seq_len)
        #     print('take: ', time.time() - st)
        #     s.append(out1)
        #     mape.append(out2)
        
        # dicts = {'saving': s, 'mape': mape} 
        
        # df_result = pd.DataFrame(dicts)
        # df_result.to_csv(f'result/smallLSTM_inference1_data_{index}.csv')
