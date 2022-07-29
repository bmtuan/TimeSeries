# from Multitask_LSTM import MultiTaskLSTM
from LSTMAttention import MultiTaskLSTM
from model_utils import *
from utils import *
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument("-in_seq","--input_seq_len",help="input sequence length",type=int,default=10)
parser.add_argument("-out_seq","--output_seq_len",help="output sequence length",type=int,default=10)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=1e-4)
parser.add_argument("-hsize","--hidden_size",help="hidden size",type=int,default=32)
parser.add_argument("-num_lay","--number_layer",help="number of LSTM layer",type=int,default=1)
args = parser.parse_args()
dir = 'data/multitask_train'
col = ['MAE', 'MSE', 'RMSE', 'MAPE', 'CORR']
index = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7','Task 8']
result = np.zeros((8,2))
total_time = 0
for i,path in enumerate(os.listdir(dir)):
    model_path = f'checkpoint/Single_Task_{i+1}/best4.pth'
    (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator
    ) = prepare(
        input_path=os.path.join(dir, path),
        input_len=args.input_seq_len,
        output_len=args.output_seq_len,
    )
    model = MultiTaskLSTM(
                input_seq_len=args.input_seq_len,
                # input_seq_len=length,
                output_seq_len=args.output_seq_len,
                number_layer=args.number_layer,
                input_size=len(train_df.columns),
                output_size=len(train_df.columns),
                hidden_size=args.hidden_size,
                device=device,
            )

    model.to(device)
    model.load_state_dict(copyStateDict(torch.load(model_path)))
    st = time.time()
    # metrics = predict(model, test_iterator)
    metrics = inference_3(
        model=model,
        test_df=test_df,
        input_length=args.input_seq_len,
        output_length=args.output_seq_len,
        off_size=0,
        mape_threshold=3
    )
    # print('time: ', round(time.time() - st,2))
    total_time += time.time() - st
    result[i] = metrics
    # print(metrics)
# print("total time: ", np.round(total_time, 2))
# print(model.)
result = np.round(result, 2)
df = pd.DataFrame(result, index = index)
df.to_csv('result/7.csv')
# print(df)