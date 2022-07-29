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
model_path = '/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/MultiTaskLSTM_0507/multi.pth'
col = ['MAE', 'MSE', 'RMSE', 'MAPE', 'CORR']
index = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7','Task 8']
result = np.zeros((8,5))

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
model.load_state_dict(copyStateDict(torch.load(model_path)))
st = time.time()
mutil_predict(model, test_iterator, 8)
print('time: ', np.round(time.time() - st, 2))
# for i in range(8):
#     (
#     train_df,
#     valid_df,
#     test_df,
#     train_iterator,
#     valid_iterator,
#     test_iterator
# ) = prepare_multitask_test(
#     input_path=dir,
#     input_len=args.input_seq_len,
#     output_len=args.output_seq_len,
#     task=i
# )
#     metrics = mutil_predict(model, test_iterator, 8, i)
#     result[i] = metrics

# result = np.round(result, 4)
# df = pd.DataFrame(result, columns=col, index = index)
# df.to_csv('result/multi_task.csv')
# # print(df)