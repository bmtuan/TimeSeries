from LSTMmodel import *

# from RNNmodel import *
# from GRUmodel import *
from utils import *
from torch.utils.data import DataLoader

# from torch.utils.data import Dataset
import torch

# from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
import pandas as pd


device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")
print(device)

from dataset import *


def prepare(
    input_path, synthetic_threshold, synthetic_sequence_length, input_len, output_len
):
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    print(len(df))
    df = preprocess(df)
    print(len(df))
    ignore_colum = [
        "time",
        "datetime",
        "Unnamed: 0",
        "NO2",
        "SO2",
        # "humidity",
        # "PM10",
        # "temp",
        # "CO",
    ]
    for column in ignore_colum:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
    pm2_5 = df["PM2_5"].values
    turn_on = cal_synthetic_turn_on(
        synthetic_threshold, synthetic_sequence_length, pm2_5
    )
    df["turn_on"] = turn_on

    train_df, valid_df, test_df, sc_train, sc_val, sc_test = get_train_valid_test_data(
        df
    )
    train_dataset = PMDataset(train_df, input_len=input_len, output_len=output_len)
    valid_dataset = PMDataset(valid_df, input_len=input_len, output_len=output_len)
    test_dataset = PMDataset(test_df, input_len=input_len, output_len=output_len)
    # use drop_last to get rid of last batch
    train_iterator = DataLoader(
        train_dataset, batch_size=32, shuffle=False, drop_last=True
    )
    valid_iterator = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, drop_last=True
    )
    test_iterator = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=True
    )

    return (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator,
        sc_train,
        sc_val,
        sc_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--confidence",
        help="Confidence score to predict",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-in",
        "--input_path",
        help="path of file csv",
        type=str,
        default="data/final_envitus.csv",
    )
    parser.add_argument(
        "-ep", "--epochs", help="Number of training epochs", type=int, default=100
    )
    parser.add_argument(
        "-in_seq",
        "--input_seq_len",
        help="input sequence length",
        type=int,
        default=180,
    )
    parser.add_argument(
        "-out_seq",
        "--output_seq_len",
        help="output sequence length",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="learning rate", type=float, default=1e-4
    )
    parser.add_argument(
        "-coef", "--coefficient_loss", help="coefficent loss", type=float, default=1e-1
    )
    parser.add_argument(
        "-path",
        "--model_path",
        help="path of save model",
        type=str,
        default="checkpoint/LSTM",
    )
    parser.add_argument(
        "-syn_thresh",
        "--synthetic_threshold",
        help="synthetic_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-syn_seq",
        "--synthetic_seq_len",
        help="synthetic sequence length",
        type=int,
        default=4,
    )
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
    )

    model = LSTM(
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        confidence=args.confidence,
        number_layer=2,
        input_size=len(train_df.columns),
        hidden_size=256,
    )

    model.to(device)

    # train phase
    model.train(
        model_path=args.model_path,
        train_iterator=train_iterator,
        valid_iterator=valid_iterator,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        coefficient=args.coefficient_loss,
    )

    # test phase
    # model.load_state_dict(copyStateDict(torch.load('/media/aimenext/disk1/tuanbm/TimeSerires/model/checkpoint/LSTM/99_0.011958550283606484.pth')))
    # print('test batch')
    model.predict(iterator=test_iterator, sc_test=sc_test, confidence=args.confidence)

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
