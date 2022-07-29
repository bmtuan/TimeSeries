import argparse
from model_utils import *
from init_database import connect_fimi_MRA, connect_signal_MRA
from datetime import datetime, timedelta
import requests
import time
parser = argparse.ArgumentParser()
parser.add_argument("-out_seq","--output_seq_len",help="output sequence length",type=int,default=10)
parser.add_argument("-path_model","--model_path",help="path of save model",type=str,default="checkpoint/sensor_12")
parser.add_argument("-hsize","--hidden_size",help="hidden size",type=int,default=32)
parser.add_argument("-num_lay","--number_layer",help="number of LSTM layer",type=int,default=1)
args = parser.parse_args()

folders = os.listdir(args.model_path)
input_lengths = []
for folder in folders:
    input_lengths.append(int(folder.split('_')[0]))
fimi_MRA = connect_fimi_MRA()
signal_MRA = connect_signal_MRA()
list_model = load_list_model(
    path=args.model_path,
    input_lengths=input_lengths,
    output_length=args.output_seq_len,
    number_layer=args.number_layer,
    input_size=1,
    output_size=1,
    hidden_size=args.hidden_size,
    device=device
)
mape_threshold = 5
check_mape = 0
now = datetime.now()
max_time = now

while True:
    now = datetime.now()
    if now >= max_time:
        start_time = (now - timedelta(minutes=61)).strftime("%d-%m-%Y %H:%M")
        end_time = now.strftime("%d-%m-%Y %H:%M")
        
        df = fetch_data(start_time, end_time)
        
        predict = []
        original = []

        while check_mape > mape_threshold:
            ensemble_outputs = ensemble_predict(
                list_model, 
                input_lengths, 
                args.output_seq_len, 
                df['PM2_5'].values.reshape((len(df), 1)))
            print(df['PM2_5'].values)
            predict.append(np.round(ensemble_outputs[0][0], 1))
            time_now = datetime.now() + timedelta(minutes=1)
            signal_MRA.insert_one({"time": time_now, "is_on": True})
            print(f'waiting... to {time_now}')
            time.sleep(60)

            end = datetime.now().strftime("%d-%m-%Y %H:%M")
            start = (datetime.now() - timedelta(minutes=61)).strftime("%d-%m-%Y %H:%M")
            df = fetch_data(start, end)
            original.append(df['PM2_5'].values[-1])

            print('original: ', original)
            print('predict: ', predict)
            
            check_mape = mean_absolute_percentage_error(original, predict) * 100
            print('check_mape: ',check_mape)

        y = ensemble_predict(
            list_model=list_model,
            input_lengths=input_lengths,
            output_length=args.output_seq_len,
            input=df['PM2_5'].values.reshape((len(df), 1))
        )

        timeline = [now + timedelta(minutes=i) for i in range(1, len(y) + 6)]
        signal_dict = [{"time": timeline[i], "is_on": False} for i in range(len(y))]
        for i in range(5):
            signal_dict.append({"time": timeline[i+args.output_seq_len], "is_on": True})
        pm25_dict = [{"time": timeline[i], "PM2_5": np.round(y[i][0],2)} for i in range(len(y))]
        check_mape = 100
        # insert to db
        signal_MRA.insert_many(signal_dict)
        fimi_MRA.insert_many(pm25_dict)
        max_time = max(timeline)
        print(f'waiting... to {max_time}')