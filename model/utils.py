from dataset import *
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import os
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pandas import DataFrame
import copy
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,)
device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

            
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-3*iqr
    fence_high = q3+0.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def scale_data(df):
    sc = MinMaxScaler()
    # print(df)
    df_scaled = pd.DataFrame(sc.fit_transform(df), columns=df.columns)
    # print(df_scaled)
    return df_scaled, sc


def get_train_valid_test_data(df):
    ls_col = list(df.keys())
    new_ls = ["PM2_5"]
    ls_col.remove("PM2_5")
    new_ls.extend(ls_col)
    new_df = df[new_ls]
    train_df, valid_df = train_test_split(new_df, test_size=0.4, shuffle=False)
    val_df, test_df = train_test_split(valid_df, test_size=0.5, shuffle=False)
    train_df, sc_train = scale_data(train_df)
    val_df, sc_val = scale_data(val_df)
    test_df, sc_test = scale_data(test_df)

    return train_df, val_df, test_df, sc_train, sc_val, sc_test

def get_train_valid_test_data_2(df):
    train_df, valid_df = train_test_split(df, test_size=0.4, shuffle=False)
    val_df, test_df = train_test_split(valid_df, test_size=0.5, shuffle=False)

    return train_df, val_df, test_df



def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def plot_metrics(train_loss, val_loss, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figsize=(20, 8))
    plt.title("Loss")
    plt.plot(train_loss, label="train/linear")
    plt.plot(val_loss, label="val/binary")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center")
    plt.savefig(os.path.join(folder,filename))  # save the figure to file


def plot_results(y_original, y_predict, folder, filename, y_inference=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figsize=(30, 14))
    plt.title("PM2_5")
    plt.plot(y_original, label="Original")
    plt.plot(y_predict, label="Predcit")
    if y_inference is not None:
        plt.plot(y_inference, label="Inference")
    plt.xlabel("Time steps")
    plt.legend(loc="upper center")
    plt.savefig(os.path.join(folder,filename), dpi=200)  # save the figure to file


def cal_missing_value(df):
    count = 0
    for index, row in df.iterrows():
        minute = int(row["time"].strftime("%M"))
        if index != 0:
            if (minute - 1 != previous_minute) and (minute + 59 != previous_minute):
                count += 1
                # print(previous_minute, minute)
        previous_minute = minute
    return count


def cal_energy(count, total):
    if count > total:
        count = total
    w_active = 3.185
    w_sleep = 1.617
    return 1 - (count * w_sleep + (total - count) * w_active) / (total * w_active)


def boolean(string):
    return string.lower() in ("True", "true", "TRUE")


def preprocess(dataset):
    df = DataFrame()
    previous_row = None
    print('preprocess..')
    for idx, row in dataset.iterrows():
        date_time = row["time"]
        minute = int(date_time.strftime("%M"))
        if len(df.index) == 0:
            df = df.append(row, ignore_index=True)
        if previous_row is not None:
            if previous_minute == minute:
                continue
            if (previous_minute == minute - 1) or (previous_minute == 59 and minute == 0):
                df = df.append(row, ignore_index=True)
            else:
                while (minute - 1 != previous_minute) and (minute + 59 != previous_minute):
                    previous_row["time"] = previous_row["time"] + pd.DateOffset(minutes=1)
                    previous_minute = int(previous_row["time"].strftime("%M"))
                    df = df.append(previous_row, ignore_index=True)
                df = df.append(row, ignore_index=True)

        previous_minute = minute
        previous_row = row

    return df




def save_train_info(dict_data, path_save):
    with open(os.path.join(path_save,"infor.json"),"w") as file:
        json.dump(dict_data,file)


def prepare(input_path, input_len, output_len):
    df = pd.read_csv(input_path)
    df = df[-40000:]
    # print(np.std(df['PM2_5'].values))
    df["time"] = pd.to_datetime(df["time"])
    ignore_colum = [
        "time",
        "datetime",
        "Unnamed: 0",
        "Unnamed: 0.1",
        "NO2",
        "SO2",
        "humidity",
        "PM10",
        "temp",
        "CO",
        "PM1_0",
        "temperature",
        "PM10_0"
    ]
    for column in ignore_colum:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
    # print(df.describe())
    train_df, valid_df, test_df = get_train_valid_test_data_2(df)
    train_dataset = PMDataset2(train_df, input_len=input_len, output_len=output_len)
    valid_dataset = PMDataset2(valid_df, input_len=input_len, output_len=output_len)
    test_dataset = PMDataset2(test_df, input_len=input_len, output_len=output_len)
    # use drop_last to get rid of last batch
    train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=True)
    test_iterator = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    return (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator)
    
    
def prepare_multitask_test(input_path, input_len, output_len, task):
    df_result = pd.DataFrame()
    for index, path in enumerate(os.listdir(input_path)):
        if index == task:
            df = pd.read_csv(os.path.join(input_path, path))
            ignore_colum = [
                "time",
                "datetime",
                "Unnamed: 0",
                "Unnamed: 0.1",
                "NO2",
                "SO2",
                "humidity",
                "PM10",
                "temp",
                "CO",
                "PM1_0",
                "temperature",
                "PM10_0"
            ]
            for column in ignore_colum:
                if column in df.columns:
                    df.drop(column, axis=1, inplace=True)
            for repeat_path in os.listdir(input_path):
                df_result[repeat_path[:-4]] = df['PM2_5'].values[-40000:]
        

    train_df, valid_df, test_df = get_train_valid_test_data_2(df_result)
    train_dataset = PMDataset2(train_df, input_len=input_len, output_len=output_len)
    valid_dataset = PMDataset2(valid_df, input_len=input_len, output_len=output_len)
    test_dataset = PMDataset2(test_df, input_len=input_len, output_len=output_len)
    # use drop_last to get rid of last batch
    train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=True)
    test_iterator = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    return (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator)
    
def prepare_multitask(input_path, input_len, output_len, task=None):
    cluster = [1,2,3,4,5,6,7,8]
    # cluster = [2,5 ,7 ]
    df_result = pd.DataFrame()
    if task != None:
        list_path = [os.listdir(input_path)[task]]
    else:
        list_path = os.listdir(input_path)
    for idx, path in enumerate(list_path):
      if idx + 1 in cluster:
        df = pd.read_csv(os.path.join(input_path, path))
        ignore_colum = [
            "time",
            "datetime",
            "Unnamed: 0",
            "Unnamed: 0.1",
            "NO2",
            "SO2",
            "humidity",
            "PM10",
            "temp",
            "CO",
            "PM1_0",
            "temperature",
            "PM10_0"
        ]
        for column in ignore_colum:
            if column in df.columns:
                df.drop(column, axis=1, inplace=True)
        df_result[path[:-4]] = df['PM2_5'].values[-40000:]
        
    
    train_df, valid_df, test_df = get_train_valid_test_data_2(df_result)
    train_dataset = PMDataset2(train_df, input_len=input_len, output_len=output_len)
    valid_dataset = PMDataset2(valid_df, input_len=input_len, output_len=output_len)
    test_dataset = PMDataset2(test_df, input_len=input_len, output_len=output_len)
    # use drop_last to get rid of last batch
    train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=True)
    test_iterator = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    return (
        train_df,
        valid_df,
        test_df,
        train_iterator,
        valid_iterator,
        test_iterator)
    
    
def prepare2(input_path, input_len, output_len):
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    # df = remove_outlier(df, 'PM2_5')

    df = df[-20000:]
    ignore_colum = [
        "time",
        "datetime",
        "Unnamed: 0",
        "NO2",
        "SO2",
        "humidity",
        "PM10",
        "temp",
        "CO",
        "PM1_0",
        "temperature",
        "PM10_0"
    ]
    for column in ignore_colum:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)

    train_df, valid_df, test_df, sc_train, sc_val, sc_test = get_train_valid_test_data(df)
    train_dataset = PMDataset2(train_df, input_len=input_len, output_len=output_len)
    valid_dataset = PMDataset2(valid_df, input_len=input_len, output_len=output_len)
    test_dataset = PMDataset2(test_df, input_len=input_len, output_len=output_len)
    # use drop_last to get rid of last batch
    train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=True)
    test_iterator = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

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

def evaluate_metrics(y_original, y_predict):
    loss_mae = mean_absolute_error(y_original, y_predict)
    loss_rmse = mean_squared_error(y_original, y_predict, squared=False)
    loss_mse = mean_squared_error(y_original, y_predict, squared=True)
    loss_mape = mean_absolute_percentage_error(y_original, y_predict) * 100
    corr = pearsonr(y_original, y_predict)
    print(f"Loss MAE: {round(loss_mae,2)}")
    print(f"Loss MSE: {round(loss_mse,2)}")
    print(f"Loss RMSE: {round(loss_rmse,2)}")
    print(f"Loss MAPE: {round(loss_mape,2)}")
    print(f"CORR: {round(corr[0]*100,2)}")
    
    return loss_mae, loss_rmse, loss_mse, loss_mape, corr[0]*100
    

def fetch_data(start, end, mra=True):
    if mra:
        fimi = f"http://202.191.57.62:8086/fimi-mra/get_pm25?start_time={start}&end_time={end}"
    else:
        fimi = f"http://202.191.57.62:8086/sensor/get_sensors_by_id?start_time={start}:00&end_time={end}:00&device_id=fimi_23"

    response = requests.get(fimi)
    content = response.content.decode("UTF-8")
    res = json.loads(content)
    pm2_5 = [instance["PM2_5"] for instance in res["data"]]

    time = [
        datetime.strptime(instance["time"].split('.')[0], "%Y-%m-%dT%H:%M:%S")
        for instance in res["data"]
    ]
    gt_dict = {"time": time, "PM2_5": pm2_5}
    df = pd.DataFrame(gt_dict)
    df = preprocess(df)
    
    return df

def evaluate_realtime(start, end):
    predict_df = fetch_data(start, end)
    original_df = fetch_data(start, end, False)
    
    predict_pm25 = predict_df['PM2_5'].values
    original_pm25 = original_df['PM2_5'].values

    mape = mean_absolute_percentage_error(original_pm25[1:], predict_pm25) * 100
    print('MAPE: ', mape)
    
if __name__ == '__main__':
    # df = pd.read_csv('/media/aimenext/disk1/tuanbm/TimeSerires/model/data/1306/sensor_12.csv')
    # df["time"] = pd.to_datetime(df["time"])
    # print(len(df))
    # df = remove_outlier(df, 'PM2_5')
    # df = preprocess(df)
    # print(len(df))
    # df.to_csv('/media/aimenext/disk1/tuanbm/TimeSerires/model/data/train/sensor_12.csv')
    # start = '20-06-2022 11:25'
    # end = '20-06-2022 12:35'
    # evaluate_realtime(start, end)
    print(cal_energy(10, 15))