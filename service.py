import time
from init_model import *
import requests
from datetime import datetime, timedelta
import json
from init_database import connect_fimi_MRA, connect_signal_MRA

input_length = 180
output_length = 30
synthetic_thresold = 0.45
synthetic_seq_len = 4
c_score = 0.5


def process(model, at_least=181):
    now = datetime.now()
    start_time = (now - timedelta(minutes=at_least)).strftime("%d-%m-%Y %H:%M")
    end_time = datetime.now().strftime("%d-%m-%Y %H:%M")
    fimi = f"http://202.191.57.62:8086/sensor/get_sensors_by_id?start_time={start_time}:00&end_time={end_time}:00&device_id=fimi_17"
    fimi_mra = f"http://202.191.57.62:8086/fimi-mra/get_pm25/?start_time={start_time}&end_time={end_time}"

    response = requests.get(fimi_mra)
    content = response.content.decode("UTF-8")
    res = json.loads(content)
    print(res["data"])
    pm2_5 = [instance["PM2_5"] for instance in res["data"]]
    print("aaaaaaaaaaaaaaaa", len(pm2_5))
    if len(pm2_5) < 180:
        response = requests.get(fimi)
        content = response.content.decode("UTF-8")
        res = json.loads(content)
    print("bbbbbbbbbbbbbbbb", len(pm2_5))
    pm2_5 = [instance["PM2_5"] for instance in res["data"]]
    if len(pm2_5) < 180:
        time = [now + timedelta(minutes=i) for i in range(1, 10 + 1)]
        signal_dict = [{"time": time[i], "is_on": True} for i in range(len(time))]
        pm25_dict = [{"time": time[i], "PM2_5": None} for i in range(len(time))]
        return signal_dict, pm25_dict
    datatime = [
        datetime.strptime(instance["time"], "%Y-%m-%dT%H:%M:%S")
        for instance in res["data"]
    ]
    gt_dict = {"datetime": datatime, "PM2_5": pm2_5}

    df = pd.DataFrame(gt_dict)
    df = preprocess(df)

    inference_df, sc_test = prepare_inference(
        df=df,
        synthetic_threshold=synthetic_thresold,
        synthetic_sequence_length=synthetic_seq_len,
    )

    pred_pm2_5, is_on = model.inference(inference_df, input_length, c_score, sc_test)

    time = [now + timedelta(minutes=i) for i in range(1, len(pred_pm2_5) + 1)]
    signal_dict = [{"time": time[i], "is_on": is_on[i]} for i in range(len(pred_pm2_5))]
    pm25_dict = [
        {"time": time[i], "PM2_5": pred_pm2_5[i]} for i in range(len(pred_pm2_5))
    ]
    return signal_dict, pm25_dict


if __name__ == "__main__":
    model = load_model()
    fimi_MRA = connect_fimi_MRA()
    signal_MRA = connect_signal_MRA()
    max_time = datetime.now().strftime("%d-%m-%Y %H:%M")
    while True:
        time.sleep(50)
        if max_time <= datetime.now().strftime("%d-%m-%Y %H:%M"):

            signal_dict, pm25_dict = process(model)

            for index in range(len(signal_dict)):
                s_time = signal_dict[index]["time"].strftime("%d-%m-%Y %H:%M")
                max_time = max(max_time, s_time)

                s_time = datetime.strptime(s_time, "%d-%m-%Y %H:%M")
                e_time = s_time + timedelta(minutes=1)
                querry = {"time": {"$gte": s_time, "$lte": e_time}}
                count_signal = signal_MRA.count_documents(querry)
                count_pm = fimi_MRA.count_documents(querry)
                if count_signal == 0:
                    signal_MRA.insert_one(signal_dict[index])
                if count_pm == 0 and pm25_dict[index]["PM2_5"] != None:
                    fimi_MRA.insert_one(pm25_dict[index])
