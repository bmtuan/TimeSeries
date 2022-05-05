from init_model import *
import requests
from datetime import datetime, timedelta
import json
from init_database import init_db

input_length = 180
output_length = 30
synthetic_thresold = 0.45
synthetic_seq_len = 4
c_score = 0.8


def process(model, at_least=300):
    now = datetime.now()
    start_time = (now - timedelta(minutes=at_least)).strftime("%d-%m-%Y %H:%M")
    end_time = datetime.now().strftime("%d-%m-%Y %H:%M")
    url = f"http://202.191.57.62:8086/sensor/get_sensors_by_id?start_time={start_time}:00&end_time={end_time}:00&device_id=fimi_14"

    response = requests.get(url)
    content = response.content.decode("UTF-8")
    res = json.loads(content)
    pm2_5 = [instance["PM2_5"] for instance in res["data"]]
    datatime = [
        datetime.strptime(instance["time"], "%Y-%m-%dT%H:%M:%S")
        for instance in res["data"]
    ]
    gt_dict = {"datetime": datatime, "PM2_5": pm2_5}

    df_result = pd.DataFrame(gt_dict)
    df = preprocess(df_result)

    inference_df, sc_test = prepare_inference(
        df=df,
        synthetic_threshold=synthetic_thresold,
        synthetic_sequence_length=synthetic_seq_len,
    )

    pm2_5, is_on = model.inference(inference_df, input_length, c_score, sc_test)

    time = [max(datatime) + timedelta(minutes=i) for i in range(1, len(pm2_5) + 1)]
    result_dict = [
        {"time": time[i], "pm2_5": pm2_5[i], "is_on": is_on[i]}
        for i in range(len(pm2_5))
    ]

    return result_dict


if __name__ == "__main__":
    model = load_model()
    my_collection = init_db()

    while True:

        result_dict = process(model)
        for instance in result_dict:
            s_time = instance["time"].strftime("%Y-%m-%d %H:%M")
            s_time = datetime.strptime(s_time, "%Y-%m-%d %H:%M")
            e_time = s_time + timedelta(minutes=1)
            querry = {"time": {"$gte": s_time, "$lte": e_time}}
            count = my_collection.count_documents(querry)
            if count == 0:
                print("Insert instance: ", instance)
                my_collection.insert_one(instance)
