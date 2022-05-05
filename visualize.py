from init_model import *
import requests
from datetime import datetime, timedelta
import json
import time

input_length = 180
output_length = 30
synthetic_thresold = 0.45
synthetic_seq_len = 4
c_score = 0.5


def plot_train_test_results(base_y, y_predict, y_original):
    plt.figure(figsize=(25, 7))
    plt.plot(
        np.arange(len(base_y)),
        base_y,
        color=(0.2, 0.42, 0.72),
        linewidth=2,
        label="Base",
    )
    plt.plot(
        np.arange(len(base_y) - 1, len(base_y) + len(y_predict) - 1),
        y_predict,
        color=(0.4, 0.4, 0.01),
        linewidth=2,
        label="Prediction",
    )
    plt.plot(
        np.arange(len(base_y) - 1, len(base_y) + len(y_predict) - 1),
        y_original,
        color=(0.76, 0.01, 0.01),
        linewidth=2,
        label="Original",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.legend(loc="upper left")
    name = datetime.now().strftime("%d-%m-%Y %H:%M")
    name = name.replace(" ", "_")
    name = name.replace(":", "_")
    plt.savefig(f"visualize/{name}.png")

    print("save figure...")


def process(model, at_least=300):
    now = datetime.now()
    now = now - timedelta(minutes=60)
    start_time_str = (now - timedelta(minutes=at_least)).strftime("%d-%m-%Y %H:%M")
    end_time_str = now.strftime("%d-%m-%Y %H:%M")
    url = f"http://202.191.57.62:8086/sensor/get_sensors_by_id?start_time={start_time_str}:00&end_time={end_time_str}:00&device_id=fimi_13"

    response = requests.get(url)
    content = response.content.decode("UTF-8")
    res = json.loads(content)
    base_pm2_5 = [instance["PM2_5"] for instance in res["data"]]
    datatime = [
        datetime.strptime(instance["time"], "%Y-%m-%dT%H:%M:%S")
        for instance in res["data"]
    ]
    gt_dict = {"datetime": datatime, "PM2_5": base_pm2_5}

    base_df = pd.DataFrame(gt_dict)
    # base_df = preprocess(df_result)

    inference_df, sc_test = prepare_inference(
        df=base_df,
        synthetic_threshold=synthetic_thresold,
        synthetic_sequence_length=synthetic_seq_len,
    )

    pred_pm2_5, is_on = model.inference(inference_df, input_length, c_score, sc_test)

    st = now.strftime("%d-%m-%Y %H:%M")
    et = (now + timedelta(minutes=output_length)).strftime("%d-%m-%Y %H:%M")

    print("st ", st)
    print("et ", et)
    url = f"http://202.191.57.62:8086/sensor/get_sensors_by_id?start_time={st}:00&end_time={et}:00&device_id=fimi_13"

    response = requests.get(url)
    content = response.content.decode("UTF-8")
    res = json.loads(content)

    gt_pm2_5 = [instance["PM2_5"] for instance in res["data"]]
    datatime = [
        datetime.strptime(instance["time"], "%Y-%m-%dT%H:%M:%S")
        for instance in res["data"]
    ]
    gt_dict = {"datetime": datatime, "PM2_5": gt_pm2_5}

    df_result = pd.DataFrame(gt_dict)
    # gt_df = preprocess(df_result)
    gt_df = df_result
    print("base_pm2_5 ", len(base_df["PM2_5"][-input_length:]))
    print("pred_pm2_5 ", len(pred_pm2_5))
    print("gt_pm2_5 ", len(gt_df["PM2_5"][:output_length]))
    if len(pred_pm2_5) > 1:
        plot_train_test_results(
            base_df["PM2_5"][-input_length:], pred_pm2_5, gt_df["PM2_5"][:output_length]
        )
        return True
    else:
        print("Nothing visualize")
        return False


if __name__ == "__main__":
    model = load_model()
    check = False
    while True:
        check = process(model)
        time.sleep(30)
