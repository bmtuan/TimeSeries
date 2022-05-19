from init import *
from dataset import *


def cal_synthetic_turn_on(threshold_std, seq_length, pm2_5):

    turn_on = []

    for index, _ in enumerate(pm2_5):
        if index < seq_length:
            turn_on.append(1)
        else:
            std = np.std(
                pm2_5[index - int(seq_length / 2) : index + int(seq_length / 2)]
            )
            # print(std)
            if std > threshold_std:
                turn_on.append(1)
            else:
                turn_on.append(0)

    return turn_on


def scale_data(df):
    sc = MinMaxScaler()
    df_scaled = pd.DataFrame(sc.fit_transform(df), columns=df.columns)
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


def get_test_data(df):
    base_df = pd.read_csv("data/final_envitus.csv")
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
    ]
    for column in ignore_colum:
        if column in base_df.columns:
            base_df.drop(column, axis=1, inplace=True)
    ls_col = list(base_df.keys())
    new_ls = ["PM2_5"]
    ls_col.remove("PM2_5")
    new_ls.extend(ls_col)
    new_df = base_df[new_ls]
    _, sc_test = scale_data(new_df)
    test_sc = sc_test.fit_transform(df)
    test_df = pd.DataFrame(test_sc, columns=df.columns)
    return test_df, sc_test


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
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center")
    plt.savefig(folder + filename)  # save the figure to file


def plot_results(y_original, y_predict, folder, filename, y_inference=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # print(y_original.shape)
    # print(y_predict)
    # print(y_inference)
    plt.figure(figsize=(30, 14))
    plt.title("PM2_5")
    plt.plot(y_original, label="Original")
    plt.plot(y_predict, label="Predcit")
    if y_inference is not None:
        plt.plot(y_inference, label="Inference")
    plt.xlabel("Time steps")
    plt.legend(loc="upper center")
    plt.savefig(folder + filename, dpi=200)  # save the figure to file


def cal_missing_value(df):
    count = 0
    for index, row in df.iterrows():
        minute = int(row["datetime"].strftime("%M"))
        if index != 0:
            if (minute - 1 != previous_minute) and (minute + 59 != previous_minute):
                count += 1
                print(previous_minute, minute)
        previous_minute = minute
    return count


def cal_energy(count, total):
    if count > total:
        count = total
    w_active = 3.185
    w_sleep = 1.617

    return 1 - (count * w_sleep + (total - count) * w_active) / (total * w_active)


def prepare_test(
    input_path, synthetic_threshold, synthetic_sequence_length, input_len, output_len
):

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = preprocess(df)
    # ignore_colum = ["datetime", "Unnamed: 0", "NO2", "SO2", "PM1_0", "CO"]
    ignore_colum = [
        "time",
        "datetime",
        "Unnamed: 0",
        "CO",
        "NO2",
        "SO2",
        "humidity",
        "temp",
        "PM10",
        "PM1_0",
    ]
    for column in ignore_colum:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)

    pm2_5 = df["PM2_5"].values
    turn_on = cal_synthetic_turn_on(
        synthetic_threshold, synthetic_sequence_length, pm2_5
    )
    df["turn_on"] = turn_on
    test_df, sc_test = get_test_data(df)

    test_dataset = PMDataset(test_df, input_len=input_len, output_len=output_len)
    # use drop_last to get rid of last batch
    test_iterator = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=True
    )

    return test_df, test_iterator, sc_test


def boolean(string):
    return string.lower() in ("True", "true", "TRUE")


def preprocess(dataset):
    df = DataFrame()
    previous_row = None

    for _, row in dataset.iterrows():
        # date_time = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
        date_time = row["time"]
        minute = int(date_time.strftime("%M"))
        if len(df.index) == 0:
            df = df.append(row, ignore_index=True)
        if previous_row is not None:
            if previous_minute == minute:
                continue
            if (previous_minute == minute - 1) or (
                previous_minute == 59 and minute == 0
            ):
                df = df.append(row, ignore_index=True)
            else:
                while (minute - 1 != previous_minute) and (
                    minute + 59 != previous_minute
                ):
                    previous_row["time"] = previous_row["time"] + pd.DateOffset(
                        minutes=1
                    )
                    previous_minute = int(previous_row["time"].strftime("%M"))
                    df = df.append(previous_row, ignore_index=True)
                    df = df.append(row, ignore_index=True)

        previous_minute = minute
        previous_row = row

    return df


def prepare_inference(
    df,
    synthetic_threshold,
    synthetic_sequence_length,
):
    ignore_colum = ["datetime", "Unnamed: 0", "NO2", "SO2", "PM1_0", "CO"]
    for column in ignore_colum:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)

    pm2_5 = df["PM2_5"].values
    turn_on = cal_synthetic_turn_on(
        synthetic_threshold, synthetic_sequence_length, pm2_5
    )
    df["turn_on"] = turn_on
    test_df, sc_test = get_test_data(df)

    return test_df, sc_test
