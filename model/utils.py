from model.init import *


def cal_synthetic_turn_on(threshold_std, seq_length, pm2_5):

    turn_on = []

    for index, _ in enumerate(pm2_5):
        if index < seq_length:
            turn_on.append(1)
        else:
            std = np.std(pm2_5[index - int(seq_length/2) : index + int(seq_length/2)])
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
    new_ls = ['PM2_5']
    ls_col.remove('PM2_5')
    new_ls.extend(ls_col)
    new_df = df[new_ls]
    train_df, valid_df = train_test_split(new_df, test_size=0.4, shuffle=False)
    val_df, test_df = train_test_split(valid_df, test_size=0.5, shuffle=False)
    train_df, sc_train = scale_data(train_df)
    val_df, sc_val = scale_data(val_df)
    test_df, sc_test = scale_data(test_df)

    return train_df, val_df, test_df, sc_train, sc_val, sc_test

def get_test_data(df):
    ls_col = list(df.keys())
    new_ls = ['PM2_5']
    ls_col.remove('PM2_5')
    new_ls.extend(ls_col)
    new_df = df[new_ls]
    test_df, sc_test = scale_data(new_df)
    
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
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('Epoch')
    plt.legend(loc='upper center')
    plt.savefig(folder + filename)  # save the figure to file


def plot_results(y_original, y_predict, folder, filename,y_inference=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # print(y_original.shape)
    # print(y_predict)
    # print(y_inference)
    plt.figure(figsize=(30, 14))
    plt.title('PM2_5')
    plt.plot(y_original, label='Original')
    plt.plot(y_predict, label='Predcit')
    if y_inference is not None:
        plt.plot(y_inference, label='Inference')
    plt.xlabel('Time steps')
    plt.legend(loc='upper center')
    plt.savefig(folder + filename, dpi=200)  # save the figure to file



def cal_missing_value(df):
  count = 0
  for index, row in df.iterrows():
    minute = int(row['datetime'].strftime("%M"))
    if index != 0:
      if (minute - 1 != previous_minute) and (minute + 59 != previous_minute):
        count +=1
        print(previous_minute, minute) 
    previous_minute = minute
  return count

def cal_energy(count, total):
    if count > total:
        count = total
    w_active = 3.185
    w_sleep = 1.617

    return 1 - (count * w_sleep + (total - count) * w_active ) / (total * w_active)