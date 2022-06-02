from utils import *
from datetime import datetime
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import wandb
# wandb.init(project="time_series", entity="bm-tuan")
import copy

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

def init_hidden(x: torch.Tensor, hidden_size: int, num_layer: int,xavier: bool = True):
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_layer, x.shape[0], hidden_size)).to(device)
    return Variable(torch.zeros(num_layer, x.shape[0], hidden_size)).to(device)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_seq_len): 
        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        out, state = self.lstm(x)
  
        return out, state

class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, mode):
        super(LSTM_Decoder, self).__init__()
        self.mode = mode
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2_1 = nn.Linear(hidden_size, 16)
        self.fc2_2 = nn.Linear(16, 8)
        self.fc2_3 = nn.Linear(8, 1)
        self.Relu = nn.ReLU()

    def forward(self, x, state):

        output, hidden_state = self.lstm(x, (state))
        out_pm25 = self.fc1(output)

        output_c = self.fc2_1(output)
        output_c = self.Relu(output_c)
        output_c = self.fc2_2(output_c)
        output_c = self.Relu(output_c)
        out_prob = torch.sigmoid(self.fc2_3(output_c))

        return out_pm25, out_prob, hidden_state

class AT_LSTM_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers ,mode):
        super(AT_LSTM_Decoder, self).__init__()
        self.mode = mode
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.fc1 = nn.Linear(hidden_size, output_size - 1)
        self.fc2_1 = nn.Linear(hidden_size, 16)
        self.fc2_2 = nn.Linear(16, 8)
        self.fc2_3 = nn.Linear(8, 1)
        self.Relu = nn.ReLU()

    def attention_net(self, lstm_output, final_state):
        hidden = final_state[-1,:,:]
        if len(hidden.shape) != 2:
            hidden = hidden.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).unsqueeze(1)

        return new_hidden_state
    
    def forward(self, x, prev_hidden):
        output, (hidden_state, cell_state)= self.lstm(x, (prev_hidden))
        attn_output = self.attention_net(output, hidden_state)
        out_feature = self.fc1(attn_output)

        output_c = self.fc2_1(attn_output)
        output_c = self.Relu(output_c)
        output_c = self.fc2_2(output_c)
        output_c = self.Relu(output_c)
        output_c = torch.sigmoid(self.fc2_3(output_c))
        
        return out_feature, output_c, (hidden_state, cell_state)
    
class LSTM(nn.Module):
    def __init__(self,input_seq_len,output_seq_len,confidence,number_layer,input_size,output_size,hidden_size,mode,att=True):
        super(LSTM, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.number_layer = number_layer
        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.confidence = confidence
        self.mode = mode

        self.encoder = LSTM_Encoder(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers= self.number_layer,
            input_seq_len=self.input_seq_len
            )
        if not att:
            self.decoder = LSTM_Decoder(
                input_size=self.input_size, 
                output_size=self.output_size, 
                hidden_size=self.hidden_size, 
                num_layers=self.number_layer, 
                mode=self.mode
                )
        else:
            print('AT_LSTM_Decoder')
            self.decoder = AT_LSTM_Decoder(
                input_size=self.input_size, 
                output_size=self.output_size, 
                hidden_size=self.hidden_size, 
                num_layers=self.number_layer, 
                mode=self.mode
                )

    def forward(self, x):
        outputs_feature = torch.zeros(x.shape[0], self.output_seq_len, self.output_size-1)
        outputs_c = torch.zeros(x.shape[0], self.output_seq_len, 1)
        _, input_encoded = self.encoder(x)
        decoder_input = x[:, -1, :]
        decoder_input = torch.reshape(decoder_input, (x.shape[0], 1, x.shape[2]))
        decoder_hidden = input_encoded
        
        for t in range(self.output_seq_len):
            out_feature, out_c, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            outputs_c[:, t, :] = out_c.squeeze(1)
            outputs_feature[:, t, :] = out_feature.squeeze(1)
            third_tensor = torch.cat((out_feature, out_c), 2)
            decoder_input = third_tensor.to(device)

        return outputs_feature, outputs_c

    def train(self,train_iterator,valid_iterator,learning_rate,num_epochs,coefficient,model_path):
        list_train_loss = []
        list_val_loss = []
        losses_feature = []
        losses_c = []
        
        best_loss = 999999
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        criterion = RMSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        for epoch in tqdm(range(num_epochs)):
            epoch_train_loss = 0
            for x, y1, y2 in train_iterator:
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                feature_loss = 0
                c_loss = 0

                optimizer.zero_grad()
                output_feature, output_c = self.forward(x)
                output_feature, output_c = output_feature.to(device), output_c.to(device)
           
                feature_loss = criterion(output_feature, y1)
                c_loss = criterion(output_c, y2)
                # loss = (1 - coefficient) * feature_loss + coefficient * c_loss
                loss = feature_loss + c_loss
                
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_loss = epoch_train_loss / len(train_iterator)
            epoch_val_loss = 0
            feature_loss_item = 0
            c_loss_item = 0

            with torch.no_grad():
                for x, y1, y2 in valid_iterator:
                    
                    x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                    output_feature, output_c = self.forward(x)
                    
                    output_feature, output_c = output_feature.to(device), output_c.to(device)  
                    feature_loss = criterion(output_feature, y1)
                    c_loss = criterion(output_c, y2)
                    
                    # loss = (1 - coefficient) * feature_loss + coefficient * c_loss
                    loss = feature_loss + c_loss
                    
                    feature_loss_item += feature_loss.item()
                    c_loss_item += c_loss.item()
                    epoch_val_loss += loss.item()

                val_loss = epoch_val_loss / len(valid_iterator)
                val_feature_loss = feature_loss_item / len(valid_iterator)
                val_c_loss = c_loss_item / len(valid_iterator)

                if val_loss < best_loss:
                    name = f"{epoch}_{np.round(val_loss, 4)}.pth"
                    torch.save(self.state_dict(), os.path.join(model_path, name))
                    print(f"\tSave best checkpoint with best loss: {val_loss:.4f}")
                    best_loss = val_loss
            scheduler.step()
            losses_feature.append(val_feature_loss)
            losses_c.append(val_c_loss)
            list_train_loss.append(train_loss)
            list_val_loss.append(val_loss)
            print(f"\t Val loss: {epoch_val_loss / len(valid_iterator):.4f}")
        plot_metrics(losses_feature, losses_c, model_path, "individual.png")
        plot_metrics(list_train_loss, list_val_loss, model_path, "loss.png")

    def predict(self, iterator, sc_test):
        
        feature_original = []
        feature_predict = []

        c_predict = []
        c_original = []
        
        with torch.no_grad():
            for x, y1, y2 in iterator:
                
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                feature_output, c_output = self.forward(x)  # batch_size, output_seq, num_feature

                feature_predict.append(feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1))
                feature_original.append(y1.detach().cpu().numpy()[0, :, 0].reshape(-1))
                c_predict.append(c_output.detach().cpu().numpy()[0, :, 0].reshape(-1))
                c_original.append(y2.detach().cpu().numpy()[0, :, 0].reshape(-1))
                
        feature_predict = np.reshape(feature_predict, (-1))
        feature_original = np.reshape(feature_original, (-1))
        c_predict = np.reshape(c_predict, (-1))
        c_original = np.reshape(c_original, (-1))
        
        y_pred_ = np.expand_dims(feature_predict, 1)
        y_preds = np.repeat(y_pred_, self.input_size, 1)
        y_inv = sc_test.inverse_transform(y_preds)
        y_predict = y_inv[:, 0]

        y_orig_ = np.expand_dims(feature_original, 1)
        y_origs = np.repeat(y_orig_, self.input_size, 1)
        y_inv_ori = sc_test.inverse_transform(y_origs)
        y_original = y_inv_ori[:, 0]

        evaluate_metrics(y_original, y_predict)
        evaluate_metrics(c_original, c_predict)
        
        plot_results(c_original,c_predict,'/media/aimenext/disk1/tuanbm/TimeSerires/model/output','test_batch_confidence.png')
        plot_results(y_original,y_predict,'/media/aimenext/disk1/tuanbm/TimeSerires/model/output','test_batch_feature.png')

    def predict_real_time(self, input_tensor):
        feature_output, c_output = self.forward(input_tensor)  # batch_size, output_seq, num_feature
        c_output = c_output.detach().numpy()
        feature_output = feature_output.detach().numpy()

        c_output = np.reshape(c_output, (c_output.shape[1], c_output.shape[2]))
        y_linear_predict = np.reshape(feature_output, (feature_output.shape[1], feature_output.shape[2]))

        return y_linear_predict, c_output

    def eval_realtime(
        self,
        test_df,
        input_length,
        output_length,
        confidence,
        sc_test,
        synthetic_threshold,
        synthetic_seq_len,
        name,
    ):
        pm2_5 = test_df.iloc[:, 0:1].values
        turn_on_list = test_df.iloc[:, 1:2].values

        pm2_5_copy = copy.deepcopy(pm2_5)
        # inverse scale
        number_feature = len(test_df.columns)
        pm2_5 = np.repeat(pm2_5, number_feature, 1)
        pm2_5 = sc_test.inverse_transform(pm2_5)
        pm2_5 = pm2_5[:, 0].reshape(len(pm2_5), 1)

        y_predict = []
        pm2_5_predict = pm2_5[:input_length]

        input = pm2_5_copy[0:input_length]
        turn_on = turn_on_list[0:input_length]
        turn_on = turn_on.reshape(-1).tolist()
        i = 0
        count = 0

        while i < len(pm2_5):
            # prepare input
            feature = input[len(input) - input_length : len(input)]
            feature_sc = np.repeat(feature, number_feature, 1)
            feature_sc = sc_test.inverse_transform(feature_sc)
            feature_sc = feature_sc[:, 0].reshape(input_length, 1)

            turn_on_input = np.array(turn_on[-input_length:])
            turn_on_input = turn_on_input.reshape(input_length, 1)
            feature = np.concatenate((feature, turn_on_input), axis=1)
            feature = feature.reshape(1, feature.shape[0], feature.shape[1])
            tensor_x = torch.tensor(feature)
            tensor_x = tensor_x.to(device)
            linear_output, prob_output = self.predict_real_time(
                tensor_x.float(), confidence=confidence
            )
            linear_output = linear_output.reshape(linear_output.shape[0], 1)
            y_preds = np.repeat(linear_output, 2, 1)
            y_inv = sc_test.inverse_transform(y_preds)
            y_pred_inv = y_inv[:, 0]
            y_pred_inv = y_pred_inv.reshape(y_pred_inv.shape[0], 1)
            y_pred_inv = np.round(y_pred_inv, 1)
            pm2_5_predict = np.concatenate([pm2_5_predict, y_pred_inv])

            flag = -1

            for index, prob in enumerate(prob_output):
                if prob:
                    flag = index
                    break
            if flag != -1:
                count += flag
                y_predict = y_predict + y_pred_inv[:flag].reshape(-1).tolist()
                input = np.concatenate((input, linear_output[:flag]))
                y_predict = (
                    y_predict + pm2_5[flag + i : flag + i + 10].reshape(-1).tolist()
                )
                input = np.concatenate((input, pm2_5_copy[flag + i : flag + i + 10]))
                i += flag + 10
            else:
                y_predict = y_predict + y_pred_inv.reshape(-1).tolist()
                input = np.concatenate((input, linear_output))
                i += output_length
                count += len(linear_output.tolist())

            turn_on_synthetic = cal_synthetic_turn_on(
                synthetic_threshold,
                synthetic_seq_len,
                y_predict[-(output_length + 10) :],
            )
            turn_on = turn_on + turn_on_synthetic[-(len(input) - len(turn_on)) :]

        # cal loss
        loss_mae = mean_absolute_error(pm2_5, y_predict[: len(pm2_5)])
        loss_rmse = mean_squared_error(pm2_5, y_predict[: len(pm2_5)], squared=False)

        pm2_5 = [i + 1 for i in pm2_5]
        y_predict = [i + 1 for i in y_predict]

        loss_mape = mean_absolute_percentage_error(pm2_5, y_predict[: len(pm2_5)]) * 100
        r2 = r2_score(pm2_5, y_predict[: len(pm2_5)])
        # percent_save = count/len(pm2_5)*100
        percent_save = cal_energy(count, len(pm2_5)) * 100
        if percent_save > 100:
            percent_save = 100
        print(f"{count}/{len(pm2_5)}")
        print(f"Save {percent_save}% times")
        print(f"Loss MAE: {loss_mae:.4f}")
        print(f"Loss RMSE: {loss_rmse:.4f}")
        print(f"Loss MAPE: {loss_mape:.4f}")
        print(f"R2: {r2:.4f}")
        plot_results(
            pm2_5,
            y_predict,
            "output/",
            f"inference_{name}_{confidence}.png",
        )

        return percent_save, loss_mape

    def eval_realtime_2(
        self,
        test_df,
        input_length,
        output_length,
        confidence,
        sc_test,
        synthetic_threshold,
        synthetic_seq_len,
        name,
    ):
        pm2_5 = test_df.iloc[:, 0:1].values
        turn_on_list = test_df.iloc[:, 1:2].values

        pm2_5_copy = copy.deepcopy(pm2_5)
        # inverse scale
        number_feature = len(test_df.columns)
        pm2_5 = np.repeat(pm2_5, number_feature, 1)
        pm2_5 = sc_test.inverse_transform(pm2_5)
        pm2_5 = pm2_5[:, 0].reshape(len(pm2_5), 1)

        y_predict = []

        # pm2_5_predict = pm2_5[:input_length]
        pm2_5 = pm2_5[input_length:]
        input = pm2_5_copy[0:input_length]
        turn_on = turn_on_list[0:input_length]
        turn_on = turn_on.reshape(-1).tolist()
        i = 0
        count = 0

        while i < len(pm2_5):
            # prepare input
            feature = input[len(input) - input_length : len(input)]
            feature_sc = np.repeat(feature, number_feature, 1)
            feature_sc = sc_test.inverse_transform(feature_sc)
            feature_sc = feature_sc[:, 0].reshape(input_length, 1)

            turn_on_input = np.array(turn_on[-input_length:])
            turn_on_input = turn_on_input.reshape(input_length, 1)
            feature = np.concatenate((feature, turn_on_input), axis=1)
            feature = feature.reshape(1, feature.shape[0], feature.shape[1])
            tensor_x = torch.tensor(feature)
            tensor_x = tensor_x.to(device)
            linear_output, prob_output = self.predict_real_time(
                tensor_x.float(), confidence=confidence
            )
            linear_output = linear_output.reshape(linear_output.shape[0], 1)
            y_preds = np.repeat(linear_output, 2, 1)
            y_inv = sc_test.inverse_transform(y_preds)
            y_pred_inv = y_inv[:, 0]
            y_pred_inv = y_pred_inv.reshape(y_pred_inv.shape[0], 1)
            y_pred_inv = np.round(y_pred_inv, 1)
            # print('y_pred_inv', y_pred_inv)
            # pm2_5_predict = np.concatenate([pm2_5_predict, y_pred_inv])

            flag = -1
            for index, prob in enumerate(prob_output):
                if prob:
                    flag = index
                    break
            if flag != -1:
                count += flag
                y_predict = y_predict + y_pred_inv[:flag].reshape(-1).tolist()
                input = np.concatenate((input, linear_output[:flag]))

                # y_predict = (
                #     y_predict + pm2_5[flag + i : flag + i + 10].reshape(-1).tolist()
                # )
                # input = np.concatenate((input, pm2_5_copy[flag + i : flag + i + 10]))

                # i += flag + 10
                for index, _ in enumerate(pm2_5[i:]):
                    # print(index)
                    y_predict.append(float(pm2_5[i + index]))

                    input = np.concatenate(
                        (input, pm2_5_copy[i + index : i + index + 1])
                    )
                    loss_mape_check = (
                        mean_absolute_percentage_error(
                            pm2_5[i : i + index + 1], y_predict[-index - 1 :]
                        )
                        * 100
                    )
                    if loss_mape_check < 2:
                        # print(i)
                        # print(loss_mape_check)
                        i += index + 1
                        break

            else:
                y_predict = y_predict + y_pred_inv.reshape(-1).tolist()
                input = np.concatenate((input, linear_output))
                i += output_length
                count += output_length

            turn_on_synthetic = cal_synthetic_turn_on(
                synthetic_threshold,
                synthetic_seq_len,
                y_predict[-(output_length + 10) :],
            )
            turn_on = turn_on + turn_on_synthetic[-(len(input) - len(turn_on)) :]

        # print("pm2_5:", len(pm2_5))
        # print("y_predict:", len(y_predict))
        # cal loss

        percent_save = cal_energy(count, len(pm2_5)) * 100
        if percent_save > 100:
            percent_save = 100
        loss_mae = mean_absolute_error(pm2_5, y_predict[: len(pm2_5)])
        loss_rmse = mean_squared_error(pm2_5, y_predict[: len(pm2_5)], squared=False)

        pm2_5 = [i + 1 for i in pm2_5]
        y_predict = [i + 1 for i in y_predict]

        loss_mape = mean_absolute_percentage_error(pm2_5, y_predict[: len(pm2_5)]) * 100
        r2 = r2_score(pm2_5, y_predict[: len(pm2_5)])

        percent_save = cal_energy(count, len(pm2_5)) * 100
        if percent_save > 100:
            percent_save = 100
        print(f"{count}/{len(pm2_5)}")
        print(f"Save {percent_save}% times")
        print(f"Loss MAE: {loss_mae:.4f}")
        print(f"Loss RMSE: {loss_rmse:.4f}")
        print(f"Loss MAPE: {loss_mape:.4f}")
        print(f"R2: {r2:.4f}")

        plot_results(
            pm2_5,
            y_predict,
            "output/",
            f"inference_{name}_{confidence}.png",
        )

        return percent_save, loss_mape

    def eval_realtime_3(
        self,
        test_df,
        input_length,
        output_length,
        confidence,
        sc_test,
        synthetic_threshold,
        synthetic_seq_len,
        name,
    ):
        pm2_5 = test_df.iloc[:, 0:1].values
        turn_on_list = test_df.iloc[:, 1:2].values

        pm2_5_copy = copy.deepcopy(pm2_5)
        # inverse scale
        number_feature = len(test_df.columns)
        pm2_5 = np.repeat(pm2_5, number_feature, 1)
        pm2_5 = sc_test.inverse_transform(pm2_5)
        pm2_5 = pm2_5[:, 0].reshape(len(pm2_5), 1)

        y_predict = []
        pm2_5_predict = pm2_5[:input_length]

        input = pm2_5_copy[0:input_length]
        turn_on = turn_on_list[0:input_length]
        turn_on = turn_on.reshape(-1).tolist()
        i = 0
        count = 0

        while i < len(pm2_5):
            # prepare input
            feature = input[len(input) - input_length : len(input)]
            feature_sc = np.repeat(feature, number_feature, 1)
            feature_sc = sc_test.inverse_transform(feature_sc)
            feature_sc = feature_sc[:, 0].reshape(input_length, 1)

            turn_on_input = np.array(turn_on[-input_length:])
            turn_on_input = turn_on_input.reshape(input_length, 1)
            feature = np.concatenate((feature, turn_on_input), axis=1)
            feature = feature.reshape(1, feature.shape[0], feature.shape[1])
            tensor_x = torch.tensor(feature)
            tensor_x = tensor_x.to(device)
            linear_output, prob_output = self.predict_real_time(
                tensor_x.float(), confidence=confidence
            )
            linear_output = linear_output.reshape(linear_output.shape[0], 1)
            y_preds = np.repeat(linear_output, 2, 1)
            y_inv = sc_test.inverse_transform(y_preds)
            y_pred_inv = y_inv[:, 0]
            y_pred_inv = y_pred_inv.reshape(y_pred_inv.shape[0], 1)
            y_pred_inv = np.round(y_pred_inv, 1)
            pm2_5_predict = np.concatenate([pm2_5_predict, y_pred_inv])

            # flag = -1

            # for index, prob in enumerate(prob_output):
            #     if prob:
            #         flag = index
            #         break
            # if flag != -1:
            #     count += flag
            #     y_predict = y_predict + y_pred_inv[:flag].reshape(-1).tolist()
            #     input = np.concatenate((input, linear_output[:flag]))
            #     y_predict = (
            #         y_predict + pm2_5[flag + i : flag + i + 10].reshape(-1).tolist()
            #     )
            #     input = np.concatenate((input, pm2_5_copy[flag + i : flag + i + 10]))
            #     i += flag + 10
            # else:
            y_predict = y_predict + y_pred_inv.reshape(-1).tolist()
            input = np.concatenate((input, linear_output))
            i += output_length
            count += len(linear_output.tolist())
            y_predict = y_predict + pm2_5[i : +i + 10].reshape(-1).tolist()
            input = np.concatenate((input, pm2_5_copy[i : i + 10]))
            i += 10
            turn_on_synthetic = cal_synthetic_turn_on(
                synthetic_threshold,
                synthetic_seq_len,
                y_predict[-(output_length + 10) :],
            )
            turn_on = turn_on + turn_on_synthetic[-(len(input) - len(turn_on)) :]

        # cal loss
        loss_mae = mean_absolute_error(pm2_5, y_predict[: len(pm2_5)])
        loss_rmse = mean_squared_error(pm2_5, y_predict[: len(pm2_5)], squared=False)

        pm2_5 = [i + 1 for i in pm2_5]
        y_predict = [i + 1 for i in y_predict]

        loss_mape = mean_absolute_percentage_error(pm2_5, y_predict[: len(pm2_5)]) * 100
        r2 = r2_score(pm2_5, y_predict[: len(pm2_5)])
        # percent_save = count/len(pm2_5)*100
        percent_save = cal_energy(count, len(pm2_5)) * 100
        if percent_save > 100:
            percent_save = 100
        # print(f"{count}/{len(pm2_5)}")
        # print(f"Save {percent_save}% times")
        # print(f"Loss MAE: {loss_mae:.4f}")
        # print(f"Loss RMSE: {loss_rmse:.4f}")
        # print(f"Loss MAPE: {loss_mape:.4f}")
        # print(f"R2: {r2:.4f}")
        print("pm2_5:", len(pm2_5))
        print("pm2_5_predict:", len(pm2_5_predict.reshape(-1)))
        print("y_predict:", len(y_predict))
        plot_results(
            pm2_5,
            y_predict,
            "output/",
            f"inference_{name}.png",
        )

        return percent_save, loss_mape

    def inference(self, test_df, input_length, confidence, sc_test):
        test_df = test_df[-input_length - 10 :]
        pm2_5 = test_df.iloc[:, 0:1].values
        turn_on_list = test_df.iloc[:, 1:2].values
        pm2_5_copy = copy.deepcopy(pm2_5)
        # inverse scale
        number_feature = len(test_df.columns)
        pm2_5 = np.repeat(pm2_5, number_feature, 1)
        pm2_5 = sc_test.inverse_transform(pm2_5)
        pm2_5 = pm2_5[:, 0].reshape(len(pm2_5), 1)

        y_predict = []

        # pm2_5_predict = pm2_5[:input_length]
        pm2_5 = pm2_5[input_length:]
        input = pm2_5_copy[0:input_length]
        turn_on = turn_on_list[0:input_length]
        turn_on = turn_on.reshape(-1).tolist()

        feature = input[len(input) - input_length : len(input)]
        feature_sc = np.repeat(feature, number_feature, 1)
        feature_sc = sc_test.inverse_transform(feature_sc)
        feature_sc = feature_sc[:, 0].reshape(input_length, 1)

        turn_on_input = np.array(turn_on[-input_length:])
        turn_on_input = turn_on_input.reshape(input_length, 1)
        feature = np.concatenate((feature, turn_on_input), axis=1)
        feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        tensor_x = torch.tensor(feature)
        tensor_x = tensor_x.to(device)
        linear_output, prob_output = self.predict_real_time(
            tensor_x.float(), confidence=confidence
        )
        linear_output = linear_output.reshape(linear_output.shape[0], 1)
        y_preds = np.repeat(linear_output, 2, 1)
        y_inv = sc_test.inverse_transform(y_preds)
        y_pred_inv = y_inv[:, 0]
        y_pred_inv = y_pred_inv.reshape(y_pred_inv.shape[0], 1)
        y_pred_inv = np.round(y_pred_inv, 1)

        flag = -1
        for index, prob in enumerate(prob_output):
            if prob:
                flag = index
                break

        if flag != -1:
            y_predict = y_predict + y_pred_inv[:flag].reshape(-1).tolist()
            is_on = [False for i in range(flag)]
            y_predict.extend(None for i in range(10))
            is_on.extend(True for i in range(10))
            # y_predict = y_predict + y_pred_inv.reshape(-1).tolist()
            # is_on = [False for i in range(len(y_pred_inv))]
        else:
            y_predict = y_predict + y_pred_inv.reshape(-1).tolist()
            is_on = [False for i in range(len(y_pred_inv))]

        return y_predict, is_on

    def inference_cyclical(self, test_df, input_length, output_length, sc_test, name):
        feature = test_df.iloc[:, :].values
        scale_feature = copy.deepcopy(feature)
        feature = sc_test.inverse_transform(feature)

        original_feature = feature[:input_length,0:1]
        feature_predict = feature[:input_length,0:1]
        input = scale_feature[0:input_length]
        i = 0
        count = 0
        while i < len(feature):
            # prepare input
            feature_input = input[len(input) - input_length : len(input)]

            feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
            tensor_x = torch.tensor(feature_input).to(device)
            feature_output, c_output = self.predict_real_time(tensor_x.float())
            
            feature_output_ = np.repeat(feature_output, self.input_size, 1)
            feature_output_ = sc_test.inverse_transform(feature_output_)
            feature_output_ = feature_output_[:, 0:1]
            index = output_length

            feature_predict = np.concatenate([feature_predict, feature_output_[:index]])
            original_feature = np.concatenate((original_feature, feature[i:i+index,0:-1]))
            
            next_input = np.concatenate((feature_output, c_output), 1)
            input = np.concatenate((input, next_input[:index]))
            i += index + 1
            count += index + 1
            input = np.concatenate((input, scale_feature[i : i + 10]))
            i += 10

        # cal loss
        loss_mae = mean_absolute_error(original_feature, feature_predict[:len(original_feature)])
        loss_rmse = mean_squared_error(original_feature, feature_predict[:len(original_feature)], squared=False)

        loss_mape = mean_absolute_percentage_error(original_feature, feature_predict[:len(original_feature)]) * 100
        r2 = r2_score(original_feature, feature_predict[:len(original_feature)])
        percent_save = cal_energy(count, len(feature)) * 100
        if percent_save > 100:
            percent_save = 100
            
        print(f"{count}/{feature.shape[0]}")
        print(f"Save {percent_save}% times")
        print(f"Loss MAE: {loss_mae:.4f}")
        print(f"Loss RMSE: {loss_rmse:.4f}")
        print(f"Loss MAPE: {loss_mape:.4f}")
        print(f"R2: {r2:.4f}")
        # for i in range(0,1):
        plot_results(
            original_feature,
            feature_predict,
            "output/",
            f"inference_{name}.png",
        )

    def inference_confidence(self,test_df,input_length,output_length,confidence ,sc_test,name):
        feature = test_df.iloc[:, :].values
        scale_feature = copy.deepcopy(feature)
        feature = sc_test.inverse_transform(feature)

        original_feature = feature[:input_length,0:1]
        feature_predict = feature[:input_length,0:1]
        input = scale_feature[0:input_length]
        i = 0
        count = 0
        while i < len(feature):
            # prepare input
            feature_input = input[len(input) - input_length : len(input)]

            feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
            tensor_x = torch.tensor(feature_input).to(device)
            feature_output, c_output = self.predict_real_time(tensor_x.float())
            
            feature_output_ = np.repeat(feature_output, self.input_size, 1)
            feature_output_ = sc_test.inverse_transform(feature_output_)
            feature_output_ = feature_output_[:, 0:1]
            index = output_length
            check_turn = False
            for idx, c in enumerate(c_output):
                if c < 0.96:
                    # index = idx
                    check_turn = True
                    break
            # print(index)
            feature_predict = np.concatenate([feature_predict, feature_output_[:index]])
            original_feature = np.concatenate((original_feature, feature[i:i+index,0:-1]))
            
            next_input = np.concatenate((feature_output, c_output), 1)
            input = np.concatenate((input, next_input[:index]))
            i += index + 1
            count += index + 1
            if check_turn:
                input = np.concatenate((input, scale_feature[i : i + 10]))
                i += 10

        # cal loss
        loss_mae = mean_absolute_error(original_feature, feature_predict[:len(original_feature)])
        loss_rmse = mean_squared_error(original_feature, feature_predict[:len(original_feature)], squared=False)

        loss_mape = mean_absolute_percentage_error(original_feature, feature_predict[:len(original_feature)]) * 100
        r2 = r2_score(original_feature, feature_predict[:len(original_feature)])
        percent_save = cal_energy(count, len(feature)) * 100
        if percent_save > 100:
            percent_save = 100
            
        print(f"{count}/{feature.shape[0]}")
        print(f"Save {percent_save}% times")
        print(f"Loss MAE: {loss_mae:.4f}")
        print(f"Loss RMSE: {loss_rmse:.4f}")
        print(f"Loss MAPE: {loss_mape:.4f}")
        print(f"R2: {r2:.4f}")
        # for i in range(0,1):
        plot_results(
            original_feature,
            feature_predict,
            "output/",
            f"inference_{name}.png",
        )
