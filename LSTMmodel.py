from init import *
from utils import *


class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        out, state = self.lstm(x)
        return out, state


class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0):
        super(LSTM_Decoder, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2_1 = nn.Linear(hidden_size, 256)
        self.fc2_2 = nn.Linear(256, 64)
        self.fc2_3 = nn.Linear(64, 10)
        self.fc2_4 = nn.Linear(10, output_size)

    def forward(self, x, state):

        # state: (num_layers, batch_size, hidden_size)
        output, hidden_state = self.lstm(x, (state))
        out_pm = self.fc1(output)

        output_2 = self.fc2_1(output)
        output_2 = self.fc2_2(output_2)
        output_2 = self.fc2_3(output_2)
        out_prob = torch.sigmoid(self.fc2_4(output_2))

        return out_pm, out_prob, hidden_state


class LSTM(nn.Module):
    def __init__(self,  input_seq_len, output_seq_len, confidence, number_layer, input_size=2, output_size=1, hidden_size=12):
        super(LSTM, self).__init__()
        self.output_seq_len = output_seq_len
        self.input_size = input_size
        self.number_layer = number_layer
        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.confidence = confidence
        self.encoder = LSTM_Encoder(self.input_size, self.hidden_size, self.number_layer)
        self.decoder = LSTM_Decoder(
            2, self.output_size, self.hidden_size, self.number_layer)

    def forward(self, x):
        outputs = torch.zeros(x.shape[0], self.output_seq_len, 1)
        outputs_prob = torch.zeros(x.shape[0], self.output_seq_len, 1)
        encoder_output, encoder_hidden = self.encoder(x)

        decoder_input = torch.zeros(x.shape[0], 1, 2).cuda()
        # decoder_input = x[:, -1, :]
        # decoder_input = torch.reshape(
        #     decoder_input, (x.shape[0], 1, x.shape[2]))
        decoder_hidden = encoder_hidden
        for t in range(self.output_seq_len):
            out_linear, out_prob, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)
            in_prob = out_prob < 1 - self.confidence
            in_prob = in_prob.long()
            in_prob = 1 - in_prob
            outputs_prob[:, t, :] = out_prob.squeeze(1)
            outputs[:, t, :] = out_linear.squeeze(1)

            # keep row height and append in columns
            third_tensor = torch.cat((out_linear, in_prob), 2)
            decoder_input = third_tensor.to(device)

        return outputs, outputs_prob

    def train(self, train_iterator, valid_iterator, learning_rate, num_epochs, coefficient, model_path):
        list_train_loss = []
        list_val_loss = []
        loss1 = []
        loss2 = []
        best_loss = 999999
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        criterion_binary = nn.BCELoss()
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.9)

        for epoch in tqdm(range(num_epochs)):
            epoch_train_loss = 0
            for x, y1, y2 in train_iterator:
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

                linear_loss = 0
                binary_loss = 0

                optimizer.zero_grad()
                outputs, outputs_prob = self.forward(x)
                outputs, outputs_prob = outputs.to(
                    device), outputs_prob.to(device)

                linear_loss = criterion(outputs, y1)
                binary_loss = criterion_binary(outputs_prob, y2)
                  
                # print('linear_loss', linear_loss.item())
                # print('binary_loss', binary_loss.item())
                loss = (1 - coefficient) * linear_loss + \
                    coefficient * binary_loss
                # loss = linear_loss + binary_loss * coef
                # loss = (2 * linear_loss * binary_loss * coefficient) / (linear_loss + binary_loss)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_loss = epoch_train_loss / len(train_iterator)
            epoch_val_loss = 0
            linear_loss_item = 0
            binary_loss_item = 0
            
            with torch.no_grad():
                for x, y1, y2 in valid_iterator:
                    x, y1, y2 = x.to(device), y1.to(device), y2.to(device)


                    outputs, outputs_prob = self.forward(x)
                    outputs, outputs_prob = outputs.to(
                        device), outputs_prob.to(device)

                    linear_loss = criterion(outputs, y1)
                    binary_loss = criterion_binary(outputs_prob, y2)
                    loss = (1 - coefficient) * linear_loss + \
                        coefficient * binary_loss
                    linear_loss_item += linear_loss.item()
                    binary_loss_item += binary_loss.item()
                    # loss = (2 * linear_loss * binary_loss) / (linear_loss + binary_loss)
                    epoch_val_loss += loss.item()

                val_loss = epoch_val_loss / len(valid_iterator)
                val_linear_loss = linear_loss_item / len(valid_iterator)
                val_binary_loss = binary_loss_item / len(valid_iterator)

                if val_loss < best_loss:
                    torch.save(self.state_dict(), model_path)
                    print(
                        f'\tSave best checkpoint with best loss: {val_loss:.4f}')
                    best_loss = val_loss
            scheduler.step()
            loss1.append(val_linear_loss)
            loss2.append(val_binary_loss)
            list_train_loss.append(train_loss)
            list_val_loss.append(val_loss)
            print(f'\t Val loss: {epoch_val_loss / len(valid_iterator):.4f}')
        plot_metrics(loss1, loss2, 'output/', 'metric1.png')
        plot_metrics(list_train_loss, list_val_loss, 'output/', 'metric2.png')

    def predict(self, iterator, sc_test, confidence):
        y_linear_original = []
        y_linear_predict = []
        y_classcification_original = []
        y_classcification_predict = []
        with torch.no_grad():
            for x, y1, y2 in iterator:
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                
                linear_outputs, prob_output = self.forward(
                    x)  # batch_size, output_seq, num_feature
                linear_outputs, prob_output = linear_outputs.to(
                    device), prob_output.to(device)

                y_linear_predict.append(linear_outputs.detach().cpu().numpy()[
                                        :, 0, :].reshape(-1))
                y_linear_original.append(
                    y1.detach().cpu().numpy()[:, 0, :].reshape(-1))

                prob_output = prob_output > (1 - confidence)
                prob_output = prob_output.long()

                y_classcification_predict.append(
                    prob_output.detach().cpu().numpy()[:, 0, :].reshape(-1))
                y_classcification_original.append(
                    y2.detach().cpu().numpy()[:, 0, :].reshape(-1))

        y_linear_predict = np.reshape(y_linear_predict, (-1))
        y_linear_original = np.reshape(y_linear_original, (-1))
        y_classcification_predict = np.reshape(y_classcification_predict, (-1))
        y_classcification_original = np.reshape(
            y_classcification_original, (-1))

        accuracy = accuracy_score(
            y_classcification_predict, y_classcification_original)
        precision = precision_score(
            y_classcification_predict, y_classcification_original)
        recall = recall_score(y_classcification_predict,
                              y_classcification_original)
        f1 = f1_score(y_classcification_predict, y_classcification_original)

        print(f'Accuracy : {accuracy:.4f}')
        print(f'Precision : {precision:.4f}')
        print(f'Recall : {recall:.4f}')
        print(f'F1 : {f1:.4f}')

        y_pred_ = np.expand_dims(y_linear_predict, 1)
        y_preds = np.repeat(y_pred_, self.input_size, 1)

        y_inv = sc_test.inverse_transform(y_preds)
        y_pred_true = y_inv[:, 0]

        y_orig_ = np.expand_dims(y_linear_original, 1)
        y_origs = np.repeat(y_orig_, self.input_size, 1)

        y_inv_ori = sc_test.inverse_transform(y_origs)
        y_orig_true = y_inv_ori[:, 0]

        y_predict = y_pred_true
        y_original = y_orig_true

        loss_mae = mean_absolute_error(y_original, y_predict)
        loss_rmse = mean_squared_error(y_original, y_predict, squared=False)
        loss_mape = mean_absolute_percentage_error(y_original +1, y_predict +1) * 100
        r2 = r2_score(y_original, y_predict)

        print(f"Loss MAE: {loss_mae}")
        print(f"Loss RMSE: {loss_rmse}")
        print(f"Loss MAPE: {loss_mape}")
        print(f"R2: {r2}")
        
        # plot_results(y_original, y_predict, 'output/', 'test.png')
        

    def predict_real_time(self, input_tensor, confidence):
        linear_outputs, prob_output = self.forward(
            input_tensor)  # batch_size, output_seq, num_feature
        prob_output = prob_output.detach().numpy()
        # print(prob_output)
        prob_output = 1 - prob_output
        # print('prob_output: ', prob_output)
        prob_output = prob_output > (1 - confidence)
        prob_output = np.reshape(prob_output, (-1))

        linear_outputs = linear_outputs.detach().numpy()
        y_linear_predict = np.reshape(linear_outputs, (-1))

        return y_linear_predict, prob_output

    def eval_realtime(self, test_df, input_length, output_length, confidence, sc_test, synthetic_threshold, synthetic_seq_len):
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

        input = pm2_5_copy[0: input_length]
        turn_on = turn_on_list[0: input_length]
        turn_on = turn_on.reshape(-1).tolist()
        i = 0
        count = 0

        while i < len(pm2_5):
            # prepare input
            feature = input[len(input) - input_length: len(input)]
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
                tensor_x.float(), confidence=confidence)
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
                y_predict = y_predict + \
                    y_pred_inv[:flag].reshape(-1).tolist()
                input = np.concatenate((input, linear_output[:flag]))
                y_predict = y_predict + \
                    pm2_5[flag + i: flag + i +
                          30].reshape(-1).tolist()
                input = np.concatenate(
                    (input, pm2_5_copy[flag + i: flag + i + 30]))
                i += flag + 30
            else:
                y_predict = y_predict + y_pred_inv.reshape(-1).tolist()
                input = np.concatenate((input, linear_output))
                i += output_length
                count += len(linear_output.tolist())

            turn_on_synthetic = cal_synthetic_turn_on(
                synthetic_threshold, synthetic_seq_len, y_predict[-(output_length + 10):])
            turn_on = turn_on + \
                turn_on_synthetic[-(len(input) - len(turn_on)):]

        # cal loss
        loss_mae = mean_absolute_error(pm2_5, y_predict[:len(pm2_5)])
        loss_rmse = mean_squared_error(
            pm2_5, y_predict[:len(pm2_5)], squared=False)
        
        pm2_5 = [i+1 for i in pm2_5]
        y_predict = [i+1 for i in y_predict]
        
        loss_mape = mean_absolute_percentage_error(
            pm2_5, y_predict[:len(pm2_5)] )*100
        r2 = r2_score(pm2_5, y_predict[:len(pm2_5)])
        # percent_save = count/len(pm2_5)*100
        percent_save = cal_energy(count, len(pm2_5))*100
        if percent_save > 100:
            percent_save = 100
        print(f'{count}/{len(pm2_5)}')
        print(f"Save {percent_save}% times")
        print(f"Loss MAE: {loss_mae:.4f}")
        print(f"Loss RMSE: {loss_rmse:.4f}")
        print(f"Loss MAPE: {loss_mape:.4f}")
        print(f"R2: {r2:.4f}")
        # plot_results(pm2_5, pm2_5_predict.reshape(-1), 'output/',
        #              'inference.png', y_predict)
        
        return percent_save, loss_mape

    def eval_realtime_2(self, test_df, input_length, output_length, confidence, sc_test, synthetic_threshold, synthetic_seq_len):
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
        input = pm2_5_copy[0: input_length]
        turn_on = turn_on_list[0: input_length]
        turn_on = turn_on.reshape(-1).tolist()
        i = 0
        count = 0

        while i < len(pm2_5):
            # prepare input
            feature = input[len(input) - input_length: len(input)]
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
                tensor_x.float(), confidence=confidence)
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

                y_predict = y_predict + \
                    pm2_5[flag + i: flag + i + 20].reshape(-1).tolist()
                input = np.concatenate(
                    (input, pm2_5_copy[flag + i: flag + i + 20]))

                i += flag + 20
                for index, _ in enumerate(pm2_5[i:]):
                    # print(index)
                    y_predict.append(float(pm2_5[i + index]))

                    input = np.concatenate(
                        (input, pm2_5_copy[i + index: i + index + 1]))
                    loss_mape_check = mean_absolute_percentage_error(
                        pm2_5[:len(y_predict)], y_predict)*100
                    if loss_mape_check < 20:
                        break
                i += index + 1
            else:
                y_predict = y_predict + y_pred_inv.reshape(-1).tolist()
                input = np.concatenate((input, linear_output))
                i += output_length
                count += output_length

            turn_on_synthetic = cal_synthetic_turn_on(
                synthetic_threshold, synthetic_seq_len, y_predict[-(output_length + 10):])
            turn_on = turn_on + \
                turn_on_synthetic[-(len(input) - len(turn_on)):]

        print('pm2_5:', len(pm2_5))
        print('y_predict:', len(y_predict))
        # cal loss
        
        percent_save = cal_energy(count, len(pm2_5))*100
        if percent_save > 100:
            percent_save = 100
        loss_mae = mean_absolute_error(pm2_5, y_predict[:len(pm2_5)])
        loss_rmse = mean_squared_error(
            pm2_5, y_predict[:len(pm2_5)], squared=False)
        
        pm2_5 = [i+1 for i in pm2_5]
        y_predict = [i+1 for i in y_predict]
        
        loss_mape = mean_absolute_percentage_error(
            pm2_5 , y_predict[:len(pm2_5)] )*100
        r2 = r2_score(pm2_5, y_predict[:len(pm2_5)])
        print(f'{count}/{len(pm2_5)}')
        print(f"Save {percent_save}% times")
        print(f"Loss MAE: {loss_mae:.4f}")
        print(f"Loss RMSE: {loss_rmse:.4f}")
        print(f"Loss MAPE: {loss_mape:.4f}")
        print(f"R2: {r2:.4f}")
        # plot_results(pm2_5, y_predict, 'output/',
        #              f'inference_{confidence}.png')

        return percent_save, loss_mape
