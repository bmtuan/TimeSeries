from init import *
from const import *


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        out, state = self.lstm(x)
        return out, state


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, dropout=0):
        super(Decoder, self).__init__()
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


class lstm_seq2seq(nn.Module):
    def __init__(self, input_size=2, output_size=1, input_seq_len=120, output_seq_len=10, hidden_size=12, confidence=THRESHOLD_PREDICT):
        super(lstm_seq2seq, self).__init__()
        self.output_seq_len = output_seq_len
        self.input_size = input_size
        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.confidence = confidence
        self.encoder = Encoder(self.input_size, self.hidden_size)
        self.decoder = Decoder(
            self.input_size, self.output_size, self.hidden_size)

    def forward(self, x):
        outputs = torch.zeros(x.shape[0], self.output_seq_len, 1)
        outputs_prob = torch.zeros(x.shape[0], self.output_seq_len, 1)
        encoder_output, encoder_hidden = self.encoder(x)

        # decoder_input_ = torch.zeros(x.shape[0], 1, self.input_size)
        decoder_input = x[:, -1, :]
        decoder_input = torch.reshape(
            decoder_input, (x.shape[0], 1, x.shape[2]))
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
            decoder_input = third_tensor

        return outputs, outputs_prob

    def train(self, train_iterator, valid_iterator, learning_rate=0.001, num_epochs=50, coefficient=COEFFICIENT_LOSS, thresh=THRESHOLD_PREDICT):
        list_train_loss = []
        list_val_loss = []
        best_loss = 999999
        losses = np.full(num_epochs, np.nan)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        criterion_binary = nn.BCELoss()
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=4, gamma=0.8)

        for epoch in tqdm(range(num_epochs)):
            epoch_train_loss = 0
            list_val_acc = []
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
                loss = (1 - coefficient) * linear_loss + \
                    coefficient * binary_loss

                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_loss = epoch_train_loss / len(train_iterator)
            epoch_val_loss = 0

            with torch.no_grad():
                for x, y1, y2 in valid_iterator:
                    x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

                    linear_loss = 0
                    binary_loss = 0
                    outputs, outputs_prob = self.forward(x)
                    outputs, outputs_prob = outputs.to(
                        device), outputs_prob.to(device)

                    linear_loss = criterion(outputs, y1)
                    binary_loss = criterion_binary(outputs_prob, y2)
                    loss = (1 - coefficient) * linear_loss + \
                        coefficient * binary_loss
                    epoch_val_loss += loss.item()

                val_loss = epoch_val_loss / len(valid_iterator)
                if val_loss < best_loss:
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(
                        f'\tSave best checkpoint with best loss: {val_loss:.4f}')
                    best_loss = val_loss
            scheduler.step()
            list_train_loss.append(train_loss)
            list_val_loss.append(val_loss)
            print(f'\t Val loss: {epoch_val_loss / len(valid_iterator):.4f}')

        return list_train_loss, list_val_loss

    def predict(self, iterator, threshold=THRESHOLD_PREDICT):
        y_linear_original = []
        y_linear_predict = []
        y_classcification_original = []
        y_classcification_predict = []

        with torch.no_grad():
            for x, y1, y2 in iterator:
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

                linear_outputs, prob_output = self.forward(
                    x)  # batch_size, output_seq, num_feature
                linear_outputs, outputs_prob = linear_outputs.to(
                    device), outputs_prob.to(device)

                y_linear_predict.append(linear_outputs.detach().numpy()[
                                        :, 0, :].reshape(-1))
                y_linear_original.append(
                    y1.detach().numpy()[:, 0, :].reshape(-1))

                prob_output = prob_output > threshold
                prob_output = prob_output.long()

                y_classcification_predict.append(
                    prob_output.detach().numpy()[:, 0, :].reshape(-1))
                y_classcification_original.append(
                    y2.detach().numpy()[:, 0, :].reshape(-1))

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
        loss_mape = mean_absolute_percentage_error(y_original, y_predict)*100
        r2 = r2_score(y_original, y_predict)

        print(f"Loss MAE: {loss_mae}")
        print(f"Loss RMSE: {loss_rmse}")
        print(f"Loss MAPE: {loss_mape}")
        print(f"R2: {r2}")

        return y_classcification_predict, y_predict, y_original

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
