from model import *
from utils import *
from init import *
from const import *
from dataloader import *


num_epochs = 20
learning_rate = 0.001

df = pd.read_csv('data/final_envitus.csv')
df = df.drop(['datetime'], axis=1)
pm2_5 = df.iloc[:, 0:1].values


turn_on = cal_synthetic_turn_on(
    SYNTHETIC_THRESHOLD, SYNTHETIC_SEQUENCE_LENGTH, pm2_5)
df['turn_on'] = turn_on


train_df, valid_df, test_df, sc_train, sc_val, sc_test = get_train_valid_test_data(
    df)


train_dataset = PMDataset(train_df, input_len=120, output_len=10)
valid_dataset = PMDataset(valid_df, input_len=120, output_len=10)
test_dataset = PMDataset(test_df, input_len=120, output_len=10)
# use drop_last to get rid of last batch
train_iterator = DataLoader(
    train_dataset, batch_size=32, shuffle=False, drop_last=True)
valid_iterator = DataLoader(
    valid_dataset, batch_size=32, shuffle=False, drop_last=True)
test_iterator = DataLoader(test_dataset, batch_size=32,
                           shuffle=False, drop_last=True)


model = lstm_seq2seq(input_seq_len=INPUT_SEQUENCE_LENGTH,
                     output_seq_len=OUTPUT_SEQUENCE_LENGTH)
model.to(device)


list_train_loss = []
list_val_loss = []
best_loss = 999999
losses = np.full(num_epochs, np.nan)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
criterion_binary = nn.BCELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)


for epoch in tqdm(range(num_epochs)):
    epoch_train_loss = 0
    for x, y1, y2 in train_iterator:
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

        linear_loss = 0
        binary_loss = 0

        optimizer.zero_grad()

        outputs, outputs_prob = model.forward(x)
        outputs, outputs_prob = outputs.to(device), outputs_prob.to(device)

        linear_loss = criterion(outputs, y1)
        binary_loss = criterion_binary(outputs_prob, y2)
        loss = (1 - COEFFICIENT_LOSS) * linear_loss + \
            COEFFICIENT_LOSS * binary_loss

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
            outputs, outputs_prob = model.forward(x)
            outputs, outputs_prob = outputs.to(device), outputs_prob.to(device)

            linear_loss = criterion(outputs, y1)
            binary_loss = criterion_binary(outputs_prob, y2)
            loss = (1 - COEFFICIENT_LOSS) * linear_loss + \
                COEFFICIENT_LOSS * binary_loss
            epoch_val_loss += loss.item()

        val_loss = epoch_val_loss / len(valid_iterator)
        if val_loss < best_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'\tSave best checkpoint with best loss: {val_loss:.4f}')
            best_loss = val_loss
        scheduler.step()
        list_train_loss.append(train_loss)
        list_val_loss.append(val_loss)
        print(f'\t Val loss: {epoch_val_loss / len(valid_iterator):.4f}')

plot_metrics(list_train_loss, list_val_loss, 'output/', 'metric.png')
