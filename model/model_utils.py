from sklearn.metrics import mean_absolute_percentage_error
import torch
import numpy as np
from utils import evaluate_metrics, plot_metrics, plot_results, cal_energy, synth_mape
from tqdm import tqdm
import os
import wandb
import copy

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

def test_mutil_model(list_model, iterator):
    
    feature_original = []
    feature_predict = []
    
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            feature_outputs = np.zeros((len(list_model), y.shape[1]))
            ensemble_outputs = []
            for idx, model in enumerate(list_model):
                feature_output = model.forward(x)  # batch_size, output_seq, num_feature
                feature_outputs[idx] = feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1)
            for i in range(y.shape[1]):
                ensemble_outputs.append(np.mean(feature_outputs[:, i]))
            
            feature_predict.extend(ensemble_outputs)
            feature_original.append(y.detach().cpu().numpy()[0, :, 0].reshape(-1))

        feature_predict = np.reshape(feature_predict, (-1))
        feature_original = np.reshape(feature_original, (-1))
        
        print('feature')
        evaluate_metrics(feature_original, feature_predict)

def test_error_model(list_model, test_iterator):
    
    feature_original = []
    feature_predict = []
    W = np.array([4.17, 4.4, 5.35])
    W = [w / np.sum(W) for w in W]
    with torch.no_grad():
        for x, y in test_iterator:
            x, y = x.to(device), y.to(device)
            feature_outputs = np.zeros((len(list_model), y.shape[1]))
            ensemble_outputs = []
            for idx, model in enumerate(list_model):
                feature_output = model.forward(x)  # batch_size, output_seq, num_feature
                feature_outputs[idx] = feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1)
            for i in range(y.shape[1]):
                ensemble_outputs.append(np.sum(feature_outputs[:, i] * W))
            feature_predict.extend(ensemble_outputs)
            feature_original.append(y.detach().cpu().numpy()[0, :, 0].reshape(-1))

        feature_predict = np.reshape(feature_predict, (-1))
        feature_original = np.reshape(feature_original, (-1))
        
        evaluate_metrics(feature_original, feature_predict)

def inference_ensemble(list_model, test_df, input_length, output_length, name):
    feature = test_df.iloc[:, :].values

    original_feature = feature[:1]
    feature_predict = feature[:1]
    input = feature[0:60]
    i = 0
    count = 0
    W = np.array([1, 1, 1])
    W = [w / np.sum(W) for w in W]
    while i < len(feature):
        # prepare input
        feature_outputs = np.zeros((len(list_model), output_length))
        ensemble_outputs = []
        for idx, model in enumerate(list_model):
            feature_input = input[len(input) - input_length[idx] : len(input)]
            feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
            tensor_x = torch.FloatTensor(feature_input).to(device)
            feature_output = model.forward(tensor_x)
            feature_output = feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1)
            feature_outputs[idx] = feature_output
        for k in range(output_length):
            ensemble_outputs.append(np.sum(feature_outputs[:, k] * W))
        ensemble_outputs = np.array(ensemble_outputs).reshape(output_length,1)
        # print(synth_mape(ensemble_outputs.reshape(-1).tolist()))
        feature_predict = np.concatenate((feature_predict, ensemble_outputs))
        original_feature = np.concatenate((original_feature, feature[i:i+output_length]))
        
        input = np.concatenate((input, ensemble_outputs))
        i += output_length
        count += output_length
        input = np.concatenate((input, feature[i : i + 20]))
        i += 20

    # cal loss
    evaluate_metrics(original_feature, feature_predict[:len(original_feature)])

    percent_save = cal_energy(count, len(feature)) * 100
    if percent_save > 100:
        percent_save = 100
        
    print(f"{count}/{feature.shape[0]}")
    print(f"Save {percent_save}% times")

    plot_results(
        original_feature,
        feature_predict[:len(original_feature)],
        "output/",
        f"inference_{name}.png",
    )
            
        
def train(
    model,
    train_iterator,
    valid_iterator,
    num_epochs,
    criterion,
    model_path,
    optimizer,
    scheduler):
    
    list_train_loss = []
    list_val_loss = []
            
    best_loss = 999999
    for epoch in tqdm(range(num_epochs)):
            epoch_train_loss = 0
            for x, y in train_iterator:
                
                x, y = x.to(device), y.to(device)
                feature_loss = 0

                optimizer.zero_grad()
                
                output_feature = model.forward(x)
                output_feature = output_feature.to(device)
           
                feature_loss = criterion(output_feature, y)
                
                feature_loss.backward()
                optimizer.step()
                epoch_train_loss += feature_loss.item()

            train_loss = epoch_train_loss / len(train_iterator)
            epoch_val_loss = 0

            with torch.no_grad():
                for x, y in valid_iterator:
                    
                    x, y = x.to(device), y.to(device)
                    output_feature = model.forward(x)
                    
                    output_feature = output_feature.to(device)
                    feature_loss = criterion(output_feature, y)
                    
                    
                    epoch_val_loss += feature_loss.item()

                val_loss = epoch_val_loss / len(valid_iterator)


                if val_loss < best_loss:
                    name = 'best.pth'
                    torch.save(model.state_dict(), os.path.join(model_path, name))
                    best_loss = val_loss
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                })

            scheduler.step()
            list_train_loss.append(train_loss)
            list_val_loss.append(val_loss)


def predict(model, iterator):
    
    feature_original = []
    feature_predict = []
    
    with torch.no_grad():
        for x, y in iterator:
            
            x, y = x.to(device), y.to(device)
            feature_output = model.forward(x)  # batch_size, output_seq, num_feature

            # feature_predict.append(feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1))
            # feature_original.append(y.detach().cpu().numpy()[0, :, 0].reshape(-1))
            feature_predict.append(np.append(
                feature_output.detach().cpu().numpy()[:, 0, :].reshape(-1), 
                feature_output.detach().cpu().numpy()[-1, 1:, :].reshape(-1)))
            feature_original.append(np.append(
                y.detach().cpu().numpy()[:, 0, :].reshape(-1), 
                y.detach().cpu().numpy()[-1, 1:, :].reshape(-1)))
        
            
    feature_predict = np.reshape(feature_predict, (-1))
    feature_original = np.reshape(feature_original, (-1))

    evaluate_metrics(feature_original, feature_predict)
    plot_results(feature_original, feature_predict,'/media/aimenext/disk1/tuanbm/TimeSerires/model/output','test_batch_feature.png')


def inference_cyclical(list_model, test_df, input_length, output_length, name):
    feature = test_df.iloc[:, :].values

    original_feature = feature[:1]
    feature_predict = feature[:1]
    input = feature[0:input_length]
    i = 0
    count = 0
    while i < len(feature):
        # prepare input
        feature_input = input[len(input) - input_length : len(input)]

        feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
        tensor_x = torch.FloatTensor(feature_input).to(device)
        feature_output = list_model.forward(tensor_x)
        feature_output = feature_output.detach().numpy().squeeze(0)

        
        feature_predict = np.concatenate((feature_predict, feature_output))
        original_feature = np.concatenate((original_feature, feature[i:i+output_length]))
        input = np.concatenate((input, feature_output))
        i += output_length
        count += output_length
        input = np.concatenate((input, feature[i : i + 20]))
        i += 20

    # cal loss
    evaluate_metrics(original_feature, feature_predict[:len(original_feature)])

    percent_save = cal_energy(count, len(feature)) * 100
    if percent_save > 100:
        percent_save = 100
        
    print(f"{count}/{feature.shape[0]}")
    print(f"Save {percent_save}% times")

    plot_results(
        original_feature,
        feature_predict[:len(original_feature)],
        "output/",
        f"inference_{name}.png",
    )


def inference_realtime(list_model, test_df, input_length):
    test_df = test_df[-input_length:]
    feature_input = test_df.iloc[:, :].values

    W = np.array([1, 1, 1])
    W = [w / np.sum(W) for w in W]
    
    feature_outputs = np.zeros((len(list_model), 60))
    ensemble_outputs = []
    for idx, model in enumerate(list_model):
        feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
        tensor_x = torch.FloatTensor(feature_input).to(device)
        feature_output = model.forward(tensor_x)
        feature_output = feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1)
        feature_outputs[idx] = feature_output
        
    for k in range(60):
        ensemble_outputs.append(np.sum(feature_outputs[:, k] * W))


def inference_ensemble_final(list_model, test_df, input_lengths, output_length, name):
    feature = test_df.iloc[:, :].values

    original_feature = feature[:1]
    feature_predict = feature[:1]
    input = feature[0:60]
    i = 0
    count = 0
    W = np.array([1, 1, 1])
    W = [w / np.sum(W) for w in W]
    turn_on = False
    while i < len(feature):
        ensemble_outputs = ensemble_predict(list_model, input_lengths, output_length, input)
        s_mape = synth_mape(ensemble_outputs)
        for mape in s_mape:
            # print(mape)
            if mape > 0.001:
                turn_on = True
        feature_predict = np.concatenate((feature_predict, ensemble_outputs))
        original_feature = np.concatenate((original_feature, feature[i:i+output_length]))
        
        input = np.concatenate((input, ensemble_outputs))
        i += output_length
        count += output_length
        if turn_on:
            input = np.concatenate((input, feature[i : i + 5]))
            i += 5
            
            check_mape = 100
            predict = []
            original = []
            if i > len(feature):
                break
            while check_mape > 2:
                ensemble_outputs = ensemble_predict(list_model, input_lengths, output_length, input)
                predict.append(ensemble_outputs[0])
                original.append(feature[i])
                
                input = np.concatenate((input, feature[i : i + 1]))
                i += 1
                # print(i)
                check_mape = mean_absolute_percentage_error(original, predict) * 100
                # print(check_mape)
        
        

    # cal loss
    evaluate_metrics(original_feature, feature_predict[:len(original_feature)])

    percent_save = cal_energy(count, len(feature)) * 100
    if percent_save > 100:
        percent_save = 100
        
    print(f"{count}/{feature.shape[0]}")
    print(f"Save {percent_save}% times")

    plot_results(
        original_feature,
        feature_predict[:len(original_feature)],
        "output/",
        f"inference_{name}.png",
    )
    
    
def ensemble_predict(list_model, input_lengths, output_length, input):
    feature_outputs = np.zeros((len(list_model), output_length))
    ensemble_outputs = []
    W = np.array([1, 1, 1])
    W = [w / np.sum(W) for w in W]
    for idx, model in enumerate(list_model):
        feature_input = input[len(input) - input_lengths[idx] : len(input)]
        feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
        tensor_x = torch.FloatTensor(feature_input).to(device)
        feature_output = model.forward(tensor_x)
        feature_output = feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1)
        feature_outputs[idx] = feature_output
    for k in range(output_length):
        ensemble_outputs.append(np.sum(feature_outputs[:, k] * W))
    
    
    # ensemble_outputs = np.array(ensemble_outputs).reshape(output_length, 1)
    return np.array(ensemble_outputs).reshape(output_length, 1)