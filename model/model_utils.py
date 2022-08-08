from sklearn.metrics import mean_absolute_percentage_error
import torch
import numpy as np
from utils import *
from tqdm import tqdm
import os
import wandb
from LSTMAttention import AttentionLSTM

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

def test_mean_ensemble_model(list_model, iterator):
    
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
        plot_results(
            feature_original,
            feature_predict[:len(feature_original)],
            "output/",
            f"test_cyclical_ensemble.png",
        )

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

def inference_cyclical_ensemble(list_model, test_df, input_lengths, output_length):
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
            feature_input = input[len(input) - input_lengths[idx] : len(input)]
            feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
            tensor_x = torch.FloatTensor(feature_input).to(device)
            feature_output = model.forward(tensor_x)
            feature_output = feature_output.detach().cpu().numpy()[0, :, 0].reshape(-1)
            feature_outputs[idx] = feature_output
        for k in range(output_length):
            ensemble_outputs.append(np.sum(feature_outputs[:, k] * W))
        ensemble_outputs = np.array(ensemble_outputs).reshape(output_length,1)
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
        f"inference_cyclical_ensemble.png",
    )
            
        
def train(model,train_iterator,valid_iterator,num_epochs,criterion,model_path,optimizer,scheduler):
    
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
                # print('feature loss:', feature_loss.item())
                # # total = 0
                # for i in range(4):
                #     single_loss =  criterion(output_feature[:, :, i], y[:, : ,i])
                #     # total += single_loss.item()
                #     print(f'single loss {i}: ', single_loss.item())
                # print('total loss: ', total / 8)
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
            feature_predict.append(np.append(
                feature_output.detach().cpu().numpy()[:, 0, :].reshape(-1), 
                feature_output.detach().cpu().numpy()[-1, 1:, :].reshape(-1)))
            feature_original.append(np.append(
                y.detach().cpu().numpy()[:, 0, :].reshape(-1), 
                y.detach().cpu().numpy()[-1, 1:, :].reshape(-1)))
            
    feature_predict = np.reshape(feature_predict, (-1))
    feature_original = np.reshape(feature_original, (-1))

    return evaluate_metrics(feature_original, feature_predict)
    # plot_results(feature_original, feature_predict,'/media/aimenext/disk1/tuanbm/TimeSerires/model/output','test_batch_feature.png')


def inference_1(model, test_df, input_length, output_length, off_size):
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
        feature_output = model.forward(tensor_x)
        feature_output = feature_output.detach().numpy().squeeze(0)

        
        feature_predict = np.concatenate((feature_predict, feature_output))
        original_feature = np.concatenate((original_feature, feature[i:i+output_length]))
        input = np.concatenate((input, feature_output))
        i += output_length
        count += output_length
        input = np.concatenate((input, feature[i : i + off_size]))
        i += off_size

    # cal loss

    _, _, _, mape , _ = evaluate_metrics(original_feature.reshape(-1), feature_predict[:len(original_feature)].reshape(-1))

    percent_save = cal_energy(count, len(feature)) * 100
    if percent_save > 100:
        percent_save = 100
        
    # print(f"{count}/{feature.shape[0]}")
    print(f"Save {percent_save}% times")
    
    return mape, percent_save
    # plot_results(
    #     original_feature,
    #     feature_predict[:len(original_feature)],
    #     "output/",
    #     f"inference_{name}.png",
    # )

def inference_2(model, test_df, input_length, output_length, beta = 0.5, off_size=5):
    feature = test_df.iloc[:, :].values

    original_feature = feature[:1]
    feature_predict = feature[:1]
    input = feature[0:input_length]
    i = 0
    count = 0
    while i < len(feature):
        feature_input = input[len(input) - input_length : len(input)]

        feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
        tensor_x = torch.FloatTensor(feature_input).to(device)
        feature_output = model.forward(tensor_x)
        feature_output = feature_output.detach().numpy().squeeze(0)
        t = -1
        for idx in range(len(feature_output) - 1):
            if abs(feature_output[idx] - feature_output[idx-1]) > beta:
                t = idx
                break
        if t == -1:
            t = output_length
        
        feature_predict = np.concatenate((feature_predict, feature_output[:t]))
        original_feature = np.concatenate((original_feature, feature[i:i+t]))
        input = np.concatenate((input, feature_output[:t]))
        i += t
        count += t
        input = np.concatenate((input, feature[i : i + off_size]))
        i += off_size

    # cal loss

    _, _, _, mape, _ = evaluate_metrics(original_feature.reshape(-1), feature_predict[:len(original_feature)].reshape(-1))

    percent_save = cal_energy(count, len(feature)) * 100
    if percent_save > 100:
        percent_save = 100
        
    # print(f"{count}/{feature.shape[0]}")
    print(f"Save {percent_save}% times")

    return mape, percent_save

def inference_3(model, test_df, input_length, output_length, mape_threshold = 5, off_size = 5):
    feature = test_df.iloc[:, :].values

    original_feature = feature[:1]
    feature_predict = feature[:1]
    input = feature[0:input_length]
    i = 0
    count = 0
    import pdb
    while i < len(feature):
        # prepare input
        feature_input = input[len(input) - input_length : len(input)]

        feature_input = feature_input.reshape(1, feature_input.shape[0], feature_input.shape[1])
        tensor_x = torch.FloatTensor(feature_input).to(device)
        # print(tensor_x.shape)
        feature_output = model.forward(tensor_x)
        feature_output = feature_output.detach().numpy().squeeze(0)

        
        feature_predict = np.concatenate((feature_predict, feature_output))
        original_feature = np.concatenate((original_feature, feature[i:i+output_length]))
        input = np.concatenate((input, feature_output))
        i += output_length
        count += output_length
        input = np.concatenate((input, feature[i : i + off_size]))
        i += off_size
        check_mape = 100
        
        predict = []
        original = []
        if i > len(feature):
            break
        while i<len(feature) and check_mape > mape_threshold:
            # print(input[-input_length:].shape)
            x = input[-input_length:].reshape(1, input[-input_length:].shape[0], 1)
            tensor_x = torch.FloatTensor(x).to(device)
            # print('tensor_x: ', tensor_x)
            outputs = model(tensor_x)
            # print(outputs.detach().cpu().numpy().reshape(-1))
            predict.append(outputs.detach().cpu().numpy().reshape(-1)[0])
            original.append(feature[i])
            
            input = np.concatenate((input, feature[i : i + 1]))
            i += 1
            # print('original: ', original)
            # print('predict: ', predict)
            check_mape = mean_absolute_percentage_error(original, predict) * 100
            # print(check_mape)
            # pdb.set_trace()
    # cal loss

    _, _, _, mape, _ = evaluate_metrics(original_feature.reshape(-1), feature_predict[:len(original_feature)].reshape(-1))

    percent_save = cal_energy(count, len(feature)) * 100
    if percent_save > 100:
        percent_save = 100
        
    # print(f"{count}/{feature.shape[0]}")
    print(f"Save {percent_save}% times")

    return mape, percent_save
def inference_ensemble_final(list_model, test_df, input_lengths, output_length, mape_threshold):
    feature = test_df.iloc[:, :].values

    original_feature = feature[:1]
    feature_predict = feature[:1]
    input = feature[0:60]
    i = 0
    count = 0
    W = np.array([1, 1, 1])
    W = [w / np.sum(W) for w in W]
    while i < len(feature):
        ensemble_outputs = ensemble_predict(list_model, input_lengths, output_length, input)
        feature_predict = np.concatenate((feature_predict, ensemble_outputs))
        original_feature = np.concatenate((original_feature, feature[i:i+output_length]))
        
        input = np.concatenate((input, ensemble_outputs))
        i += output_length
        count += output_length
        input = np.concatenate((input, feature[i : i + 5]))
        i += 5
        
        check_mape = 100
        predict = []
        original = []
        if i > len(feature):
            break
        while check_mape > mape_threshold:
            ensemble_outputs = ensemble_predict(list_model, input_lengths, output_length, input)
            predict.append(ensemble_outputs[0])
            original.append(feature[i])
            
            input = np.concatenate((input, feature[i : i + 1]))
            i += 1
            check_mape = mean_absolute_percentage_error(original, predict) * 100
        
        

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
        f"inference_ensemble_final.png",
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
        
    return np.array(ensemble_outputs).reshape(output_length, 1)


def load_list_model(path, input_lengths, output_length, number_layer, input_size, output_size, hidden_size, device):
    list_model = []
    list_paths = []
    for length in input_lengths:
        list_paths.append(os.path.join(path, f'{length}_AttentionLSTM/best.pth'))
    for model_path in list_paths:
        model = AttentionLSTM(
                input_seq_len=length,
                output_seq_len=output_length,
                number_layer=number_layer,
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                device=device,
            ) 

        model.to(device)
        model.load_state_dict(copyStateDict(torch.load(model_path)))
        list_model.append(model)
    return list_model




def mutil_predict(model, iterator, num_task, test=None):
    result = np.zeros((8,5))
    feature_original = np.zeros((num_task,0))
    feature_predict = np.zeros((num_task,0))
    
    with torch.no_grad():
        for x, y in iterator:
            
            x, y = x.to(device), y.to(device)
            feature_output = model.forward(x)  # batch_size, output_seq, num_feature
            feature_output = feature_output.detach().cpu().numpy()
            # print(feature_output)
            y = y.detach().cpu().numpy()
            
            feature_predict = np.concatenate((
                feature_predict, 
                np.concatenate((feature_output[:, 0, :], feature_output[-1, 1:, :]), axis=0).transpose()), axis=1)
            feature_original = np.concatenate((
                feature_original, 
                np.concatenate((y[:, 0, :], y[-1, 1:, :]), axis=0).transpose()), axis=1)
    if test != None:
        return evaluate_metrics(feature_original[test], feature_predict[test])
    
    for i in range(num_task):
        print(f'Task {i+1}')
        metrics = evaluate_metrics(feature_original[i], feature_predict[i])
        result[i] = metrics
        

def multi_inference(model, iterator, num_task, test=None):
    result = np.zeros((8,5))
    feature_original = np.zeros((num_task,0))
    feature_predict = np.zeros((num_task,0))
    
    with torch.no_grad():
        for x, y in iterator:
            
            x, y = x.to(device), y.to(device)
            feature_output = model.forward(x)  # batch_size, output_seq, num_feature
            feature_output = feature_output.detach().cpu().numpy()
            # print(feature_output)
            y = y.detach().cpu().numpy()
            
            feature_predict = np.concatenate((
                feature_predict, 
                np.concatenate((feature_output[:, 0, :], feature_output[-1, 1:, :]), axis=0).transpose()), axis=1)
            feature_original = np.concatenate((
                feature_original, 
                np.concatenate((y[:, 0, :], y[-1, 1:, :]), axis=0).transpose()), axis=1)
    if test != None:
        return evaluate_metrics(feature_original[test], feature_predict[test])
    
    for i in range(num_task):
        print(f'Task {i+1}')
        metrics = evaluate_metrics(feature_original[i], feature_predict[i])
        result[i] = metrics