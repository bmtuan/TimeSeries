import torch
import torch.nn as nn
from torch.autograd import Variable

def init_hidden(x: torch.Tensor, hidden_size: int, num_layer: int,xavier: bool = True):
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_layer, x.shape[0], hidden_size)).to(device)
    return Variable(torch.zeros(num_layer, x.shape[0], hidden_size)).to(device)

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
    
class LSTMAttention_Decoder(nn.Module):
    def __init__(self, input_seq_len, hidden_size, num_layers, input_size, output_size, device): 
        super(LSTMAttention_Decoder,self).__init__()

        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.num_layers =  num_layers
        self.input_size = input_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim = self.hidden_size, num_heads = 1)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.leakyRelu = nn.LeakyReLU(0.01)

        self.device = device    

    def forward(self, x, prev_hidden):
        
        x, (h_out, c_out) = self.lstm(x, prev_hidden)
        q = self.leakyRelu(self.fc1(x))
        k = self.leakyRelu(self.fc1(x))
        v = self.leakyRelu(self.fc1(x))
        attn_output, _ = self.attention(q,k,v)
        out = self.leakyRelu(self.fc2(attn_output))
        return out, (h_out, c_out)


class AttentionLSTM(nn.Module):
    def __init__(self,input_seq_len,output_seq_len,number_layer,input_size,output_size,hidden_size,device):
        super(AttentionLSTM, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.number_layer = number_layer
        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        self.encoder = LSTM_Encoder(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers= self.number_layer,
            input_seq_len=self.input_seq_len,
            )
        self.decoder = LSTMAttention_Decoder(
            input_size=self.input_size, 
            output_size=self.output_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.number_layer, 
            input_seq_len=self.input_seq_len,
            device=self.device,
            )

    def forward(self, x):
        outputs_feature = torch.zeros(x.shape[0], self.output_seq_len, self.output_size)
        _, input_encoded = self.encoder(x)
        decoder_input = x[:, -1, :]
        decoder_input = torch.reshape(decoder_input, (x.shape[0], 1, x.shape[2]))
        decoder_hidden = input_encoded

        for t in range(self.output_seq_len):
            out_feature, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            outputs_feature[:, t, :] = out_feature.squeeze(1)
            decoder_input = out_feature.to(self.device)

        return outputs_feature
    

class MultiTaskLSTM(nn.Module):
    def __init__(self,input_seq_len,output_seq_len,number_layer,input_size,output_size,hidden_size,device):
        super(MultiTaskLSTM, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.number_layer = number_layer
        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        self.encoder = LSTM_Encoder(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers= self.number_layer,
            input_seq_len=self.input_seq_len,
            )
        
        self.decoder = nn.ModuleList()
        for i in range(self.input_size):
            self.decoder.append(LSTMAttention_Decoder(
            input_size=1, 
            output_size=1, 
            hidden_size=self.hidden_size, 
            num_layers=1, 
            input_seq_len=self.input_seq_len,
            device=self.device))

    def forward(self, x):
        outputs_feature = torch.zeros(x.shape[0], self.output_seq_len, self.output_size)
        inputs_feature = torch.zeros(x.shape[0], 1, self.input_size).to(self.device)
        hidden_state = torch.zeros(self.input_size, x.shape[0], self.hidden_size).to(self.device)
        cell_state = torch.zeros(self.input_size, x.shape[0], self.hidden_size).to(self.device)

        _, input_encoded = self.encoder(x)

        decoder_input = x[:, -1, :]
        decoder_input = torch.reshape(decoder_input, (x.shape[0], 1, x.shape[2]))

        hidden = torch.mean(input_encoded[0], 0).unsqueeze(0)
        cell = torch.mean(input_encoded[1] , 0).unsqueeze(0)
        for t in range(self.output_seq_len):
            if t == 0:
                for i in range(self.input_size):
                    out_feature, decoder_hidden = self.decoder[i](decoder_input[:, : , i].clone().unsqueeze(1), (hidden, cell))
                    
                    hidden_state[i, :, : ] = decoder_hidden[0]
                    cell_state[i, :, : ] = decoder_hidden[1]
                    outputs_feature[:, t, i] = out_feature.squeeze(1).squeeze(1)
                    inputs_feature[:, :, i] = out_feature.squeeze(1)
                                    
                decoder_input = inputs_feature
            else:
                for i in range(self.input_size):
                    out_feature, decoder_hidden = self.decoder[i](decoder_input[:, : , i].clone().unsqueeze(1), 
                                                                  (hidden_state[i, : ,:].clone().unsqueeze(0), cell_state[i, :, :].clone().unsqueeze(0)))
                    
                    hidden_state[i, :, : ] = decoder_hidden[0]
                    cell_state[i, :, : ] = decoder_hidden[1]
                    outputs_feature[:, t, i] = out_feature.squeeze(1).squeeze(1)
                    inputs_feature[:, :, i] = out_feature.squeeze(1)
                                    
                decoder_input = inputs_feature
        return outputs_feature

class ClusteringMultiTaskLSTM(nn.Module):
    def __init__(self,input_seq_len,output_seq_len,number_layer,input_size,output_size,hidden_size,cluster,device):
        super(ClusteringMultiTaskLSTM, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.number_layer = number_layer
        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        self.cluster = cluster
        self.encoder = nn.ModuleList()
        for i in range(max(self.cluster) + 1):
            self.encoder.append(LSTM_Encoder(
                input_size=self.cluster.count(i), 
                hidden_size=self.hidden_size, 
                num_layers= self.number_layer,
                input_seq_len=self.input_seq_len,
                )
            )
        self.decoder = nn.ModuleList()
        for i in range(self.input_size):
            self.decoder.append(LSTMAttention_Decoder(
            input_size=1, 
            output_size=1, 
            hidden_size=self.hidden_size, 
            num_layers=1, 
            input_seq_len=self.input_seq_len,
            device=self.device))
            
            
    def merge_hidden(self, tensor_1, tensor_2):
        
        hidden_state = torch.zeros(2, tensor_1.shape[1], self.hidden_size).to(self.device)
        hidden_state[0, : , :] = tensor_1
        hidden_state[1, : , :] = tensor_2
        
        return torch.mean(hidden_state, 0).unsqueeze(0)
    
    
    def forward(self, x):
        outputs_feature = torch.zeros(x.shape[0], self.output_seq_len, self.output_size)
        inputs_feature = torch.zeros(x.shape[0], 1, self.input_size).to(self.device)
        hidden_state = torch.zeros(self.input_size, x.shape[0], self.hidden_size).to(self.device)
        cell_state = torch.zeros(self.input_size, x.shape[0], self.hidden_size).to(self.device)
        list_x = [torch.zeros(x.shape[0], self.input_seq_len, self.cluster.count(i)).to(self.device) for i in range(max(self.cluster) + 1)]   
        hidden = torch.zeros(max(self.cluster) + 1, x.shape[0] ,self.hidden_size).to(self.device)
        cell = torch.zeros(max(self.cluster) + 1, x.shape[0] ,self.hidden_size).to(self.device)
        
        for i in range(max(self.cluster) + 1):
            count = 0
            for index, j in enumerate(self.cluster):
                if j == i:
                    list_x[i][:, : , count] = x[:,:,index]
                    count += 1
        
        for i in range(max(self.cluster) + 1):
            _, input_encoded = self.encoder[i](list_x[i])
            hidden[i, :, :] = input_encoded[0][-1:,:,:]
            cell[i, :, :] = input_encoded[1][-1,:,:]

        mean_hidden = torch.mean(hidden, 0).unsqueeze(0)
        mean_cell = torch.mean(cell , 0).unsqueeze(0)
        
        decoder_input = x[:, -1, :]
        decoder_input = torch.reshape(decoder_input, (x.shape[0], 1, x.shape[2]))
        
        for t in range(self.output_seq_len):
            if t == 0:
                for i in range(self.input_size):
                    out_feature, decoder_hidden = self.decoder[i](
                        decoder_input[:, : , i].clone().unsqueeze(1), 
                        (self.merge_hidden(hidden[self.cluster[i]], mean_hidden) , self.merge_hidden(cell[self.cluster[i]], mean_cell)))
                    
                    hidden_state[i, :, : ] = decoder_hidden[0]
                    cell_state[i, :, : ] = decoder_hidden[1]
                    outputs_feature[:, t, i] = out_feature.squeeze(1).squeeze(1)
                    inputs_feature[:, :, i] = out_feature.squeeze(1)
                                    
                decoder_input = inputs_feature
            else:
                for i in range(self.input_size):
                    out_feature, decoder_hidden = self.decoder[i](decoder_input[:, : , i].clone().unsqueeze(1), 
                                                                  (hidden_state[i, : ,:].clone().unsqueeze(0), cell_state[i, :, :].clone().unsqueeze(0)))
                    
                    hidden_state[i, :, : ] = decoder_hidden[0]
                    cell_state[i, :, : ] = decoder_hidden[1]
                    outputs_feature[:, t, i] = out_feature.squeeze(1).squeeze(1)
                    inputs_feature[:, :, i] = out_feature.squeeze(1)
                                    
                decoder_input = inputs_feature
                
        return outputs_feature

if __name__ == '__main__':
    # from torchsummary import summary
    from torchinfo import summary
    device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")
    print(device)
    cluster = [1 ,2 ,1 , 1, 2, 0, 2, 0]
    sizex = (32,10, 1) # batch, length, feature
    model = MultiTaskLSTM(
        input_seq_len=10,
        output_seq_len=10,
        number_layer=1,
        input_size=sizex[2],
        output_size=sizex[2],
        hidden_size=32,
        device=device,
        # cluster=cluster
    )
    model.to(device)
    
    x = torch.rand(sizex).to(device)
    # y = model(x)
    # print(y.shape)
    summary(model, sizex)