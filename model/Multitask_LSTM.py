import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, hidden_size):
        super(FullyConnected, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.Relu = nn.ReLU()
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.Relu(out)
        out = self.fc2(out)
        out = self.Relu(out)
        out = self.fc3(out)
        out = self.Relu(out)
        return out

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

class LSTMAttention(nn.Module):
    def __init__(self, input_seq_len, hidden_size, num_layers, input_size, output_size, device): 
        super(LSTMAttention,self).__init__()

        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.num_layers =  num_layers
        self.input_size = input_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim = self.hidden_size, num_heads = 1)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc_task = nn.ModuleList(FullyConnected(hidden_size=self.hidden_size).to(device) for i in range(self.input_size))
        self.leakyRelu = nn.LeakyReLU(0.01)

        self.device = device    
 
    def forward(self, x, prev_hidden):

        x, (h_out, c_out) = self.lstm(x, prev_hidden)
        q = self.leakyRelu(self.fc1(x))
        k = self.leakyRelu(self.fc1(x))
        v = self.leakyRelu(self.fc1(x))
        attn_output, _ = self.attention(q,k,v)

        out = [fc(attn_output) for fc in self.fc_task]

        out_tensor = out[0]
        for i in range(1, self.input_size):
            out_tensor = torch.cat((out_tensor, out[i]), 2)

        return out_tensor, (h_out, c_out)


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
        self.decoder = LSTMAttention(
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


if __name__ == '__main__':
    device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskLSTM(
        input_seq_len=1,
        output_seq_len=1,
        number_layer=1,
        input_size=8,
        output_size=8,
        hidden_size=32,
        device=device
    )
    model.to(device)
    size = (32, 1, 8) # batch, length, feature
    x = torch.rand(size).to(device)
    y = model(x)
    print(y.shape)