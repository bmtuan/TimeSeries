import torch
import torch.nn as nn

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_seq_len, kernel_size, embed_size): 
        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(self.input_size, embed_size, kernel_size = self.kernel_size, stride = 1, padding = int((self.kernel_size-1)/2))
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        # x = x.permute(0,2,1)
        # x = torch.tanh(self.conv1(x))
        # x = x.permute(0,2,1)
        out, state = self.lstm(x)
  
        return out, state
    
class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, kernel_size, embed_size):
        super(LSTM_Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,

            batch_first=True,
        )
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(self.input_size, embed_size, kernel_size = self.kernel_size, stride = 1, padding = int((self.kernel_size-1)/2))
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.Relu = nn.ReLU()

    def forward(self, x, state):
        # x = x.permute(0,2,1)
        # x = torch.tanh(self.conv1(x))
        # x = x.permute(0,2,1)
        output, hidden_state = self.lstm(x, (state))
        output = self.Relu(output)
        out_pm25 = self.fc1(output)

        return out_pm25, hidden_state


class LSTM(nn.Module):
    def __init__(self,input_seq_len,output_seq_len,number_layer,input_size,output_size,hidden_size,embed_size,kernel_size,device):
        super(LSTM, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.number_layer = number_layer
        self.input_seq_len = input_seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        self.kernel_size = kernel_size
        self.encoder = LSTM_Encoder(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers= self.number_layer,
            input_seq_len=self.input_seq_len,
            kernel_size=kernel_size,
            embed_size=embed_size
            )
        self.decoder = LSTM_Decoder(
            input_size=self.input_size, 
            output_size=self.output_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.number_layer,
            kernel_size=kernel_size,
            embed_size=embed_size
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