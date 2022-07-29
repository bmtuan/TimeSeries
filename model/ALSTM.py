import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
class Model(nn.Module):
  def __init__(self, input_size, rnn_hidden_unit, num_stocks, seq_len, mlp_hidden=16, drop_prob=0.5):
    super(Model, self).__init__()
    self.hidden_size = rnn_hidden_unit * num_stocks
    self.rnn_hidden_unit = rnn_hidden_unit
    self.input_size = input_size
    self.drop_prob = drop_prob
    self.seq_len = seq_len

    self.lstm1 = nn.LSTM(input_size, self.hidden_size, batch_first=True)
    self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
    self.embedding = nn.Linear(self.input_size, self.hidden_size)
    self.attn = nn.Linear(self.hidden_size * 2, self.seq_len)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.drop_prob)
    self.lstm3 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
    self.bn_1 = nn.BatchNorm1d(self.hidden_size)
    self.mlp_1 = nn.Linear(rnn_hidden_unit, mlp_hidden)
    self.act_1 = nn.ReLU()
    self.mlp_2 = nn.Linear(mlp_hidden, 1)

  def forward(self, inputs):
    # print(inputs.shape)
    batch_size = inputs.shape[0]
    self.num_stock = inputs.shape[1]
    num_features = inputs.shape[3]
    inputs = inputs.permute(0, 2, 1, 3) # batch_size, seq_len, num_stock, num_feature
    inputs = torch.reshape(inputs, (batch_size , self.seq_len, num_features * self.num_stock))
    output, h = self.lstm1(inputs)
    enc_output, (h_s, c_s) = self.lstm2(output, h)

    # Dùng torch zeros để khởi tạo input cũng được. Hoặc dùng time step trước đó là inputs[:, -1, :]
    # embedded = self.embedding(torch.zeros(inputs.size(0), 1, inputs.size(2)))
    embedded = self.embedding(inputs[:, -1:, :])
    embedded = self.dropout(embedded)
    attn_weights = F.softmax(self.attn(torch.cat((embedded, h_s[-1].unsqueeze(1)), 2)), dim=2)
    attn_applied = torch.bmm(attn_weights, enc_output)
    output = torch.cat((embedded, attn_applied), 2)
    output = self.attn_combine(output)
    output = F.relu(output)
    output, _ = self.lstm3(output, (h_s, c_s))
    output = output[:, -1, :]
    output = self.bn_1(output) # batch norm cần nhiều hơn 1 giá trị. (batch_size != 1)
    output = self.dropout(output)
    output = torch.reshape(output, (batch_size, self.num_stock, 1, self.rnn_hidden_unit))
    output = self.mlp_1(output)
    output = F.relu(output)
    output = self.mlp_2(output)
    output = torch.reshape(output, (batch_size, self.num_stock, 1))
    return output
  
  


if __name__ == '__main__':
  writer = SummaryWriter()
      
  model  = Model(
    input_size=10,
    rnn_hidden_unit=12,
    num_stocks=5,
    seq_len=10
  )
  size = (10, 5, 10, 2)
  x = torch.rand(size)
  writer.add_graph(model, x)
  writer.close()
  # summary(model, input_size=(10, 5, 10, 2))
  # print(x.shape)
  # y = model(x)
  # make_dot(y, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
