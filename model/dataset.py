from model.init import *


class PMDataset(Dataset):
    def __init__(self, data_df, input_len, output_len, transform=None):
        self.data = data_df
        self.number_column = self.data.columns
        
        # name = ["PM2_5", "PM10", "temp", "humidity","turn_on"]
        # number = [1, 2, 3, 4, 5]
        # dic = dict(zip(name, number))
        # self.dic = dic
        
        self.transform = transform
        self.input_len = input_len
        self.output_len = output_len
        self.data_len = self.data.shape[0]
        self.X = np.array(self.data).astype(np.float32)

    def __getitem__(self, index):

        X = self.X[index:index + self.input_len, :]
        Y1 = self.X[index + self.input_len: index +
                    self.input_len + self.output_len, 0:1]
        Y2 = self.X[index + self.input_len: index +
                    self.input_len + self.output_len, -1:]
        return X, Y1, Y2

    def __len__(self):
        return self.data_len - (self.input_len + self.output_len - 1)
