from init import *


class PMDataset(Dataset):
    def __init__(self, data_df, input_len, output_len, transform=None):
        self.data = data_df
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
                    self.input_len + self.output_len, 1:2]
        return X, Y1, Y2

    def __len__(self):
        return self.data_len - (self.input_len + self.output_len - 1)
