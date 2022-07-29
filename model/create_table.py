import pandas as pd

multi = pd.read_csv('result/multi_task.csv')
single = pd.read_csv('result/single_task.csv')
col = ['Task','MAE', 'MSE', 'RMSE', 'MAPE', 'CORR']
method = []
for i in range(8):
    method.append('Single Task')
    method.append('Multi Task')
df = pd.DataFrame(columns=col)

for i in range(0,16,2):
    df.loc[i] = single.iloc[int(i/2)].values
    df.loc[i+1] = multi.iloc[int(i/2)].values
df.insert(1, "Method", method)
df.to_csv('result/table.csv')