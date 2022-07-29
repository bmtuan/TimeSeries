import matplotlib.pyplot as plt
import pandas as pd 
import os
import numpy as np
# def plot_results(y_original, y_predict, folder, filename, y_inference=None):
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     plt.figure(figsize=(30, 14))
#     plt.title("PM2_5")
#     plt.plot(y_original, label="Original")
#     plt.plot(y_predict, label="Predcit")
#     if y_inference is not None:
#         plt.plot(y_inference, label="Inference")
#     plt.xlabel("Time steps")
#     plt.legend(loc="upper center")
#     plt.savefig(os.path.join(folder,filename), dpi=200)  # save the figure to file
col = ['Task 1','Task 2','Task 3','Task 4','Task 5','Task 6','Task 7','Task 8', ]
dir = 'data/multitask_train'
df_result = pd.DataFrame()
for path in os.listdir(dir):
    df = pd.read_csv(os.path.join(dir, path))
    df_result[path[:-4]] = df['PM2_5'].values[-40000:]
method = ['pearson']
for m in method:
    corr = df_result.corr(method = m)
corr = np.round(corr, 2)
values = corr.values
df = pd.DataFrame(values, columns=col, index=col)
print(df)
for i in range(8):
    high = []
    high.append(f'Task {i+1}')
    for j in range(8):
        if j <= i:
            continue
        if values[i,j] > 0.3:
            high.append(f'Task {j+1}')
    print(high)
# print(df)
# df.to_csv('result/corr.csv')