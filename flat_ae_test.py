from eval import *
import pandas as pd
import numpy as np

train_label, train_loss, test_label, test_loss = run_eval(model_path='models/flat_ae_250.pt', train_path='data/sensor_train.csv', test_path='data/sensor_test.csv')

threshold = roc_auc_curve(test_label, test_loss)

plot_histograms(test_loss, test_label, title='Test Set', threshold=threshold)
plot_histograms(train_loss, train_label, title='Train Set', threshold=threshold) 

# threshold = 0.04439916834235191
# df = pd.read_csv('data/sensor.csv')
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# length = len(df)
# sensor_cols = [col for col in df.columns if 'sensor_' in col]

# model_path='models/flat_ae_250.pt'
# model = torch.load(model_path, weights_only=False)

# criterion = nn.MSELoss()

# errors = []
# labels = []
# steps = []

# model.eval()
# for i in range(0, length - 500, 10):

#     input = torch.tensor(df[sensor_cols][i:i+500].to_numpy(), dtype=torch.float32) / model.sensor_scales
#     input = torch.nan_to_num(input, nan=0.0)
#     input = input.reshape(1, -1)

#     decoded = model(input)
#     reconstruct_loss = criterion(decoded, input)
#     reconstruct_loss = reconstruct_loss.item()

#     label = reconstruct_loss >= threshold

#     steps.append(i)
#     errors.append(reconstruct_loss)
#     labels.append(label)

# new_column = pd.Series(errors, index=steps)
# df['reconstruct_loss'] = new_column

# unique_strings = df['machine_status'].unique()
# colors = plt.cm.viridis(np.linspace(0, 1, len(unique_strings)))

# plt.figure()
# for i, string_label in enumerate(unique_strings):
#     subset = df[df['machine_status'] == string_label]
#     plt.scatter(subset['timestamp'], subset['reconstruct_loss'], 
#                 c=[colors[i]], marker='x', label=string_label)
# plt.legend()
# plt.show()