from flat_ae import Autoencoder
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

def run_eval(model_path, train_path, test_path):

    model = torch.load(model_path, weights_only=False)

    # Load train data
    train = pd.read_csv(train_path)
    train = train.sort_index(axis=1)
    train_groups = train['group_index'].unique()
    sensor_cols = [col for col in train.columns if 'sensor_' in col]

    criterion = nn.MSELoss()

    model.eval()
    train_label = []
    train_loss = []
    for group in train_groups:
        df = train[train['group_index'] == group]
        label = df['group_label'].unique()[0]
        train_label.append(label)

        input = torch.tensor(df[sensor_cols].to_numpy(), dtype=torch.float32) / model.sensor_scales
        input = torch.nan_to_num(input, nan=0.0)
        input = input.reshape(1, -1)

        decoded = model(input)
        reconstruct_loss = criterion(decoded, input)
        reconstruct_loss = reconstruct_loss.item()
        train_loss.append(reconstruct_loss)


    # Load test data
    test = pd.read_csv(test_path)
    test = test.sort_index(axis=1)
    test_groups = test['group_index'].unique()

    test_label = []
    test_loss = []
    for group in test_groups:
        df = test[test['group_index'] == group]
        label = df['group_label'].unique()[0]
        test_label.append(label)

        input = torch.tensor(df[sensor_cols].to_numpy(), dtype=torch.float32) / model.sensor_scales
        input = torch.nan_to_num(input, nan=0.0)
        input = input.reshape(1, -1)

        decoded = model(input)
        reconstruct_loss = criterion(decoded, input)
        reconstruct_loss = reconstruct_loss.item()
        test_loss.append(reconstruct_loss)

    return train_label, train_loss, test_label, test_loss

# train_label, train_loss, test_label, test_loss = run_eval(model_path='models/flat_ae.pt', train_path='sensor_train.csv', test_path='sensor_test.csv')

def roc_auc_curve(labels, losses):
    fpr, tpr, thresholds = roc_curve(labels, losses)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")

    best_i = np.argmax(np.array(tpr) - np.array(fpr))
    best_threshold = thresholds[best_i]
    best_fpr = fpr[best_i]
    best_tpr = tpr[best_i]
    print(f"Optimal Threshold: {best_threshold} with tpr: {best_tpr} and fpr: {best_fpr}")

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(best_fpr, best_tpr, 'or', label=f'Optimal Threshold {best_threshold:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return best_threshold

# roc_auc_curve(test_label, test_loss)

def plot_histograms(values, labels, title, threshold):
    values_0 = [values[i] for i, label in enumerate(labels) if label == 0]
    values_1 = [values[i] for i, label in enumerate(labels) if label == 1]
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(values_0, bins=np.arange(0, 0.15, 0.005), alpha=0.7, label='Label 0')
    plt.hist(values_1, bins=np.arange(0, 0.15, 0.005), alpha=0.7, label='Label 1')
    plt.plot([threshold, threshold], [0, max(plt.gca().get_ylim())], 'r--')
    
    plt.title(f'Distribution of Reconstruction Loss by Class for {title}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot_histograms(test_loss, test_label, title='Test Set')
# plot_histograms(train_loss, train_label, title='Train') 