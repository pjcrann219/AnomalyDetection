import pandas as pd
import random

def generate_sets(LENGTH, train_percent):   
    # Load in data, convert timestamp
    df = pd.read_csv('data/sensor.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add group index, cutoff last incomplete set, add group label
    df['group_index'] = df.index // LENGTH
    last_group = max(df['group_index']) - 1
    df = df[df['group_index'] <= last_group]
    df['group_label'] = df['machine_status'].isin(['BROKEN', 'RECOVERING']).groupby(df['group_index']).transform('any').astype(int)

    # Get list of group true/false sets
    true_group_ids = df[df['group_label'] == 1]['group_index'].unique().tolist()
    false_group_ids = df[df['group_label'] == 0]['group_index'].unique().tolist()

    train_ids = random.sample(false_group_ids, k=int(len(false_group_ids)*train_percent))

    test_ids = [i for i in false_group_ids if i not in train_ids]
    test_ids.extend(true_group_ids)

    df_train = df[df['group_index'].isin(train_ids)]
    df_test = df[df['group_index'].isin(test_ids)]

    print(f"train: num_ids: {len(train_ids)}, status's: {df_train['machine_status'].unique()}")
    print(f"test: num_ids: {len(test_ids)}, status's: {df_test['machine_status'].unique()}")

    df_train.to_csv('data/sensor_train.csv')
    df_test.to_csv('data/sensor_test.csv')

LENGTH = 500
train_percent = 0.90
generate_sets(LENGTH, train_percent)
