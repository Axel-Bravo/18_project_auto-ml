#%% Imports & declarations
import numpy as np
import pandas as pd
import AutoMl as ml


def data_preprocessing(dataset: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    temp_data = dataset.iloc[:, 1:]
    id_data = dataset.iloc[:, 0]
    if temp_data.shape[1] == 55:
        return temp_data.iloc[:, :-1], temp_data.iloc[:, -1], id_data
    else:
        return temp_data, id_data


#%% Data load and preprocessing
data_train = pd.read_csv('data/train.csv')
X_train, y_train, id_train = data_preprocessing(data_train)

y_train = pd.DataFrame(y_train)
y_train.columns = ['Y']

train = pd.concat([X_train, y_train], axis=1)

#%% AutoMl use
bot = ml.AutoMl(name='test')
bot.load_data(train_data=train, test_data=train)
bot.optimize(n_jobs=1)
