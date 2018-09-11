# %%  trick sampling

"""
1 - Categorico?
        - crea categoria falsa con muchos datos

    regresion?
        - crea categoria falsa + categorias para los resultados

2 - applica smote o adasyn

3 - sobre escribe los resultados del train

4 - Testear

5 - close #19 y #20
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

data = pd.read_csv('data/data_train_cat_nas.csv')
data.rename(columns={'SalePrice': 'Y'}, inplace=True)
data['Y'] = data['Y'].astype('float64')

# Artificial Categories for numerical data
num_categories = 250
q_transformer = QuantileTransformer()
temp_Y = q_transformer.fit_transform(data['Y'].sort_values().values.reshape(-1, 1))

temp_splits_frontier = np.linspace(start=0, stop=temp_Y.max(), num=num_categories)
temp_splits_cat = np.linspace(start=1, stop=num_categories, num=num_categories).astype('int')
temp_splits = pd.concat([pd.DataFrame(temp_splits_frontier, columns=['Y_frontier']),
                         pd.DataFrame(temp_splits_cat, columns=['Y_category'])], axis=1)

temp_Y = pd.concat([pd.DataFrame(temp_Y), pd.DataFrame([0] * len(temp_Y))], axis=1)
temp_Y.columns = ['Y', 'Y_category']

cat_pos = 0  # Holds the actual category position assigned to "Y"
for element in range(len(temp_Y['Y'])):
    if temp_Y.iloc[element, 0] > temp_splits.iloc[cat_pos, 0]:
        cat_pos += 1

    temp_Y.iloc[element, 1] = temp_splits.iloc[cat_pos, 1]  # Assign the proper category

temp_Y['Y'] = q_transformer.inverse_transform(temp_Y['Y'].values.reshape(-1, 1))
data['Y_category'] = temp_Y['Y_category'].values

# Calculate the size of the dummy category
size_categories = round(temp_Y.groupby('Y_category').size().max() - temp_Y.groupby('Y_category').size().mean())
no_dummy_cat_size = num_categories * size_categories  # Size with already created categories

data_size_desired = 20000  # Temporal para tener referencia

dummy_cat_size = round(data_size_desired/(num_categories + 1))
if dummy_cat_size > temp_Y.groupby('Y_category').size().max():  # New category will do an effect
    # Create Dummy Category
    dummy_cat = data.sample(n=dummy_cat_size, replace=True)
    dummy_cat['Y_category'] = (num_categories + 1)

    data = pd.concat([data, dummy_cat]).reset_index(drop=True)

# Apply Over-Sampling
from imblearn.over_sampling import SMOTE

data_overspl, _ = SMOTE(k_neighbors=2).fit_sample(data.drop(columns=['Y_category']), data['Y_category'])


#%% Data - Feature Selection
from sklearn.ensemble import RandomForestRegressor
from itertools import compress
## Recursive Feature Elimination
from sklearn.feature_selection import RFE

rfr = RandomForestRegressor()  # We select a ML estimator
rfe = RFE(estimator=rfr, n_features_to_select=round(len(train_data.columns)*0.6), step=1)
rfe.fit(train_data.drop(columns=['Y']), train_data['Y'])
select_features_rfe = list(rfe.get_support())

train_data_columns = list(train_data.columns)
train_data_columns.remove('Y')
train_data_columns_tokeep = list(compress(train_data_columns, select_features_rfe))

train_data_reduced = train_data.loc[:,train_data_columns_tokeep]
train_data_reduced = pd.concat([train_data_reduced, train_data['Y']], axis=1)

#%% Data Augmentation
from numpy.random import normal as normal_gen


def oversampler(data: pd.DataFrame, sample_size: int, std_coef: float = 0.5) -> pd.DataFrame:
    """
    Over samples data, treating each column independently, naive assumption of data distribution as "normal"
    :param data: numerical data
    :type data: pd.DataFrame
    :param sample_size: sample size to be generated
    :type sample_size: int
    :param std_coef: standar deviation multiplication coefficient
    :type std_coef: float
    :return: Data oversampled - pd.DataFrame
        Same columns as the given data
    """

    column_names = data.columns
    data_oversampled = pd.DataFrame([[0] *  len(column_names)] * sample_size,columns=column_names)

    for column in column_names:
        column_data = data[column]

        noise = normal_gen(loc= 0, scale=column_data.std() * std_coef, size=sample_size)
        temp_oversampled = column_data.sample(n=sample_size, replace=True)
        temp_oversampled = temp_oversampled.add(noise)

        data_oversampled[column] = temp_oversampled.values

    return data_oversampled


train_data_over = oversampler(data=train_data_reduced, sample_size=500, std_coef=0.25)