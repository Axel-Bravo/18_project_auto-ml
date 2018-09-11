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
from math import floor
from sklearn.preprocessing import QuantileTransformer

data = pd.read_csv('data/data_train_cat_nas.csv')
data.drop(columns=['Unnamed: 0', 'Id'], inplace=True)
data.rename(columns={'SalePrice': 'Y'}, inplace=True)

# Artificial Categories for numerical data
num_categories = 250
smote_k_neighbors = 4
data_size_desired = 5000  # Temporal para tener referencia

if num_categories > floor(len(data['Y'])/smote_k_neighbors):  # Number of categories limiter
    num_categories = floor(len(data['Y'])/smote_k_neighbors)
    print('Number of synthetic categories has been limited, due to "SMOTE" requirements.')

q_transformer = QuantileTransformer()
temp_Y = q_transformer.fit_transform(data['Y'].sort_values().values.reshape(-1, 1))
temp_Y_category = pd.DataFrame(np.linspace(start=0, stop=len(temp_Y),num=num_categories+1))
temp_Y_category = temp_Y_category.apply(lambda x: floor(float(x)), axis=1)

temp_Y = pd.concat([pd.DataFrame(temp_Y, columns=['Y']),
                    pd.DataFrame([0]*len(temp_Y), columns=['Y_category'])], axis=1)

for element in range(1, len(temp_Y['Y'])):
    temp_Y.iloc[element, 1] = temp_Y_category[temp_Y_category < element].index.max()  # Assign the proper category

temp_Y['Y'] = q_transformer.inverse_transform(temp_Y['Y'].values.reshape(-1, 1))
data['Y_category'] = temp_Y['Y_category'].values




dummy_cat_size = round(data_size_desired/(num_categories + 1))
if dummy_cat_size > temp_Y.groupby('Y_category').size().max():  # New category will do an effect
    # Create Dummy Category
    dummy_cat = data.sample(n=dummy_cat_size, replace=True)
    dummy_cat['Y_category'] = (num_categories + 1)
    data = pd.concat([data, dummy_cat]).reset_index(drop=True)

# Apply Over-Sampling
from imblearn.over_sampling import SMOTE

data_overspl, _ = SMOTE(k_neighbors=smote_k_neighbors).fit_sample(data.drop(columns=['Y_category']), data['Y_category'])
data_overspl = pd.DataFrame(data_overspl, columns=data.drop(columns=['Y_category']).columns)

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
