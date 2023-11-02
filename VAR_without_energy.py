import pandas as pd
from VAR_helper import adfuller_test, grangers_causation_matrix, invert_transformation 
from energy_preparation import * 
from climate_preparation import *
from Emissions_VAR_helper import generate_merged_dataframe, remove_with_grangers 
from gdp_preparation import * 
from policy_preparation import * 
from unemployment_preparation import * 
from wildfire_preparation import * 
from population_preparation import * 
from emissions_preparation import * 
from statsmodels.tsa.api import VAR
import numpy as np 


# running the VAR model without the energy dataframe to see if it is more accurate as there is more data to pull from 
df_no_energy = generate_merged_dataframe(energy=False)
all_column_names = df_no_energy.columns.tolist() 
all_column_names.remove('Year')
print(df_no_energy.head)

# need to use grangers causation tests to test if the other time series values cause total_emissions 
# the p value should be around 0.05 or lower to indicate a causation 
maxlag = 2
new_df_no_energy = remove_with_grangers(df_no_energy, all_column_names, maxlag)


#split into training and testing datasets
train_size = int(0.8 * len(new_df_no_energy))
train, test = new_df_no_energy[:train_size], new_df_no_energy[train_size:]

for column in train: 
    adfuller_test(train[column], name=column)
    print('\n')

# Returning emigrants and total emissions are not stationary 

# lets differentiate 
train_diff = train.diff().dropna()
for column in train_diff: 
    adfuller_test(train_diff[column], name=column)
    print('\n')

# once more 
train_diff_2 = train.diff().dropna() 
for column in train_diff_2: 
    adfuller_test(train_diff_2[column], name=column)
    print('\n')

# next lets figure out the model that gives the best emissions value 
# we will test with a variety of lag values and differentiations from 0-2 





