import pandas as pd
from VAR_helper import adfuller_test, grangers_causation_matrix, invert_transformation 
from energy_preparation import * 
from climate_preparation import *
from Emissions_VAR_helper import best_diff_and_lag_value, generate_merged_dataframe, remove_with_grangers 
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
training_sets = [train, train_diff, train_diff_2]
max_lag_value = 10 

optimal_model_setup = best_diff_and_lag_value(new_df_no_energy, training_sets, test, train, max_lag_value)

print(optimal_model_setup)
diff = optimal_model_setup[0]
lag = optimal_model_setup[1]

# Test training data vs actual test data 
string_to_add = ""
ts = train
model = VAR(train)
if diff == 1: 
    model = VAR(train_diff)
    string_to_add = '_1d'
    ts = train_diff
elif diff == 2: 
    model = VAR(train_diff_2)
    string_to_add = '_2d'
    ts = train_diff_2

model_fitted = model.fit(lag)
forecast_input = ts.values[-lag:]

fc = model_fitted.forecast(y=forecast_input, steps=len(new_df_no_energy) - len(train))
df_forecast = pd.DataFrame(fc, index=new_df_no_energy.index[-(len(new_df_no_energy) - len(train)):], columns=new_df_no_energy.columns + string_to_add)
df_results = df_forecast 

if diff == 1: 
    df_results = invert_transformation(train, df_forecast)

elif diff == 2: 
    df_results = invert_transformation(train, df_forecast, second_diff=True)

print(test.head)
# forecast into 2015-2019
fig, axes = plt.subplots(nrows=int(len(new_df_no_energy.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(new_df_no_energy.columns, axes.flatten())):
    df_results[col + '_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    test[col][-5:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()




