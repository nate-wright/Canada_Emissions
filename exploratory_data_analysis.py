import pandas as pd
from Emissions_VAR_helper import best_diff_and_lag_value, generate_merged_dataframe, remove_with_grangers
from VAR_helper import adfuller_test, grangers_causation_matrix, invert_transformation 
from energy_preparation import * 
from climate_preparation import * 
from gdp_preparation import * 
from policy_preparation import * 
from unemployment_preparation import * 
from wildfire_preparation import * 
from population_preparation import * 
from emissions_preparation import * 
from statsmodels.tsa.api import VAR
import numpy as np 

merged_df = generate_merged_dataframe() 
all_column_names = merged_df.columns.tolist() 

all_column_names = merged_df.columns.tolist() 
all_column_names.remove('Year')
# create lag variables going back three years and test correlation 

all_series = []
all_years = merged_df['Year'].unique() 
for i in range(3, len(all_years)):
    year = all_years[i]
    this_df = merged_df[merged_df['Year'] == year]
    lag_dfs = []
    for j in range(1, 4):
        lag_dfs.append(merged_df[merged_df['Year'] == (year - j)])
    data_for_series = {'Year': year}
    for col in all_column_names: 
        data_for_series[col] = this_df[col].sum() 
        for j in range(len(lag_dfs)):
            this_lag_df = lag_dfs[j]
            data_for_series[col + " - "+ str(j + 1)] = this_lag_df[col].sum() 
    
    new_series = pd.Series(data=data_for_series)
    all_series.append(new_series)

pd.set_option('display.max_rows', None)
df_with_lag = pd.DataFrame(all_series)
print(df_with_lag.corr()['Total_emissions']) 


# VAR CODE BELOW
# run a vector auto regression model on the dataframe 

# need to use grangers causation tests to test if the other time series values cause total_emissions 
# the p value should be around 0.05 or lower to indicate a causation 

maxlag = 1
new_df = remove_with_grangers(merged_df, all_column_names)

#split into training and testing datasets 
train_size = int(0.8 * len(new_df))
train, test = new_df[:train_size], new_df[train_size:]

for column in train: 
    adfuller_test(train[column], name=column)
    print('\n')

# none of the columns are stationary, lets difference all them 
train_diff = train.diff().dropna()
for column in train_diff: 
    adfuller_test(train_diff[column], name=column)
    print('\n')

# diff again 
train_diff_2 = train_diff.diff().dropna() 
for column in train_diff_2: 
    adfuller_test(train_diff_2[column], name=column)
    print('\n')

# need to de-difference twice as I differenced twice 

# next thing I am going to do is figure out the model that gives the most accurate Total_emissions value
# I will test the training data after being differentiated zero, once and twice on lag values from 0-7 
training_sets = [train, train_diff, train_diff_2]
max_lag_value = 8 
num_of_difs = 0

optimal_model_setup = best_diff_and_lag_value(new_df, training_sets, test, train, max_lag_value)

for ts in training_sets:
    col_string = ""
    if num_of_difs > 0: 
        col_string = "_" + str(num_of_difs) + "d"
    for i in range(1, max_lag_value):
        model = VAR(ts)
        model_fitted = model.fit(i)
        forecast_input = ts.values[-i:]
        fc = model_fitted.forecast(y=forecast_input, steps=len(new_df) - len(train))
        df_forecast = pd.DataFrame(fc, index=new_df.index[-(len(new_df) - len(train)):], columns=new_df.columns + col_string)
        df_results = df_forecast
        if num_of_difs == 1: 
            df_results = invert_transformation(train, df_forecast, second_diff=False)
        elif num_of_difs == 2: 
            df_results = invert_transformation(train, df_forecast, second_diff=True)
        
        # rename to forecast if it has not been changed by inverting the transformation 
        if num_of_difs == 0: 
            df_results = df_results.rename(columns={"Total_emissions": "Total_emissions_forecast"})
        # get difference between forecasted and actual values for Total_emissions 
        print(df_results['Total_emissions_forecast'])
        mean = np.mean(df_results['Total_emissions_forecast'].values - test['Total_emissions'])
        print("NUMBER OF DIFS: " + str(num_of_difs))
        print("NUMBER OF LAGS: " + str(i))
        print("MEAN: " + str(mean))
        if min_mean is None or abs(mean) < min_mean: 
            min_mean = abs(mean) 
            min_results = df_results 

    num_of_difs += 1
    max_lag_value -= 1

# this is currently manually inputted, to test against test data you must input the training data, to test further you can input all data
model = VAR(train)
model_fitted = model.fit(7)

# forecast the model 
forecast_input = train.values[-7:]

fc = model_fitted.forecast(y=forecast_input, steps=3)
df_forecast = pd.DataFrame(fc, index=new_df.index[-(len(new_df) - len(train)):], columns=new_df.columns)
df_results = df_forecast 

# forecast into 2017, 2018 and 2019
fig, axes = plt.subplots(nrows=int(len(new_df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(new_df.columns, axes.flatten())):
    df_results[col].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    test[col][-3:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()

# next I will create a forecast for the 10 years after our data set (2020-2030)
# create model based on all the data 
full_model = VAR(new_df)
model_fitted_full = model.fit(7)
# forecast the model 
forecast_input_full = new_df.values[-7:]

fc = model_fitted_full.forecast(y=forecast_input_full, steps=10)
df_forecast_full = pd.DataFrame(fc, index=['2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029'], columns=new_df.columns)
df_results_full = df_forecast_full


# forecast into 2020-2029
fig, axes = plt.subplots(nrows=int(len(new_df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(new_df.columns, axes.flatten())):
    df_results_full[col].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()


