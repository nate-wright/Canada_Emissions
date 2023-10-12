import pandas as pd 
from energy_preparation import * 
from climate_preparation import * 
from gdp_preparation import * 
from policy_preparation import * 
from unemployment_preparation import * 
from wildfire_preparation import * 
from population_preparation import * 
from emissions_preparation import * 
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import grangercausalitytests 
from statsmodels.tsa.vector_ar.vecm import coint_johansen 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import numpy as np 
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
# import plotly.offline as py
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

all_dataframes = []
# first gather all dataframes 
annual_emissions = get_total_emissions_annual() 
annual_energy = generate_annual_energy_dataset() 
all_dataframes.append(annual_energy)
#give climate years 2020 and 2021 
annual_climate = generate_annual_climate_dataframe() 
all_dataframes.append(annual_climate)
#might have to take out unnessasary columns in annual gdp and give gdp years 2020 and 2021 
annual_gdp = generate_gdp_dataframe() 
all_dataframes.append(annual_gdp)
annual_policy = generate_policy_dataframe() 
all_dataframes.append(annual_policy)
annual_unemployment = generate_unemployment_dataframe()
all_dataframes.append(annual_unemployment) 
annual_wildfire = generate_annual_wildfire_dataframe() 
all_dataframes.append(annual_wildfire)
annual_population = generate_population_dataframe()
all_dataframes.append(annual_population)

print(annual_emissions['Total_emissions'])

gdp_merge = pd.merge(annual_gdp, annual_emissions, on="Year")
energy_merge = pd.merge(annual_energy, annual_emissions, on="Year")
climate_merge = pd.merge(annual_emissions, annual_climate, on="Year")

unemployment_merge = pd.merge(annual_emissions, annual_unemployment, on="Year")
big_merge = pd.merge(unemployment_merge, annual_wildfire, on="Year")

population_merge = pd.merge(annual_population, annual_emissions, on="Year")

policy_merge = pd.merge(annual_emissions, annual_policy, on="Year")

# create a massive dataframe combining everything by year 
merged_df = annual_emissions
for df in all_dataframes: 
    merged_df = pd.merge(merged_df, df, on="Year")

all_column_names = merged_df.columns.tolist() 
print(all_column_names)

# remove last three columns as they all have NaN values 
columns_to_remove = ['Net interprovincial migration', 'Non-permanent residents, inflows', 'Non-permanent residents, outflows']
merged_df = merged_df.drop(columns=columns_to_remove)

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
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

# displaying the grangers for the df with no lag values 
# pd.set_option('display.max_columns', None)
grangers_matrix = grangers_causation_matrix(merged_df, variables=all_column_names)
print(grangers_matrix.iloc[0])
# we want to remove the columns from our merged_df that do not cause total_emissions (p value greater than 0.08)

# loop through columns in row to find p values greater than 0.08 
granger_columns = grangers_matrix.columns.tolist() 
granger_columns.remove("Total_emissions_x")
cols_to_drop = []
for col in granger_columns:
    p_value = grangers_matrix.at["Total_emissions_y", col]
    if p_value > 0.08: 
        cols_to_drop.append(col[0:len(col) - 2])

cols_to_drop.append("Year")
new_df = merged_df.drop(columns=cols_to_drop)
print(new_df.head)

#split into training and testing datasets 
train_size = int(0.8 * len(new_df))
train, test = new_df[:train_size], new_df[train_size:]

# check for stationarity + make stationary using augmented dickey fuller test 
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.") 

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
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

# next thing I am going to do is figure out the model that gives the most accurate Total_emissions value
# I will test the training data after being differentiated zero, once and twice on lag values from 0-7 
training_sets = [train, train_diff, train_diff_2]
max_lag_value = 8 
num_of_difs = 0
min_mean = None 
min_results = None 

for ts in training_sets:
    col_string = ""
    if num_of_difs > 0: 
        col_string = "_" + str(num_of_difs) + "d"
        print(col_string)
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


model = VAR(new_df)
model_fitted = model.fit(7)

# forecast the model 
forecast_input = new_df.values[-7:]

fc = model_fitted.forecast(y=forecast_input, steps=3)
# df_forecast = pd.DataFrame(fc, index=new_df.index[-(len(new_df) - len(train)):], columns=new_df.columns + '_2d')
df_forecast = pd.DataFrame(fc, index=['2020', '2021', '2022'], columns=new_df.columns )



#df_results = invert_transformation(train, df_forecast, second_diff=True)
df_results = df_forecast 
print(df_results.head)

fig, axes = plt.subplots(nrows=int(len(new_df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(new_df.columns, axes.flatten())):
    #df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_results[col].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    # test[col][-(len(new_df) - len(train)):].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()
# print(model_fitted.summary())

# forecast = model.predict(start=len(train), end = len(train) + 2)

# plt.figure(figsize=(12, 6))
# plt.plot(test.index, test['total_emissions'], label='Actual', color='blue')
# plt.plot(test.index, forecast[:, 0], label='Forecast', color='red')
# plt.legend()
# plt.title('VAR Forecast vs Actual')
# plt.show()


# run a prophet model on the dataframe 
# df_with_lag = df_with_lag.rename(columns={'Year': 'df', 'Total_emissions': "y"})
# print(df_with_lag.head)
# model = Prophet()

# ARIMA MODEL CODE BELOW 
# p, d, q = 3, 1, 1

# all_columns = df_with_lag.columns.tolist() 
# all_columns.remove("Year")
# all_columns.remove("Total_emissions")
# model = ARIMA(train['Total_emissions'], order=(p, d, q), exog=train[all_columns])
# model_fit = model.fit()
# print(model_fit.summary())

# predict = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, exog=test[all_columns]).rename('ARIMA Predictions')
# predict.plot(legend=True) 
# test['Total_emissions'].plot(legend=True)

# #plot_predict(model_fit, 2011, 2019, exog=train[all_columns])
# plt.show()
# # Forecast future values
# forecast_steps = len(test)

# forecast, stderr = model_fit.forecast(steps=forecast_steps, exog=train[all_columns])


# # Assuming test.index and forecast are pandas Series
# test_index_list = test.index.tolist()
# forecast_list = forecast.tolist()

# # Plotting with synchronized data
# plt.plot(test_index_list, forecast_list, label='Forecast', color='red')


# # Visualize the results
# plt.figure(figsize=(12, 6))
# plt.plot(test.index, test['Total_emissions'], label='Actual', color='blue')
# plt.plot(test.index, forecast, label='Forecast', color='red')
# plt.legend()
# plt.title('ARIMAX Forecast vs Actual')
# plt.show()
