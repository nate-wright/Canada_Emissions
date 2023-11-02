import pandas as pd
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

# This file is for VAR helper functions specific to predicting canadian emissions data 

# This function merges all dataframes collected for the emissions data into one 
def generate_merged_dataframe(energy=True):
    all_dataframes = []
    # first gather all dataframes 
    annual_emissions = get_total_emissions_annual() 
    if energy: 
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
    # create a massive dataframe combining everything by year 
    merged_df = annual_emissions
    for df in all_dataframes: 
        merged_df = pd.merge(merged_df, df, on="Year")
    # remove last three columns as they all have NaN values 
    columns_to_remove = ['Net interprovincial migration', 'Non-permanent residents, inflows', 'Non-permanent residents, outflows']
    merged_df = merged_df.drop(columns=columns_to_remove)
    return merged_df


#this function removes columns from the dataframe that do not correlate with total emissions based on grangers causation matrix
def remove_with_grangers(df: pd.DataFrame, all_column_names, maxlag=1) -> pd.DataFrame:
    # displaying the grangers for the df with no lag values 
    grangers_matrix = grangers_causation_matrix(df, variables=all_column_names, maxlag=maxlag)
    print(grangers_matrix.iloc[0])
    # we want to remove the columns from our df that do not cause total_emissions (p value greater than 0.08)

    # loop through columns in row to find p values greater than 0.08 
    granger_columns = grangers_matrix.columns.tolist() 
    granger_columns.remove("Total_emissions_x")
    cols_to_drop = []
    for col in granger_columns:
        p_value = grangers_matrix.at["Total_emissions_y", col]
        if p_value > 0.08: 
            cols_to_drop.append(col[0:len(col) - 2])

    cols_to_drop.append("Year")
    new_df = df.drop(columns=cols_to_drop)
    print(new_df.head)
    return new_df