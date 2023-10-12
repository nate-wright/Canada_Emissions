import pandas as pd 
import matplotlib.pyplot as plt 

# data is from 1990 - 2022
# get Canadian annual unemployment rates 

def generate_unemployment_dataframe() -> pd.DataFrame: 
    unemployment_df = pd.read_csv('data/unemployment_rate_annual.csv')
    Canada_unemployment = unemployment_df[(unemployment_df['GEO'] == 'Canada') & (unemployment_df['Labour force characteristics'] == 'Unemployment rate') & (unemployment_df['Sex'] == 'Both sexes')]
    all_years = Canada_unemployment['REF_DATE'].unique()
    list_of_series = []
    for year in all_years: 
        this_year_data = Canada_unemployment[Canada_unemployment['REF_DATE'] == year]
        series_data = {'Year': year, 'Unemployment_percent': this_year_data['VALUE'].sum()}
        new_series = pd.Series(data=series_data)
        list_of_series.append(new_series)
    
    Canada_unemployment_consolidated = pd.DataFrame(list_of_series)
    Canada_unemployment_consolidated['Year'] = pd.to_numeric(Canada_unemployment_consolidated.Year)
    return Canada_unemployment_consolidated


#generate_unemployment_dataframe()
