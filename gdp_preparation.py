import pandas as pd 
import matplotlib.pyplot as plt 

# data is from 1997 - 2019 
# total gdp in entire nation annually in millions 
def generate_gdp_dataframe() -> pd.DataFrame: 
    gdp = pd.read_csv('data/Canada_Total_GDP.csv')
    Canada_gdp = gdp[gdp['GEO'] == 'Canada']
    all_series = []
    all_years = Canada_gdp['REF_DATE'].unique() 
    for year in all_years: 
        this_year = Canada_gdp[Canada_gdp['REF_DATE'] == year]
        gdp = this_year['VALUE'].sum() 
        data_for_series = {"Year": year, "GDP": gdp}
        new_series = pd.Series(data_for_series)
        all_series.append(new_series)
    
    Canada_gdp_consolidated = pd.DataFrame(all_series)

    # need to add total gdp in years 2020 and 2021 manually 
    #     
    return Canada_gdp_consolidated


#generate_gdp_dataframe()