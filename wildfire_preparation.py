import pandas as pd 
import matplotlib.pyplot as plt 

# takes individual wildfire data from 1953 - 2021, specifies the date and hectares burned 
# this function takes the total hectares burned each year 
def generate_annual_wildfire_dataframe() -> pd.DataFrame: 
    wildfire = pd.read_csv('data/CANADA_WILDFIRES.csv')
    dates = wildfire['REP_DATE'].unique() 
    all_years = []
    for date in dates: 
        if str(date)[:4] not in all_years: 
            all_years.append(str(date)[:4])
    
    sorted_years = sorted(all_years)
    sorted_years = sorted_years[len(sorted_years) - 33: len(sorted_years) - 1]
    all_series = []
    for year in sorted_years: 
        this_year_wildfire = wildfire[wildfire['REP_DATE'].str[:4] == year]
        data_for_series = {"Year": year, "SIZE_HA": this_year_wildfire['SIZE_HA'].sum()}
        new_series = pd.Series(data=data_for_series, index=["Year", "SIZE_HA"])
        all_series.append(new_series)

    final_df = pd.DataFrame(all_series)
    final_df['Year'] = pd.to_numeric(final_df.Year)
    return final_df


#generate_annual_wildfire_dataframe() 
