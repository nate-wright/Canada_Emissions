import pandas as pd 
import matplotlib.pyplot as plt 
# takes climate data from 1940-2019 across major cities in canada including temperature and precipitation 
# want to create an annual estimation of canadian temperature and precipitation by averaging out the values of the major cities for a given year 
# focus on later years (ex. 1990-2019)

def generate_annual_climate_dataframe() -> pd.DataFrame:
    canada_climate = pd.read_csv('data/Canadian_climate_history.csv')
    #counting the number of NaN values in a column 
    total_NaN = canada_climate.isna().sum()
    #get unique values for years in dataset 
    dates = canada_climate['LOCAL_DATE'].unique() 
    all_years = []
    for date in dates: 
        if date[7:11] not in all_years: 
            all_years.append(date[7:11])
    # filter years from 1990-2019
    all_years = all_years[len(all_years) - 31: len(all_years) - 1]

    #get columns for specific cities, filter out moncton, saskatoon and whitehorse to their abundance of null values 
    mean_columns = []
    total_columns = []
    for column in canada_climate.columns: 
        if 'DATE' not in column and 'MONCTON' not in column and 'SASKATOON' not in column and 'WHITEHORSE' not in column and 'MEAN' in column:
            mean_columns.append(column)
        if 'DATE' not in column and 'MONCTON' not in column and 'SASKATOON' not in column and 'WHITEHORSE' not in column and 'TOTAL' in column:
            total_columns.append(column)
    
    all_series = []    
    # get all rows for a specific year, then get averages for each city, then get total averages 
    for year in all_years: 
        new_series = generate_new_series(canada_climate, year, mean_columns, total_columns)
        all_series.append(new_series)

    final_df = pd.DataFrame(all_series)
    final_df['Year'] = pd.to_numeric(final_df.Year)
    return final_df


def generate_new_series(full_dataset: pd.DataFrame, year: str, mean_columns: [str], total_columns: [str]) -> pd.Series:
    filtered_dataset = full_dataset[full_dataset['LOCAL_DATE'].str[7:11] == year]
    total_NaN = filtered_dataset.isna().sum()
    # create averages for the city for the full year 
    average_data = filtered_dataset[mean_columns].mean() 
    total_data = filtered_dataset[total_columns].sum()

    average_temperature = average_data.mean() 
    total_precipitation = total_data.sum() 

    data_for_series = {'Year': year, 'Avg_temp': average_temperature, 'Total_precip': total_precipitation}
    new_series = pd.Series(data=data_for_series, index=['Year', 'Avg_temp', 'Total_precip'])
    return new_series



#generate_annual_climate_dataframe()