import pandas as pd 
import matplotlib.pyplot as plt 

# data is from 2008 - 2022
# annual energy generation in canada by type, producer, and year measured in megawatt hours 

def generate_annual_energy_dataset() -> pd.DataFrame:
    canada_energy = pd.read_csv('data/canada_energy.csv')
    # get list of all generation types 
    generation_types = canada_energy['generation_type'].unique() 
    # get list of all years in dataset 
    all_dates = canada_energy['date'].unique()
    all_years = []
    for date in all_dates: 
        year = date[:4]
        if year not in all_years: 
            all_years.append(year)
    
    # get list of all production types 
    production_types = canada_energy['producer'].unique()
    # create new dataset to hold totals 
    list_of_series = []
    # loop and combine all to sum provinces with same years, generation_types and production types 
    for year in all_years: 
        for generation_type in generation_types: 
            new_series = get_series_of_specific_combination(canada_energy, year, generation_type)
            list_of_series.append(new_series)
    
    total_annual_energy = pd.DataFrame(list_of_series)
    # create new dataframe consolidating each year into one series
    new_list_of_series = []

    for year in all_years: 
        new_list_of_series.append(get_consolidated_series(total_annual_energy, year, generation_types))

    consolidated_annual_energy = pd.DataFrame(new_list_of_series)
    consolidated_annual_energy['Year'] = pd.to_numeric(consolidated_annual_energy.Year)
    return consolidated_annual_energy

def get_series_of_specific_combination(full_dataset: pd.DataFrame, year: str, generation_type: str) -> pd.Series: 
    filtered_dataset = full_dataset[(full_dataset['generation_type'] == generation_type) & (full_dataset['date'].str[0:4] == year)]
    total_energy_generation = filtered_dataset.sum()
    data_for_series = {'Year': year, 'Generation_type': generation_type, "Megawatt_hours": total_energy_generation['megawatt_hours']}
    new_series = pd.Series(data=data_for_series, index=["Year", "Generation_type", "Megawatt_hours"])
    return new_series

def get_consolidated_series(total_annual_energy: pd.DataFrame, year: str, generation_types: list[str]) -> pd.Series:
    this_year_data = total_annual_energy[total_annual_energy['Year'] == year]
    data_for_series = {"Year": year}
    for type in generation_types: 
        this_generation_type = this_year_data[this_year_data['Generation_type'] == type]
        total = this_generation_type['Megawatt_hours'].sum()
        data_for_series[type] = total

    new_series = pd.Series(data=data_for_series)
    return new_series 

#generate_annual_energy_dataset()

