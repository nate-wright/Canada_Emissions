import pandas as pd 
import matplotlib.pyplot as plt 

# data is from 1971 - 2023 
# has data on annual births, deaths, immigrants, net emigration, emigrants, returning emigrants, net non permanent residents, and residual deviation 

def generate_population_dataframe() -> pd.DataFrame: 
    total_population = pd.read_csv('data/Population_growth.csv')
    Canada_population = total_population[total_population['GEO'] == 'Canada']
    # get all individual years (1971 - 2023)
    all_date_values = Canada_population['REF_DATE'].unique() 
    all_years = []
    for date in all_date_values:
        year = date[5:]
        if year not in all_years:
            all_years.append(year)
    # get all Components of population growth
    all_components = Canada_population['Components of population growth'].unique() 
    # the dates are split into two years (ex. 1971/1972 and 1972/1973)
    # for 1972 we need to take 1971/1972 and 1972/1973 and average the values out by components of population growth 
    all_series = []
    for year in all_years:
        for component in all_components: 
            this_component_first_half = Canada_population[(Canada_population['Components of population growth'] == component) & (Canada_population['REF_DATE'].str[0:4] == year)]
            this_component_second_half = Canada_population[(Canada_population['Components of population growth'] == component) & (Canada_population['REF_DATE'].str[5:] == year)]
            first_size = this_component_first_half[this_component_first_half.columns[0]].count()
            second_size = this_component_second_half[this_component_second_half.columns[0]].count()

            if first_size == 1 and second_size == 1: 
                first_value = this_component_first_half['VALUE'].iloc[0]
                second_value = this_component_second_half['VALUE'].iloc[0]
                if not pd.isna(first_value) and not pd.isna(second_value):
                    real_value = (first_value + second_value) / 2
                    data_for_series = {'Year': year, 'Components of population growth': component, 'Value': real_value}
                    new_series = pd.Series(data=data_for_series, index=["Year", "Components of population growth", "Value"])
                    all_series.append(new_series)
                elif pd.isna(first_value) and not pd.isna(second_value):
                    data_for_series = {'Year': year, 'Components of population growth': component, 'Value': second_value}
                    new_series = pd.Series(data=data_for_series, index=["Year", "Components of population growth", "Value"])
                    all_series.append(new_series)
                elif pd.isna(second_value) and not pd.isna(first_value):
                    data_for_series = {'Year': year, 'Components of population growth': component, 'Value': first_value}
                    new_series = pd.Series(data=data_for_series, index=["Year", "Components of population growth", "Value"])
                    all_series.append(new_series)
    
            elif first_size == 1: 
                first_value = this_component_first_half['VALUE'].iloc[0]
                if not pd.isna(first_value):
                    data_for_series = {'Year': year, 'Components of population growth': component, 'Value': first_value}
                    new_series = pd.Series(data=data_for_series, index=["Year", "Components of population growth", "Value"])
                    all_series.append(new_series)

            elif second_size == 1: 
                second_value = this_component_second_half['VALUE'].iloc[0]
                if not pd.isna(second_value):
                    data_for_series = {'Year': year, 'Components of population growth': component, 'Value': second_value}
                    new_series = pd.Series(data=data_for_series, index=["Year", "Components of population growth", "Value"])
                    all_series.append(new_series)
    final_df = pd.DataFrame(all_series)

    list_of_series = []
    for year in all_years: 
        list_of_series.append(generate_consolidated_series(final_df, year, all_components))

    final_df_consolidated = pd.DataFrame(list_of_series)
    final_df_consolidated["Year"] = pd.to_numeric(final_df_consolidated.Year)
    return final_df_consolidated


def generate_consolidated_series(final_fd: pd.DataFrame, year: int, all_components: list[str]) -> pd.Series: 
    this_year_data = final_fd[final_fd['Year'] == year]
    data_for_series = {"Year": year}
    for comp in all_components: 
        this_comp = this_year_data[this_year_data['Components of population growth'] == comp]
        data_for_series[comp] = this_comp['Value'].sum() 
    
    new_series = pd.Series(data_for_series)
    return new_series
            




#generate_population_dataframe() 