import pandas as pd 
import matplotlib.pyplot as plt 

# get years where canada introduced climate policy 

def generate_policy_dataframe() -> pd.DataFrame: 
    policy = pd.read_excel('data/Renewable policy data.xlsx')
    canada_policy = policy[policy['Country'] == 'Canada']

    # get all years 
    all_years = canada_policy['Year'].unique() 
    list_of_series = [] 
    #create a new dataframe just counting the number of renewable policies by year 
    for year in range(1970, 2022):
        this_year_canada = canada_policy[canada_policy['Year'] == year]
        number_of_policies = len(this_year_canada)
        data_for_series = {'Year': year, 'Policies': number_of_policies}
        new_series = pd.Series(data=data_for_series, index=["Year", "Policies"])
        list_of_series.append(new_series)

    canada_policy_consolidated = pd.DataFrame(list_of_series)
    return canada_policy_consolidated

#generate_policy_dataframe() 
