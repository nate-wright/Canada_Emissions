import pandas as pd 
import matplotlib.pyplot as plt 

def get_total_emissions_annual() -> pd.DataFrame: 
    Canada_emissions = pd.read_csv('data/EN_GHG_Econ_Can_Prov_Terr.csv')
    Total_emissions_pyear_Canada = Canada_emissions[(Canada_emissions['Index'] == 0) & (Canada_emissions['Region'] == 'Canada')]
    Total_emissions_pyear_Canada['CO2eq'] = pd.to_numeric(Total_emissions_pyear_Canada.CO2eq)
    all_years = Total_emissions_pyear_Canada['Year'].unique() 
    all_series = []
    for year in all_years: 
        this_year_section = Total_emissions_pyear_Canada[Total_emissions_pyear_Canada['Year'] == year]
        total_emissions = this_year_section['CO2eq'].sum() 
        series_data = {"Year": year, "Total_emissions": total_emissions}
        new_series = pd.Series(series_data)
        all_series.append(new_series)
    total_emissions_consolidated = pd.DataFrame(all_series)
    return total_emissions_consolidated