import pandas as pd 
import matplotlib.pyplot as plt 

Canada_emissions = pd.read_csv('data/EN_GHG_Econ_Can_Prov_Terr.csv')

Total_emissions_pyear_Canada = Canada_emissions[(Canada_emissions['Index'] == 0) & (Canada_emissions['Region'] == 'Canada')]

Total_emissions_pyear_Canada['CO2eq'] = pd.to_numeric(Total_emissions_pyear_Canada.CO2eq)

print(Total_emissions_pyear_Canada.head)

def get_total_emissions_annual() -> pd.DataFrame: 
    all_years = Total_emissions_pyear_Canada['Year'].unique() 
    all_series = []
    for year in all_years: 
        this_year_section = Total_emissions_pyear_Canada[Total_emissions_pyear_Canada['Year'] == year]
        total_emissions = this_year_section['CO2eq'].sum() 
        series_data = {"Year": year, "Total_emissions": total_emissions}
        new_series = pd.Series(series_data)
        all_series.append(new_series)
    total_emissions_consolidated = pd.DataFrame(all_series)
    print(total_emissions_consolidated)
    return total_emissions_consolidated




Total_by_source_Canada = Canada_emissions[(Canada_emissions['Region'] == 'Canada') & (Canada_emissions['Total'] == 'y') & (Canada_emissions['Sector'].isnull())]

Total_by_source_2021 = Total_by_source_Canada[(Total_by_source_Canada['Year'] == 2021) & (Total_by_source_Canada['Source'] != "National Inventory Total")]
Total_by_source_2021['CO2eq'] = pd.to_numeric(Total_by_source_2021['CO2eq'])

Total_emissions_provincial = Canada_emissions[(Canada_emissions['Region'] != 'Canada') & (Canada_emissions['Total'] == 'y') & (Canada_emissions['Sector'].isnull())]

Filtered_provincial = Total_emissions_provincial[Total_emissions_provincial['CO2eq'] != 'x']

Filtered_provincial['CO2eq'] = pd.to_numeric(Filtered_provincial['CO2eq']) 

Cut_Out = Canada_emissions[Canada_emissions['CO2eq'] == 'x']

Largest_provincial_emission = Total_emissions_provincial.loc[int(Total_emissions_provincial[Total_emissions_provincial['CO2eq'] != 'x'].idxmax()['CO2eq'] - 1)]

#print(Total_emissions_provincial.head)
Unique_units = Canada_emissions['Unit'].unique()

All_sources = Canada_emissions['Source'].unique()

plt.xlabel('year')
plt.ylabel('emissions')
plt.scatter(Total_emissions_pyear_Canada.Year, Total_emissions_pyear_Canada.CO2eq, color='red')
plt.show()

plt.bar(Total_by_source_2021['Source'], Total_by_source_2021['CO2eq'])
plt.xlabel('Indusry')
plt.ylabel('Emissions')
plt.show()

plt.xlabel('Year')
plt.ylabel('Emissions')
all_regions = Filtered_provincial['Region'].unique()
for region in all_regions: 
    new_df = Filtered_provincial[(Filtered_provincial['Region'] == region) & (Filtered_provincial['Index'] == 0)]
    print(new_df)
    plt.plot(new_df['Year'], new_df['CO2eq'], '-o', label = region)

plt.legend()
plt.show()

