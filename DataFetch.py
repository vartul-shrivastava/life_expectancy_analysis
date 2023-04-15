import wbdata
indicators = {'SP.DYN.LE00.IN': 'life_expectancy',
              'SP.DYN.LE00.MA.IN' : 'male_life_expectancy',
              'SP.DYN.LE00.FE.IN' : 'female_life_expectancy',
              'SH.XPD.CHEX.PP.CD': 'healthcare_spending',
              'NY.GDP.PCAP.CD': 'GDP_per_capita',
              'SH.STA.OWAD.ZS': 'obesity_prevalence',
              'EN.ATM.CO2E.PC': 'carbon_emissions',
              'SE.TER.ENRR' : 'schooling',
              'SH.MED.PHYS.ZS' : 'physicians',
              'SH.STA.WASH.P5' : 'sanitation_mortality_rate',
              'SP.URB.TOTL.IN.ZS' : 'urban_population',
              'SP.RUR.TOTL.ZS' : 'rural_population',
              'SH.STA.SMSS.ZS' : 'sanitation_population_perct',
              'SL.UEM.TOTL.ZS' : 'unemployment_perct',
              'IT.CEL.SETS.P2' : 'mobile_cell_subs',
              'SI.POV.GINI' : 'GINI_index',
              'SP.POP.TOTL' : 'population'
}       

data = wbdata.get_dataframe(indicators, country='all')
data = data.reset_index()

data.columns = ['country', 'year', 'life_expectancy', 'male_life_expectancy','female_life_expectancy', 'healthcare_spending', 'GDP_per_capita', 'obesity_prevalence', 'carbon_emissions','schooling','physicians','sanitation_mortality_rate','urban_population','rural_population','sanitation_population_perct','unemployment_perct','mobile_cell_subs','GINI_index','total_population']
data = data.sort_values(['country', 'year'])

# Define income group thresholds
income_groups = {'High income': 12736, 
                 'Upper middle income': 4126, 
                 'Lower middle income': 1046, 
                 'Low income': 0}

# Define function to classify countries
def classify_country(row):
    gdp = row['GDP_per_capita']
    if gdp >= income_groups['High income']:
        return 'Developed'
    elif gdp >= income_groups['Upper middle income']:
        return 'Developing'
    elif gdp >= income_groups['Lower middle income']:
        return 'Lower middle income'
    else:
        return 'Low income'

# Apply function to create new column
data['development_status'] = data.apply(classify_country, axis=1)
data['year'] = data['year'].astype('int')

data.to_csv('backup.csv')
