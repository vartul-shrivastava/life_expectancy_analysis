
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
import plotly.express as px
import wbdata
import streamlit.components.v1 as components


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
              'SI.POV.GINI' : 'GINI_index'
}       

data = wbdata.get_dataframe(indicators, country='all')
data = data.reset_index()

data.columns = ['country', 'year', 'life_expectancy', 'male_life_expectancy','female_life_expectancy', 'healthcare_spending', 'GDP_per_capita', 'obesity_prevalence', 'carbon_emissions','schooling','physicians','sanitation_mortality_rate','urban_population','rural_population','sanitation_population_perct','unemployment_perct','mobile_cell_subs','GINI_index']
data = data.sort_values(['country', 'year'])

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
              'SI.POV.GINI' : 'GINI_index'
}       

data = wbdata.get_dataframe(indicators, country='all')
data = data.reset_index()

data.columns = ['country', 'year', 'life_expectancy', 'male_life_expectancy','female_life_expectancy', 'healthcare_spending', 'GDP_per_capita', 'obesity_prevalence', 'carbon_emissions','schooling','physicians','sanitation_mortality_rate','urban_population','rural_population','sanitation_population_perct','unemployment_perct','mobile_cell_subs','GINI_index']
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

data.to_csv('backup.csv')


# ML Model
def ml_model():
    data = pd.read_csv('backup.csv')
    # Clean data by filling missing values with mean
    data = data.fillna(data.mean())
    # Split data into training and test sets
    X = data.drop(['country', 'year', 'life_expectancy', 'male_life_expectancy','female_life_expectancy','development_status'], axis=1)
    y = data['life_expectancy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.drop('Unnamed: 0', axis=1)
    X_test = X_test.drop('Unnamed: 0', axis=1)
    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    num_features = ['healthcare_spending',          
                    'GDP_per_capita',
                    'obesity_prevalence', 'carbon_emissions', 'schooling', 'physicians', 'sanitation_mortality_rate',
                    'urban_population', 'rural_population', 'sanitation_population_perct', 'unemployment_perct',
                    'mobile_cell_subs', 'GINI_index']
    # Fit Random Forest Regressor model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    with open('model.pkl','wb') as file:
        pickle.dump(rf,file)
    st.title('Life Expectancy Prediction')
        # Collect user input
    ml_col1, ml_col2 = st.columns([1,1])
    with ml_col1:
        healthcare_spending = st.number_input('Healthcare spending (as % of GDP)', min_value=0, max_value=100, value=10)
        GDP_per_capita = st.number_input('GDP per capita (in US dollars)', min_value=0, max_value=100000, value=5000)
        obesity_prevalence = st.number_input('Obesity prevalence (as % of population)', min_value=0, max_value=100, value=20)
        carbon_emissions = st.number_input('Carbon emissions (in metric tons per capita)', min_value=0, max_value=30, value=5)
        schooling = st.number_input('Schooling (in years)', min_value=0, max_value=30, value=20)
        physicians = st.number_input('Number of physicians per 1000 population', min_value=0, max_value=10, value=2)
        sanitation_mortality_rate = st.number_input('Sanitation mortality rate (per 1000 population)', min_value=0, max_value=10, value=2)
    with ml_col2:
        urban_population = st.number_input('Urban population (as % of total population)', min_value=0, max_value=100, value=50)
        rural_population = st.number_input('Rural population (as % of total population)', min_value=0, max_value=100, value=50)
        sanitation_population_perct = st.number_input('Sanitation population percent (as % of total population)', min_value=0, max_value=100, value=50)
        unemployment_perct = st.number_input('Unemployment percent (as % of total labor force)', min_value=0, max_value=50, value=10)
        mobile_cell_subs = st.number_input('Mobile cellular subscriptions (per 100 people)', min_value=0, max_value=200, value=50)
        GINI_index = st.number_input('GINI index (measure of income inequality)', min_value=0, max_value=100, value=50)

    submit_button = st.button('Submit')
    if submit_button:
        # Create input DataFrame from user input    
        input_data = pd.DataFrame({
        'healthcare_spending': healthcare_spending,
        'GDP_per_capita': GDP_per_capita,
        'obesity_prevalence': obesity_prevalence,
        'carbon_emissions': carbon_emissions,
        'schooling': schooling,
        'physicians': physicians,
        'sanitation_mortality_rate': sanitation_mortality_rate,
        'urban_population': urban_population,
        'rural_population': rural_population,
        'sanitation_population_perct': sanitation_population_perct,
        'unemployment_perct': unemployment_perct,
        'mobile_cell_subs': mobile_cell_subs,
        'GINI_index': GINI_index
        }, index=[0])

        # Make predictions
        prediction = rf.predict(input_data)[0]

        # Display prediction
        st.write('Predicted life expectancy:', round(prediction, 2))