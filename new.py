import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
data = pd.read_csv('backup.csv')

# Create dropdown for country selection
countries = st.multiselect('Select countries', data['country'].unique())

# Filter data based on selected countries
filtered_data = data[data['country'].isin(countries)]

# Create Plotly graph
fig = px.line(filtered_data, x='year', y='life_expectancy', color='country', title='Life Expectancy Comparison')
st.plotly_chart(fig)
