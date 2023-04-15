import pandas as pd
import plotly.express as px
import streamlit as st

# Load data
data = pd.read_csv('backup.csv')

# Sidebar for year selection
year = st.sidebar.selectbox('Select a year:', options=data['year'].unique())

# Filter data based on year selection
life_expectancy = pd.melt(data[data['year'] == year], id_vars=['country', 'development_status'], value_vars=['male_life_expectancy', 'female_life_expectancy'], var_name='gender', value_name='life_expectancy')

# Plot the data
fig = px.box(life_expectancy, height=600, template='plotly_dark', y='development_status', x='life_expectancy', color='gender', hover_name='country')
fig.update_layout(title=f'Comparison of Male and Female Life Expectancy by Development Status ({year})', xaxis_title='Life Expectancy', yaxis_title='Development Status')
st.plotly_chart(fig, use_container_width=True)