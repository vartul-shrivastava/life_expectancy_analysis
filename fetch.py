
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