
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

data['year'] = data['year'].astype('int')


st.set_page_config(page_title="My App", page_icon=":rocket:", layout="wide", initial_sidebar_state="expanded")


def homepage():
    st.title("Life Expectancy Analysis on World Bank Parameters")
    st.write("Welcome to the Life Expectancy Analysis webpage!")
    fig = px.choropleth(data_frame=data, width=2000,
                        animation_frame='year',
                    locations='country', 
                    locationmode='country names', 
                    color='life_expectancy', 
                    range_color=[20, 90],
   )
    #Add some additional layout options
    fig.update_layout(geo=dict(showframe=False, showcoastlines=False,
                            projection_type='equirectangular'))
    
    # Show the figure
    st.plotly_chart(fig,use_container_width=True, height=600)
    st.write("This project aims to analyze the life expectancy of countries around the world based on various World Bank parameters.")
    st.write("Using data from the World Bank, we have created interactive graphs that allow you to explore the relationship between life expectancy and other factors such as GDP, healthcare spending, and education.")
    
    st.write("To get started, choose a parameter from the sidebar on the left and select a country or region from the dropdown menu.")
    
    st.write("We hope that this project will help you gain a better understanding of the factors that contribute to life expectancy around the world.")


def about():
    st.title("About Us")
    
    st.markdown("This project was created by the following team members from Indian Institute of Information Technology Vadodara - International Campus Diu:")
    
    st.markdown("""
    #### Team Members
    - Vartul Shrivastava
    - Yashesh Bhavsar
    - Perin Mangukiya
    - Prafulla Patil
    - Suyash Rajput
    """)
    st.markdown("""
        This project is a web application that allows users to select various life expectancy graphs using Plotly from a dropdown. 
        The graphs display life expectancy data for different countries over time. The application was built using the Streamlit library 
        for Python and deployed using Heroku.
    """)
    
    st.write("")


def gender():

    st.title("Using Gender as Proxy for Life Expectancy")
    st.write()
    st.markdown("Use the dropdown to compare life expectancy of various countries:")
    # Create dropdown for country selection
    countries = st.multiselect('Select countries', data['country'].unique(), default=['India','China'])
    filtered_data = data[data['country'].isin(countries)]

    col1, col2 = st.columns([1, 2])

    # In the first column, display the dynamic table
    with col1:
        if not filtered_data.empty:
            st.write('Data used in graph:')
            st.dataframe(filtered_data[['country', 'year', 'life_expectancy']])
        else:
            st.write('No data to display.')

    # In the second column, display the Plotly graph
    with col2:
        fig = px.line(filtered_data, x='year', y='life_expectancy', color='country', title='Life Expectancy Comparison')
        st.plotly_chart(fig)

    #graph 1
    
    grouped_data = data.groupby(['country', 'year']).mean().reset_index()
    d1 = px.line(grouped_data, x='year', y=['male_life_expectancy', 'female_life_expectancy'], title='Year by Year Life Expectancy Comparison (Male vs Female)', animation_frame='country', range_y=[data['life_expectancy'].min(), data['life_expectancy'].max()], color_discrete_map={'male_life_expectancy': 'blue', 'female_life_expectancy': 'red'})
    st.plotly_chart(d1)

    st.markdown("""
    ##### __Nation-wise Inferences from the above graph__
    - Bangladesh's life expectancy decreased in 1972 due to a devastating famine that occurred in the country from 1971 to 1974. The famine was a result of a combination of factors, including natural disasters such as floods and cyclones, political instability following the country's independence from Pakistan in 1971, and economic difficulties resulting from the war.
    - Afghanistan had the lowest life expectancy in 1984 due to a combination of factors, including ongoing conflict, political instability, and limited access to basic healthcare and nutrition. And as they were constantly in conflict with USSR.
    - n 1970, Equatorial Guinea gained independence from Spain, but political instability followed, with a series of coups and political repression. The combination of political instability, economic underdevelopment, and poor healthcare infrastructure contributed to low life expectancy in Equatorial Guinea during this period. 
    - holera epidemic in Peru in 1970 caused life expectancy dip.
    - The fragmentation of Soviet Union in 1990s caused low life expectancy in Russia for the years upcoming. And due to constant poor diet in the country, its life expectancy remained stagnant especially for the male counterparts.
    """)

    #graph 2
    life_expectancy = pd.melt(data, id_vars=['country', 'development_status'], value_vars=['male_life_expectancy', 'female_life_expectancy'], var_name='gender', value_name='life_expectancy')
    d2 = px.box(life_expectancy, template='plotly_dark', y='development_status', x='life_expectancy', color='gender', hover_name='country')
    d2.update_layout(title='Comparison of Male and Female Life Expectancy by Development Status', xaxis_title='Life Expectancy', yaxis_title='Development Status')
    st.plotly_chart(d2)

def sanitation():
    st.title("Sanitation and Life Expectancy")
    # Define the ranges for sanitation mortality rate and life expectancy
    sanitation_bins = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    life_expectancy_bins = [0, 50, 60, 70, 80, 90, 100]

    # Bin the data
    data['sanitation_bin'] = pd.cut(data['sanitation_mortality_rate'], sanitation_bins, labels=False)
    data['life_expectancy_bin'] = pd.cut(data['life_expectancy'], life_expectancy_bins, labels=False)

    # Count the number of countries in each combination of bins
    counts = data.groupby(['sanitation_bin', 'life_expectancy_bin']).size().reset_index(name='count')

    # Define the labels for the bins
    sanitation_labels = ['<10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-110', '110-120', '>120']
    life_expectancy_labels = ['<50', '50-60', '60-70', '70-80', '80-90', '>90']

    # Create the Sankey diagram
    s1 = go.Figure(data=[go.Sankey(
    node=dict(
      pad=15,
      thickness=20,
      label=sanitation_labels + life_expectancy_labels,

    ),
    link=dict(
      source=counts['sanitation_bin'],
      target=counts['life_expectancy_bin'] + len(sanitation_labels),
      value=counts['count'],
    ))])

    # Add labels to the nodes
    s1.update_layout(title='<b>Sankey Diagram</b> of Sanitation Mortality Rate <i>(left)</i> vs Life Expectancy <i>(right)</i>',    xaxis_title = "X-Axis Label",
    yaxis_title = "Y-Axis Label")
    st.plotly_chart(s1)

    s2 = px.scatter(data, x='life_expectancy', y='sanitation_mortality_rate', 
                 color='development_status',
                 hover_name='country', log_y=True, template='plotly_dark',
                 labels={'life_expectancy': 'Life Expectancy',
                         'sanitation_mortality_rate': 'Sanitation Mortality Rate',
                         'urban_population': 'Urban Population'},
                 facet_col='development_status')

    s2.update_layout(xaxis_range=[50, data['life_expectancy'].max()],
                  legend=dict(title='Development Status'),
                  font=dict(family='Arial', size=12),
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  title=dict(text='<b>Sanitation Mortality Rate<b> vs Life Expectancy by Development Status',
                             font=dict(size=20)),
                  xaxis=dict(title='Life Expectancy', 
                             titlefont=dict(family='Arial', size=16),
                             tickfont=dict(family='Arial', size=14)),
                  yaxis=dict(title='Sanitation Mortality Rate',
                             titlefont=dict(family='Arial', size=16),
                             tickfont=dict(family='Arial', size=14)))
    st.plotly_chart(s2)

def healthcare():
    pass

def carbon_emissions():
    st.title('Carbon Emission influencing Life Expectancy')
    tempDF = data[data['year'] > 2000]
    mean_value = data['schooling'].mean()
    data['schooling'] = data['schooling'].fillna(value=mean_value)
    c1 = px.scatter(tempDF, x="carbon_emissions", y="life_expectancy", color="development_status", template='plotly_dark',
                 title="<i>After 2000</i> | Relationship Between <b>Life Expectancy and Carbon Emissions</b> Per Capita by Development Status", log_x=True, width=1100,
                 marginal_x='histogram', marginal_y='histogram')
    st.plotly_chart(c1)
    
def covid():
    pass

def component():
    imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

    imageUrls = [
        "https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
        "https://images.unsplash.com/photo-1610016302534-6f67f1c968d8?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1075&q=80",
        "https://images.unsplash.com/photo-1516550893923-42d28e5677af?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=872&q=80",
        "https://images.unsplash.com/photo-1541343672885-9be56236302a?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1512470876302-972faa2aa9a4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1528728329032-2972f65dfb3f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1557744813-846c28d0d0db?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1118&q=80",
        "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1595867818082-083862f3d630?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1622214366189-72b19cc61597?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1558180077-09f158c76707?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
        "https://images.unsplash.com/photo-1520106212299-d99c443e4568?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=687&q=80",
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1571317084911-8899d61cc464?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=870&q=80",
        "https://images.unsplash.com/photo-1624704765325-fd4868c9702e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
    ]
    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

    if selectedImageUrl is not None:
        st.image(selectedImageUrl)

def corr_matrix():
    corr1 = px.imshow(data.corr(),text_auto=True, title='HeatMap : To show correlation in between the <b>relevant features</b>')
    corr1.update_layout(height = 800,width = 800,)
    st.plotly_chart(corr1)

# Set up navigation
nav = st.sidebar.radio("Navigation", ["Home","Relevant Features of Dataset", "Gender and Life Expectancy","Carbon Emissions and Life Expectancy","Sanitation and Life Expectancy","Healthcare Expenditure and Life Expectancy","Covid-19 Affecting Life Expectancy","About Us","Component"])

# Show appropriate page based on selection
if nav == "Home":
    homepage()
elif nav == "Relevant Features of Dataset":
    corr_matrix()
elif nav == "About Us":
    about()
elif nav == "Gender and Life Expectancy":
    gender()
elif nav == "Carbon Emissions and Life Expectancy":
    carbon_emissions()
elif nav == "Sanitation and Life Expectancy":
    sanitation()
elif nav == "Healthcare Expenditure and Life Expectancy":
    healthcare()
elif nav == "Covid-19 Affecting Life Expectancy":
    covid()
elif nav == "Component":
    component()
else:
    pass
