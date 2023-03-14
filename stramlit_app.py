
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

data = pd.read_csv('backup.csv')

st.set_page_config(page_title="My App", page_icon=":rocket:", layout="wide", initial_sidebar_state="expanded")


def homepage():
    st.title("Life Expectancy Analysis on World Bank Parameters")
    st.write("Welcome to the Life Expectancy Analysis webpage!.This project aims to analyze the life expectancy of countries around the world based on various World Bank parameters.")
    fig = px.choropleth(data_frame=data, width=2000,
                        animation_frame='year',
                    locations='country', 
                    locationmode='country names', 
                    color='life_expectancy', 
                    range_color=[20, 90], title="Chloropeth Chart to show Life Expectancy around world from 1960 to 2020"
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
    st.markdown("""
    By analyzing the trends and patterns in life expectancy curves over time, we can gain insights into major world events and their impact on global health. 

    Similarly, the sharp decline in life expectancy during the AIDS epidemic in the 1980s and 1990s was a reflection of the devastating impact of the disease on communities around the world. More recently, the COVID-19 pandemic has caused a significant increase in mortality rates, particularly among older adults and those with underlying health conditions.

    """)
    st.markdown("Use the dropdown to compare life expectancy of various countries:")
    # Create dropdown for country selection
    countries = st.multiselect('Select countries', data['country'].unique(), default=['India','China','Pakistan','Bangladesh'])
    filtered_data = data[data['country'].isin(countries)]

    col1, col2 = st.columns([1, 3])

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
    st.markdown("The reason for the greater life expectancy of females as compared to males can be attributed to multiple factors, such as biology, lifestyle, and societal factors. Biological factors include differences in hormonal makeup, genetics, and immune systems, with females generally having a stronger immune system than males. Lifestyle factors such as diet, exercise, and smoking can also have an impact on life expectancy, with women typically having healthier lifestyle choices. Additionally, societal factors such as occupation, access to healthcare, and cultural norms can also influence life expectancy. Overall, while the exact reasons for the difference in life expectancy between males and females may vary depending on the context, it is clear that there are a multitude of factors that contribute to this difference.")
    kol1, kol2 = st.columns([1.5, 2])
    with kol2:
        grouped_data = data.groupby(['country', 'year']).mean().reset_index()

        d1 = px.line(grouped_data, x='year', y=['male_life_expectancy', 'female_life_expectancy'], title='Year by Year Life Expectancy Comparison (Male vs Female)', animation_frame='country', range_y=[data['life_expectancy'].min(), data['life_expectancy'].max()], color_discrete_map={'male_life_expectancy': 'blue', 'female_life_expectancy': 'red'})
        st.plotly_chart(d1)
    
    with kol1:
        st.markdown('As it will be overwhelming to show dataset from 1960 to 2020. Here is the glimpse of 2020 :)')
        kol2_data = grouped_data[grouped_data['year'] == 2000].reset_index()
        st.dataframe(kol2_data[['country','male_life_expectancy', 'female_life_expectancy']])

    st.markdown("""
    ##### __Nation-wise Inferences from the above graph__
    - Bangladesh's life expectancy decreased in 1972 due to a devastating famine that occurred in the country from 1971 to 1974. The famine was a result of a combination of factors, including natural disasters such as floods and cyclones, political instability following the country's independence from Pakistan in 1971, and economic difficulties resulting from the war.
    - Afghanistan had the lowest life expectancy in 1984 due to a combination of factors, including ongoing conflict, political instability, and limited access to basic healthcare and nutrition. And as they were constantly in conflict with USSR.
    - n 1970, Equatorial Guinea gained independence from Spain, but political instability followed, with a series of coups and political repression. The combination of political instability, economic underdevelopment, and poor healthcare infrastructure contributed to low life expectancy in Equatorial Guinea during this period. 
    - holera epidemic in Peru in 1970 caused life expectancy dip.
    - The fragmentation of Soviet Union in 1990s caused low life expectancy in Russia for the years upcoming. And due to constant poor diet in the country, its life expectancy remained stagnant especially for the male counterparts.
    """)

    #graph 2
    ref1, ref2 = st.columns([1,1])
    with ref2:
        life_expectancy = pd.melt(data, id_vars=['country', 'development_status'], value_vars=['male_life_expectancy', 'female_life_expectancy'], var_name='gender', value_name='life_expectancy')
        d2 = px.box(life_expectancy, height=600, template='plotly_dark', y='development_status', x='life_expectancy', color='gender', hover_name='country')
        d2.update_layout(title='Comparison of Male and Female Life Expectancy by Development Status', xaxis_title='Life Expectancy', yaxis_title='Development Status')
        st.plotly_chart(d2)
    
    with ref1:
        st.markdown("""
        The dispersion of male and female life expectancy can vary across different income groups of countries. In developed countries, the difference between male and female life expectancy is generally smaller, with both genders having higher life expectancies compared to developing countries. This is attributed to better access to healthcare facilities, advanced medical technologies, and higher standards of living in developed countries.

        In developing countries, the dispersion between male and female life expectancy can be significant, with females generally having higher life expectancies compared to males. This can be attributed to factors such as better access to healthcare and education for females in some developing countries, as well as cultural and social practices that favor female health and wellbeing.

        In lower middle-income countries, the gap between male and female life expectancy can be larger compared to developing countries, but smaller compared to low-income countries. This can be attributed to a lack of access to healthcare facilities and education, limited access to clean water and sanitation, and higher rates of poverty and malnutrition in lower middle-income countries.

        In low-income countries, the difference between male and female life expectancy is generally the largest, with females having significantly higher life expectancies compared to males. This can be attributed to a range of factors including limited access to healthcare facilities, higher rates of poverty and malnutrition, limited access to education and information, and gender inequalities that affect women's health outcomes.
        """)


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

def corr_matrix():
    st.title('Representing Correlations between the dataset')
    corr1 = px.imshow(data.corr(),text_auto=True, title='HeatMap : To show correlation in between the <b>relevant features</b>')
    corr1.update_layout(height = 800,width = 1200,)
    st.markdown("""
    Upon analyzing the correlation matrix, it is evident that healthcare spending exhibits a strong positive correlation (0.64) with life expectancy. Additionally, GDP per capita (0.54) and carbon emissions (0.53) exhibit considerable correlation with life expectancy. These correlations will be further explored and visualized in the graphs to gain more insights.

    Contrary to popular belief, obesity prevalence (0.72) - an obvious negative factor in decreasing life expectancy - exhibits a positive correlation with life expectancy. This suggests that there may be various other latent factors that influence life expectancy. The number of physicians per 1000 people (0.65) also has a positive correlation with life expectancy, which highlights the pivotal role of healthcare infrastructure in determining life expectancy.  

    """)

    st.plotly_chart(corr1)
    st.markdown("""
    Education also plays a significant role in improving life expectancy, as evidenced by the strong positive correlation with schooling (0.72). On the other hand, mortality rate due to poor sanitation (per 1000) exhibits a negative but influential correlation (-0.84) with life expectancy. This indicates that ensuring basic sanitation facilities can significantly improve life expectancy. Furthermore, the percentage of the population migrating to urban areas (0.73) has a greater life expectancy compared to the rural population, which exhibits an inverse correlation (-0.73). 
    
    This highlights the importance of urbanization in improving the overall health and well-being of individuals. Lastly, the percentage of the population meeting basic sanitation (0.71) exhibits a positive correlation with life expectancy. However, unemployment (0.48), mobile cell subscription (0.50), and GINI Index (-0.36) have little to no effect on life expectancy, and therefore, may be dropped from the dataset
    """)
    
    st.markdown("""
    ##### Here is the summary of indicators analyzed:
    1. `life_expectancy` (SP.DYN.LE00.IN):The average number of years a newborn is expected to live if mortality patterns at the time of its birth remain constant in the future.

    2. `male_life_expectancy` (SP.DYN.LE00.MA.IN):The average number of years a newborn male is expected to live if mortality patterns at the time of his birth remain constant in the future.

    3. `female_life_expectancy` (SP.DYN.LE00.FE.IN):The average number of years a newborn female is expected to live if mortality patterns at the time of her birth remain constant in the future.

    4. `healthcare_spending` (SH.XPD.CHEX.PP.CD)  Total healthcare spending, including both public and private expenditures, expressed as a percentage of GDP.

    4. `GDP_per_capita` (NY.GDP.PCAP.CD):The total economic output of a country divided by its total population.

    5. `obesity_prevalence` (SH.STA.OWAD.ZS): The percentage of a country's adult population with a Body Mass Index (BMI) greater than or equal to 30.

    6. `carbon_emissions` (EN.ATM.CO2E.PC): The amount of carbon dioxide emitted by a country per capita.

    7. `schooling` (SE.TER.ENRR): The percentage of the population of official school age who are enrolled in tertiary education.

    8. `physicians` (SH.MED.PHYS.ZS): The number of physicians (including generalist and specialist medical practitioners) per 1,000 people.

    9. `sanitation_mortality_rate` (SH.STA.WASH.P5) - The number of deaths of children under the age of five per 1,000 live births due to poor sanitation.

    10. `urban_population` (SP.URB.TOTL.IN.ZS) - The percentage of a country's total population that lives in urban areas.

    11. `rural_population` (SP.RUR.TOTL.ZS) - The percentage of a country's total population that lives in rural areas.

    12. `sanitation_population_perct` (SH.STA.SMSS.ZS) - The percentage of the population with access to basic sanitation services.

    13. `unemployment_perct` (SL.UEM.TOTL.ZS) - The percentage of the labor force that is unemployed.

    14. `mobile_cell_subs` (IT.CEL.SETS.P2) - The number of mobile cellular telephone subscriptions per 100 people.

    15. `GINI_index` (SI.POV.GINI) - A measure of income inequality within a country, with values ranging from 0 (perfect equality) to 100 (perfect inequality).
    """)

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
