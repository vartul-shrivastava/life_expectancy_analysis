
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# Read data from 'backup.csv' and store it in a Pandas DataFrame
data = pd.read_csv('backup.csv')

# Set Streamlit page configuration settings
st.set_page_config(
    page_title="CS312 | Longevity Potential Analysis on World Bank Parameters",
    page_icon="./fav.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS to hide the footer element
hide_footer_css = """
<style>
footer {
    visibility: hidden;
}
</style>
"""

# Render the CSS using Streamlit's markdown function
st.markdown(hide_footer_css, unsafe_allow_html=True)

# Define function to display homepage
def homepage():
    # Add title and description
    st.title("Longevity Potential Analysis on World Bank Parameters")
    st.write("Welcome to the Longevity Potential Analysis webpage! This project aims to analyze the Longevity Potential of countries around the world based on various World Bank parameters.")

    # Create choropleth map
    fig = px.choropleth(data_frame=data,
                        animation_frame='year',
                        locations='country',
                        height=700,
                        locationmode='country names',
                        color='life_expectancy',
                        range_color=[20, 90])

    # Customize layout
    fig.update_layout(geo=dict(showframe=False,
                               showcoastlines=False,
                               bgcolor='rgba(0,0,0,0)',
                               projection_type='equirectangular'))

    # Display figure
    st.plotly_chart(fig, use_container_width=True, height=600)

    # Add additional description
    st.write("Using data from the World Bank, we have created interactive graphs that allow you to explore the relationship between Longevity Potential and other factors such as GDP, healthcare spending, and education.")
    st.write("To get started, choose a parameter from the sidebar on the left and select a country or region from the dropdown menu.")
    st.write("We hope that this project will help you gain a better understanding of the factors that contribute to Longevity Potential around the world.")


def about():
    st.title("About Us")
    
    st.markdown("This project is created by group of students at IIITV-ICD")
    st.markdown('''
    1. Vartul Shrivastava
    2. Perin Mangukiya
    3. Prafulla Patil
    4. Suyash Rajput
    5. Yashesh Bhavsar
    6. Jaideep Panchal
    ''')
    st.markdown("""
        This project is a web application that allows users to select various Longevity Potential graphs using Plotly from a dropdown. 
        The graphs display Longevity Potential data for different countries over time. The application was built using the Streamlit library 
        for Python.
    """)
    
    st.write("")

def gender():

    st.title("Using Gender as Proxy for Longevity Potential")
    st.write()

    data_f = data[data['year'] == 2020]
    # Count the number of countries in each development status category
    country_counts = data_f['development_status'].value_counts()
    # Create pie chart
    o1, o2 = st.columns([1,1])
    with o1:
        st.markdown("""
        By analyzing the trends and patterns in Longevity Potential curves over time, we can gain insights into major world events and their impact on global health. 

        Similarly, the sharp decline in Longevity Potential during the AIDS epidemic in the 1980s and 1990s was a reflection of the devastating impact of the disease on communities around the world. More recently, the COVID-19 pandemic has caused a significant increase in mortality rates, particularly among older adults and those with underlying health conditions.
        Globally: Longevity Potential has increased by more than 20 years between 1960 and 2020, from an average of 52.6 years in 1960 to 73.2 years in 2020. This is largely due to improvements in healthcare, sanitation, nutrition, and living standards.
        """)
    with o2:
        fig = px.pie(values=country_counts, names=country_counts.index, title='Countries by Development Status latest by 2020',height = 400)
        st.plotly_chart(fig)

    st.markdown("Use the dropdown to compare Longevity Potential of various countries:")
    # Create dropdown for country selection
    countries = st.multiselect('Select countries', data['country'].unique(), default=['India','China','Pakistan','Bangladesh'])
    filtered_data = data[data['country'].isin(countries)]

    col1, col2 = st.columns([1, 3])

    # In the first column, display the dynamic table
    with col1:
        if not filtered_data.empty:
            st.write('Data used in graph:')
            st.dataframe(filtered_data[['country', 'year', 'life_expectancy']],use_container_width=True)
        else:
            st.write('No data to display.')

    # In the second column, display the Plotly graph
    with col2:
        fig = px.line(filtered_data, x='year', y='life_expectancy', color='country', title='Longevity Potential Comparison')
        st.plotly_chart(fig,use_container_width=True)

    #graph 1
    st.markdown("The reason for the greater Longevity Potential of females as compared to males can be attributed to multiple factors, such as biology, lifestyle, and societal factors. Biological factors include differences in hormonal makeup, genetics, and immune systems, with females generally having a stronger immune system than males. Lifestyle factors such as diet, exercise, and smoking can also have an impact on Longevity Potential, with women typically having healthier lifestyle choices. Additionally, societal factors such as occupation, access to healthcare, and cultural norms can also influence Longevity Potential. Overall, while the exact reasons for the difference in Longevity Potential between males and females may vary depending on the context, it is clear that there are a multitude of factors that contribute to this difference.")
    kol1, kol2 = st.columns([1, 2])
    with kol2:
        grouped_data = data.groupby(['country', 'year']).mean().reset_index()
        d1 = px.line(grouped_data, x='year', y=['male_life_expectancy', 'female_life_expectancy'], title='Year by Year Longevity Potential Comparison (Male vs Female)', animation_frame='country', range_y=[data['life_expectancy'].min(), data['life_expectancy'].max()], color_discrete_map={'male_life_expectancy': 'blue', 'female_life_expectancy': 'red'})
        d1.update_layout(
            legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
            ))
        st.plotly_chart(d1,use_container_width=True)
    
    with kol1:
        st.markdown('As it will be overwhelming to show dataset from 1960 to 2020. Here is the glimpse of 2020 :)')
        kol2_data = grouped_data[grouped_data['year'] == 2000].reset_index()
        st.dataframe(kol2_data[['country','male_life_expectancy', 'female_life_expectancy']],use_container_width=True)

    st.markdown("""
    ##### __Nation-wise Inferences from the above graph__
    - South Africa: During the height of the HIV/AIDS epidemic in the 1990s and 2000s, life expectancies in South Africa declined significantly, particularly among women. At its lowest point in 2005, Longevity Potential in South Africa was just 52 years.
    - Zimbabwe has experienced several economic crises over the past few decades, resulting in a decline in Longevity Potential. For example, during the economic crisis in the early 2000s, life expectancies in the country dropped from around 60 years to just 44 years.
    - In beginning of 2011, Syria descended into a brutal civil war that has lasted for more than a decade. The war has had a devastating impact on the country's population, with hundreds of thousands of people killed and millions displaced. The conflict has also had a significant impact on life expectancies in the country.
    - Bangladesh's Longevity Potential decreased in 1972 due to a devastating famine that occurred in the country from 1971 to 1974. The famine was a result of a combination of factors, including natural disasters such as floods and cyclones, political instability following the country's independence from Pakistan in 1971, and economic difficulties resulting from the war.
    - Afghanistan had the lowest Longevity Potential in 1984 due to a combination of factors, including ongoing conflict, political instability, and limited access to basic healthcare and nutrition. And as they were constantly in conflict with USSR.
    - In 1970, Equatorial Guinea gained independence from Spain, but political instability followed, with a series of coups and political repression. The combination of political instability, economic underdevelopment, and poor healthcare infrastructure contributed to low Longevity Potential in Equatorial Guinea during this period. 
    - Cholera epidemic in Peru in 1970 caused Longevity Potential dip.
    - Gibraltar is a unique case because it has a small population and is heavily influenced by the economy and healthcare system of neighboring Spain, which has a relatively high Longevity Potential.
    - The fragmentation of Soviet Union in 1990s caused low Longevity Potential in Russia for the years upcoming. And due to constant poor diet in the country, its Longevity Potential remained stagnant especially for the male counterparts.
    """)

    # Sidebar for year selection
    year = st.selectbox('Select a year:', options=data['year'].unique())

    # Filter data based on year selection
    life_expectancy = pd.melt(data[data['year'] == year], id_vars=['country', 'development_status'], value_vars=['male_life_expectancy', 'female_life_expectancy'], var_name='gender', value_name='life_expectancy')

    # Plot the data
    fig = px.box(life_expectancy, height=600, template='plotly_dark', y='development_status', x='life_expectancy', color='gender', hover_name='country')
    fig.update_layout(title=f'Comparison of Male and Female Longevity Potential by Development Status ({year})', xaxis_title='Longevity Potential', yaxis_title='Development Status')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
        The dispersion of male and female Longevity Potential can vary across different income groups of countries. In developed countries, the difference between male and female Longevity Potential is generally smaller, with both genders having higher life expectancies compared to developing countries. This is attributed to better access to healthcare facilities, advanced medical technologies, and higher standards of living in developed countries.

        In developing countries, the dispersion between male and female Longevity Potential can be significant, with females generally having higher life expectancies compared to males. This can be attributed to factors such as better access to healthcare and education for females in some developing countries, as well as cultural and social practices that favor female health and wellbeing.

        In lower middle-income countries, the gap between male and female Longevity Potential can be larger compared to developing countries, but smaller compared to low-income countries. This can be attributed to a lack of access to healthcare facilities and education, limited access to clean water and sanitation, and higher rates of poverty and malnutrition in lower middle-income countries.

        In low-income countries, the difference between male and female Longevity Potential is generally the largest, with females having significantly higher life expectancies compared to males. This can be attributed to a range of factors including limited access to healthcare facilities, higher rates of poverty and malnutrition, limited access to education and information, and gender inequalities that affect women's health outcomes.
        Gender differences: There has been a significant reduction in the gender gap in Longevity Potential over the past 60 years. In 1960, women generally lived longer than men by about five years on average. However, by 2020, this gap had narrowed to around two years.
        """)

    # Load the data from a CSV file
    data_gini = pd.read_csv(r".\World in Data CSVs\gini.csv")

    # Create a list of countries to display in the chart
    countries = ['USA', 'DEU', 'SWE']

    # Filter the data to only include the selected countries
    filtered_data = data_gini[data_gini['Code'].isin(countries)]

    # Create a line chart using Plotly Express
    chart = px.line(filtered_data, x='Year', y='Gini coefficients for lifetime inequality (Peltzman (2009))', color='Country')

    # Set the chart title and axis labels
    chart.update_layout(title='GINI Index Comparison of Germany, Sweden and USA', xaxis_title='Year', yaxis_title='Gini Index')

    # Display the chart in Streamlit
    st.plotly_chart(chart,use_container_width=True)

    st.markdown("The Gini index ranges from 0 to 1, with 0 indicating perfect equality (i.e., everyone has the same income) and 1 indicating perfect inequality (i.e., one person has all the income). The inequality in years of life between people within the same country can be measured in the same way that we measure, for example, the inequality in the distribution of incomes. The idea is to estimate the extent to which a small share of a country’s population concentrates a large ‘stock of health’, hence living much longer than most of the population in the same country. The following visualization presents estimates of the inequality of lifetimes as measured by the Gini coefficient. A high Gini coefficient here means a very unequal distribution of years of life – that is, large within-country inequalities of the number of years that people live.")

def sanitation():
    st.title("Sanitation and Longevity Potential")
    col1, col2 = st.columns([1, 3])
    st.markdown("""
    Sanitation and Longevity Potential are two critical factors that affect the health of individuals and populations. The provision of adequate sanitation facilities can help prevent the spread of infectious diseases, which can have a significant impact on Longevity Potential.
    """)
    cmx = px.imshow(data[['life_expectancy','sanitation_mortality_rate','physicians']].corr(),text_auto=True, width=400)

    cmx.update_layout(coloraxis_colorbar=dict(x=.4, y=1.2, len=0.8, yanchor='top', orientation='h'))
    st.plotly_chart(cmx, use_container_width=True)
    st.markdown('''
    This can be seen in the Sanitation and Longevity Potential plot, which visualizes the correlation between sanitation mortality rate, Longevity Potential, and physicians. The plot shows that countries with higher sanitation mortality rates tend to have lower life expectancies, highlighting the importance of access to clean water and adequate sanitation facilities.

    ''')
    # Define the ranges for sanitation mortality rate and Longevity Potential
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
    s1.update_layout(title='<b>Sankey Diagram</b> of Sanitation Mortality Rate <i>(left)</i> vs Longevity Potential <i>(right)</i>',    xaxis_title = "X-Axis Label",
    yaxis_title = "Y-Axis Label")
    st.plotly_chart(s1,use_container_width=True)
    st.markdown('''
    The Sankey diagram in the plot also demonstrates the relationship between sanitation mortality rate and Longevity Potential by grouping countries into bins based on their sanitation mortality rate and Longevity Potential. The diagram illustrates the flow of countries from one bin to another based on their sanitation mortality rate and Longevity Potential, providing a visual representation of the relationship between these factors.
    ''')

    s2 = px.scatter(data, x='life_expectancy', y='sanitation_mortality_rate', 
                 color='development_status',
                 hover_name='country', log_y=True, template='plotly_dark',
                 labels={'life_expectancy': 'Longevity Potential',
                         'sanitation_mortality_rate': 'Sanitation Mortality Rate',
                         'urban_population': 'Urban Population'},
                 facet_col='development_status')

    s2.update_layout(xaxis_range=[50, data['life_expectancy'].max()],
                  legend=dict(title='Development Status'),
                  font=dict(family='Arial', size=12),
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  title=dict(text='<b>Sanitation Mortality Rate<b> vs Longevity Potential by Development Status',
                             font=dict(size=20)),
                  xaxis=dict(title='Longevity Potential', 
                             titlefont=dict(family='Arial', size=16),
                             tickfont=dict(family='Arial', size=14)),
                  yaxis=dict(title='Sanitation Mortality Rate',
                             titlefont=dict(family='Arial', size=16),
                             tickfont=dict(family='Arial', size=14)))
    st.plotly_chart(s2,use_container_width=True)

    tempDF = data.dropna(subset=['physicians'])
    fig = px.scatter(tempDF,template='plotly_dark', x='healthcare_spending', y='life_expectancy',
                    color='development_status', hover_name='country', size='physicians',
                    log_x=True, title="Analyzing <b>Healthcare Spending</b> and Longevity Potential")
    fig.update_yaxes(range=[35, 90])
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('''Furthermore, the scatter plot in the Sanitation and Longevity Potential plot shows that Longevity Potential tends to increase as sanitation mortality rates decrease, and this trend holds across different levels of development status. This reinforces the importance of sanitation facilities in promoting better health outcomes and improving Longevity Potential.
    ''')
    from plotly.subplots import make_subplots
    # Calculate average number of physicians and Longevity Potential by development status
    physicians_and_life_exp_by_status = data.groupby('development_status').agg({'physicians': 'mean', 'life_expectancy': 'mean'}).reset_index()
    physicians_and_life_exp_by_status = physicians_and_life_exp_by_status.sort_values(by='physicians')

    # Create subplot figure with two vertical subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar traces for physicians and Longevity Potential
    fig.add_trace(go.Bar(x=physicians_and_life_exp_by_status['development_status'], y=physicians_and_life_exp_by_status['physicians'],
                        name='Number of Physicians'), secondary_y=False)
    fig.add_trace(go.Line(x=physicians_and_life_exp_by_status['development_status'], y=physicians_and_life_exp_by_status['life_expectancy'],
                        name='Longevity Potential'), secondary_y=True)

    # Update layout and axis titles
    fig.update_layout(title='Average Number of Physicians and Longevity Potential by Development Status', 
                    xaxis_title='Development Status', yaxis_title='Number of Physicians',template='plotly_dark'  ,
                    yaxis2_title='Longevity Potential')

    # Show the plot
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('''
    In conclusion, the Sanitation and Longevity Potential plot emphasizes the critical role that sanitation plays in promoting better health outcomes and improving Longevity Potential. The plot demonstrates that access to clean water and sanitation facilities is essential for maintaining good health and wellbeing, and highlights the need for continued efforts to improve sanitation infrastructure globally.
    ''')

def carbon_emissions():
    st.title('GDP Per Capita and Carbon Emission influencing Longevity Potential')
    carbon1, carbon2 = st.columns([0.8,1])
    with carbon1:
        cmx = px.imshow(data[['life_expectancy','GDP_per_capita','carbon_emissions']].corr(),text_auto=True, width=400)
        cmx.update_layout(coloraxis_colorbar=dict(x=.4, y=1.2, len=0.8, yanchor='top', orientation='h'))
        st.plotly_chart(cmx,use_container_width=True)
        
    with carbon2:
        st.markdown("""There is a positive correlation between both GDP per capita and carbon emissions and Longevity Potential, which means that as GDP per capita and carbon emissions increase, so does Longevity Potential. The correlation coefficient between Longevity Potential and GDP per capita is +0.54, while the correlation coefficient between Longevity Potential and carbon emissions is +0.53. These coefficients indicate a moderate positive correlation between these variables and Longevity Potential.
        """)
        st.markdown("""
        The reason why there is a positive correlation between GDP per capita and Longevity Potential is that higher economic output typically leads to better access to healthcare, education, and sanitation, which can improve overall health and longevity. Similarly, the positive correlation between carbon emissions and Longevity Potential can be explained by the fact that carbon emissions are often associated with industrialization and economic development, which can lead to improved living conditions and better access to healthcare.
        """)

    st.markdown("""
        This data shows a clear relationship between a country's level of development and its GDP per capita. High income countries, which are typically more developed, have a much higher mean GDP per capita than low income countries. There is also a gradual increase in mean GDP per capita as countries move up the development ladder from low income to high income.

        It's worth noting that GDP per capita is just one measure of a country's economic output, and there are many other factors that contribute to a country's overall development and well-being. However, GDP per capita can be a useful indicator of a country's economic health and can provide insights into the economic disparities between different countries.
        """)
    
    mean_gdp = data.groupby('development_status')['GDP_per_capita'].mean().reset_index().sort_values(by='GDP_per_capita')
    median_gdp = data.groupby('development_status')['GDP_per_capita'].median().reset_index().sort_values(by='GDP_per_capita')

    # Create grouped bar chart with mean and median bars
    fig = px.bar(data_frame=data, x='development_status', y='GDP_per_capita', log_y=True, 
                color='development_status', labels={'GDP_per_capita':'GDP per capita'},
                title='Mean and Median GDP per capita by development status')

    # Add mean bars
    import plotly.graph_objs as go

# Calculate mean and median GDP per capita by development status
    gdp_stats = data.groupby('development_status')['GDP_per_capita'].agg(['mean', 'median']).reset_index()
    gdp_stats = gdp_stats.sort_values(by='median')

# Create grouped bar chart
    fig = go.Figure(data=[
    go.Bar(name='Mean', x=gdp_stats['development_status'], y=gdp_stats['mean'], text=gdp_stats['mean'],
           marker_color='lightskyblue'),
    go.Bar(name='Median', x=gdp_stats['development_status'], y=gdp_stats['median'], text=gdp_stats['median'],
           marker_color='gold')
    ])

    fig.update_layout(
    xaxis=dict(title='Development status'),
    yaxis=dict(title='GDP per capita', type='log'),
    barmode='group',
    title='Mean and Median GDP per capita by development status'
    )

    st.plotly_chart(fig, use_container_width=True)
    year = st.selectbox('Select a year:', options=data['year'].unique())
    tempDF = data[data['year'] == year]
    mean_value = data['schooling'].mean()
    data['schooling'] = data['schooling'].fillna(value=mean_value)
    c1 = px.scatter(tempDF, x="carbon_emissions", y="life_expectancy", hover_data=['country'],color="development_status", template='plotly_dark',
                 title="Relationship Between <b>Longevity Potential and Carbon Emissions</b> Per Capita by Development Status", log_x=True, width=1100,
                 marginal_y='histogram')
    
    st.plotly_chart(c1, use_container_width=True)

    tempDF = data[data['year'] > 2000]
    mean_value = data['schooling'].mean()
    data['schooling'] = data['schooling'].fillna(value=mean_value)
    c1 = px.scatter(tempDF, x="carbon_emissions", y="life_expectancy", hover_data=['country'],color="development_status", template='plotly_dark',
                 title="Cummulative Graph After 2000 | Relationship Between <b>Longevity Potential and Carbon Emissions</b> Per Capita by Development Status", log_x=True, width=1100,
                 marginal_y='histogram')
    
    st.plotly_chart(c1, use_container_width=True)

    st.markdown("""
    Low income countries have the lowest mean and median GDP per capita among all categories. This indicates that these countries have a relatively low level of economic development and are likely to face significant challenges in improving the standard of living of their citizens.

    Low-middle income countries have a higher mean and median GDP per capita than low income countries but lower than developing and developed countries. This suggests that these countries have made some progress in economic development, but they still have a long way to go to catch up with more developed economies.

    Developing countries have a higher mean and median GDP per capita than low-middle income countries, indicating that these countries have achieved a higher level of economic development. However, the gap between developing and developed countries remains significant, and many challenges still need to be addressed, such as poverty, inequality, and access to basic services.

    Developed countries have the highest mean and median GDP per capita among all categories, indicating that they have achieved a high level of economic development and prosperity. However, it is worth noting that even within developed countries, there can be significant disparities in income and wealth, and issues such as income inequality and social exclusion remain significant challenges in many developed economies.
    """)

    h1 = px.scatter(data, x='year',marginal_y= 'box' ,y='GDP_per_capita',log_y=True ,color='development_status',template='plotly_dark', hover_data=['country'], width=1100)
    st.plotly_chart(h1,use_container_width=True)
    pie1,  pie2 = st.columns([1,1])
    with pie1:
        healthcare_data = data[data['year'] == 2000]
        fig = px.pie(healthcare_data, values='healthcare_spending', names='development_status', title='2000 | Healthcare Spending by Development Status')
        st.plotly_chart(fig, use_container_width=True)
    with pie2:
        healthcare_data = data[data['year'] == 2019]
        fig = px.pie(healthcare_data, values='healthcare_spending', names='development_status', title='2020 | Healthcare Spending by Development Status')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
    The chart can be used to make inferences about the distribution of healthcare spending across different development statuses. For example, we can see from the chart that High income countries spend the most on healthcare, followed by Upper middle income countries, and then Lower middle income countries. Low income countries spend the least on healthcare. This information can be useful in identifying areas that may require additional support or resources for healthcare development.
    ''')
   
def obesity_prevalence():
        st.title('Obesity Prevalence and Longevity Potential')
        o1, o0, o2 = st.columns([1.2,0.1,1])
        o1.markdown("""
        <style>
        div.stColumn:first-child {
            margin-right: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True)
        countries = st.multiselect('Select countries', data['country'].unique(), default=['United States','Japan','India','Pakistan'])
        filtered_data = data[data['country'].isin(countries)]
        fig = px.line(filtered_data, x='year', y='obesity_prevalence', color='country', title='Obesity Prevalence by Country')
        st.plotly_chart(fig,use_container_width=True)
        with o1:
            fig = px.histogram(data, x='obesity_prevalence', histnorm='density', color='development_status', marginal='rug')
            fig.update_layout(
            legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
            ))
            st.plotly_chart(fig)
        with o0:
            st.markdown("     ")
        with o2:
            cmx = px.imshow(data[['life_expectancy','GDP_per_capita','obesity_prevalence']].corr(),text_auto=True, width=400)
            cmx.update_layout(coloraxis_colorbar=dict(x=.4, y=1.2, len=0.8, yanchor='top', orientation='h'))
            st.plotly_chart(cmx)

        st.markdown("""
        There is a positive correlation of 0.72 between obesity prevalence and Longevity Potential, indicating that there is a moderate relationship between the two variables. This means that as obesity prevalence increases, so does Longevity Potential. 
        
        This may seem counterintuitive, but it could be due to the fact that some of the factors that lead to higher obesity prevalence, such as access to better healthcare and a higher standard of living, may also lead to higher Longevity Potential.""")

        df1 = data.copy()
        df1 = df1.dropna(subset=['obesity_prevalence'])
        fig = px.scatter(df1,template='plotly_dark', x='GDP_per_capita', y='life_expectancy', size='obesity_prevalence',
                        color='obesity_prevalence', trendline='ols', hover_name='country',
                        log_x=True, size_max=30, title="Analyzing <b>Obesity Prevalance</b> with GDP per capita and Longevity Potential")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("""
               In terms of differences between development status categories, we can see that obesity prevalence is generally highest in developed countries, followed by developing, lower middle income, and low-income countries, in that order. This trend is consistent across all years in the dataset. The density plots show that obesity prevalence in developed countries is more spread out and has a higher peak compared to other development status categories, indicating that there are more people in developed countries who are severely obese.
        """)




def corr_matrix():
    st.title('Representing Correlations between the dataset')
    data_corr = data.drop(columns=['year', 'Unnamed: 0'])
    corr1 = px.imshow(data_corr.corr(),color_continuous_scale=px.colors.sequential.Viridis,text_auto=True, title='HeatMap : To show correlation in between the <b>relevant features</b>')
    corr1.update_layout(height = 800,width = 1200,)
    st.markdown("""
    Upon analyzing the correlation matrix, it is evident that healthcare spending exhibits a strong positive correlation (0.64) with Longevity Potential. Additionally, GDP per capita (0.54) and carbon emissions (0.53) exhibit considerable correlation with Longevity Potential. These correlations will be further explored and visualized in the graphs to gain more insights.

    Contrary to popular belief, obesity prevalence (0.72) - an obvious negative factor in decreasing Longevity Potential - exhibits a positive correlation with Longevity Potential. This suggests that there may be various other latent factors that influence Longevity Potential. The number of physicians per 1000 people (0.65) also has a positive correlation with Longevity Potential, which highlights the pivotal role of healthcare infrastructure in determining Longevity Potential.  

    """)



    st.plotly_chart(corr1,use_container_width=True)
    st.markdown("""
    Education also plays a significant role in improving Longevity Potential, as evidenced by the strong positive correlation with schooling (0.72). On the other hand, mortality rate due to poor sanitation (per 1000) exhibits a negative but influential correlation (-0.84) with Longevity Potential. This indicates that ensuring basic sanitation facilities can significantly improve Longevity Potential. Furthermore, the percentage of the population migrating to urban areas (0.73) has a greater Longevity Potential compared to the rural population, which exhibits an inverse correlation (-0.73). 
    
    This highlights the importance of urbanization in improving the overall health and well-being of individuals. Lastly, the percentage of the population meeting basic sanitation (0.71) exhibits a positive correlation with Longevity Potential. However, unemployment (0.48), mobile cell subscription (0.50), and GINI Index (-0.36) have little to no effect on Longevity Potential, and therefore, may be dropped from the dataset
    """)

    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Sort the correlation coefficients in descending order
    corr_values = corr_matrix['life_expectancy'].drop('life_expectancy').sort_values(ascending=False)

    # Create a bar chart with the correlation coefficients
    fig = go.Figure(go.Bar(
                x=corr_values.index,
                y=corr_values,
                marker={'color': corr_values.apply(lambda x: 'green' if x >= 0 else 'red')},
                text=corr_values.apply(lambda x: f'{x:.2f}'),
                textposition='outside',
            ))

    # Set the title and axis labels
    fig.update_layout(
                height = 500,
                title=f"Correlation of Longevity Potential with Other Variables",
                xaxis_title="Variable",
                yaxis_title="Correlation Coefficient",
            )

    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

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

def ml_model():
    import streamlit as st
    import pandas as pd
    import pickle
    import numpy as np

    # Load the saved model
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Load the data
    data = pd.read_csv('backup.csv')

    # Remove NaN values
    data = data.dropna()

    # Remove development status
    data = data.drop('development_status', axis=1)

    # Define features and target
    features = ['healthcare_spending', 'GDP_per_capita', 'obesity_prevalence', 'carbon_emissions', 'schooling', 'physicians', 'sanitation_mortality_rate', 'urban_population', 'rural_population', 'sanitation_population_perct', 'unemployment_perct', 'mobile_cell_subs', 'GINI_index']
    target = 'life_expectancy'

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.25, random_state=42)

    # Train the model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Define the input features and their default values
    input_features = ['healthcare_spending', 'GDP_per_capita', 'obesity_prevalence', 
                    'carbon_emissions', 'schooling', 'physicians', 'sanitation_mortality_rate',
                    'urban_population', 'rural_population', 'sanitation_population_perct', 
                    'unemployment_perct', 'mobile_cell_subs', 'GINI_index']

    default_values = {'healthcare_spending': 2000, 'GDP_per_capita': 1000, 
                    'obesity_prevalence': 10, 'carbon_emissions': 1, 'schooling': 50, 
                    'physicians': 1, 'sanitation_mortality_rate': 0.5, 'urban_population': 50, 
                    'rural_population': 50, 'sanitation_population_perct': 50, 'unemployment_perct': 5, 
                    'mobile_cell_subs': 50, 'GINI_index': 40}

    # Define a function to get the user inputs
    def get_input_values():
        input_values = {}
        for feature in input_features:
            value = st.number_input(label=feature, value=default_values[feature])
            input_values[feature] = value
        return input_values

    # Define a function to preprocess the input data
    def preprocess_data(input_values):
        data = pd.DataFrame([input_values])
        data = data[['healthcare_spending', 'GDP_per_capita', 'obesity_prevalence', 
                    'carbon_emissions', 'schooling', 'physicians', 'sanitation_mortality_rate',
                    'urban_population', 'rural_population', 'sanitation_population_perct', 
                    'unemployment_perct', 'mobile_cell_subs', 'GINI_index']]
        return data


    # Set page header
    st.write('# Longevity Potential Prediction ML App')
    st.write('The model developed predicts Longevity Potential based on various indicators such as healthcare spending, GDP per capita, and obesity prevalence. It uses advanced machine learning techniques to make accurate predictions while handling missing data and excluding development status. The model has a high R² score of 0.94, which means that it can explain 94% of the variance in the target variable, making it a very good result for a regression model. This means that the model can accurately predict Longevity Potential for different countries based on their socio-economic and healthcare indicators..')
    # Get the input values from the user
    input_values = get_input_values()
    if st.button("Predict Longevity Potential"):
        # Preprocess the input data
        data = preprocess_data(input_values)
        prediction = rf.predict(data)
        st.write('## Predicted Longevity Potential')
        st.write(f'{np.round(prediction[0], 1)} years')

def schooling():
    import plotly.express as px
    st.title("Schooling in Countries and Longevity Potential")
    st.markdown('The schooling in primary, secondary and higher level education on average shows a good moderately-positive relation of +0.72. Hence, the number of years an individual on average spents in gaining education, helps him to achieve the certain upksilled lifestyle which enables for a sustainable life ahead.')
    s1, s2 = st.columns([2,1])
    with s1:
        avg_schooling = data.groupby('development_status')['schooling'].agg(np.mean).reset_index()

        fig = px.bar(avg_schooling, x="development_status", y="schooling", color="development_status")
        fig.update_layout(title="Average Years of Schooling by Development Status",
                        xaxis_title="Development Status",
                        yaxis_title="Years of Schooling")
        st.plotly_chart(fig)
    with s2:
        cmx = px.imshow(data[['life_expectancy','schooling']].corr(),text_auto=True)
        st.plotly_chart(cmx)
    st.markdown('Here is the cumulative summation of years on y-axis and development status of each country on x-axis. We can clearly see the similar trend of education for ')
    fig = px.scatter(data, x='schooling', y='life_expectancy', color='development_status', width=1000, log_x=True,
                 hover_data=['country', 'year', 'healthcare_spending', 'obesity_prevalence'],
                 title='Relationship between Schooling and Longevity Potential by Income Group')
    st.plotly_chart(fig)
    st.markdown('''
    The graph is a scatter plot that shows the relationship between schooling and Longevity Potential, with each dot representing a country. The x-axis represents the average number of years of schooling, while the y-axis represents the Longevity Potential at birth in years. The color of the dots indicates the development status of the country, with developed countries in blue, developing countries in orange, lower-middle-income countries in green, and low-income countries in red.

    The graph suggests a strong positive correlation between schooling and Longevity Potential, with countries that have higher levels of schooling also having higher life expectancies. Additionally, the graph shows that developed countries generally have higher levels of schooling and Longevity Potential than developing countries and lower-income countries.

    The hover data on the graph provides additional information about each country, including healthcare spending and obesity prevalence, which may be useful in further analyzing the relationship between schooling and Longevity Potential.
    ''')

# Set up navigation
nav = st.sidebar.radio("Navigation", ["Home","Relevant Features of Dataset", "Gender and Longevity Potential","Carbon Emissions and Longevity Potential","Sanitation and Longevity Potential","Schooling and Longevity Potential","Obesity Prevalence and Longevity Potential","About Us","ML Model"])
st.sidebar.image("developed.png", use_column_width=True)
# Show appropriate page based on selection
if nav == "Home":
    homepage()
elif nav == "Relevant Features of Dataset":
    corr_matrix()
elif nav == "ML Model":
    ml_model()
elif nav == "Schooling and Longevity Potential":
    schooling()
elif nav == "About Us":
    about()
elif nav == "Gender and Longevity Potential":
    gender()
elif nav == "Carbon Emissions and Longevity Potential":
    carbon_emissions()
elif nav == "Sanitation and Longevity Potential":
    sanitation()
elif nav == "Obesity Prevalence and Longevity Potential":
    obesity_prevalence()
else:
    pass