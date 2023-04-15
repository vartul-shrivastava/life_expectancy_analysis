import pandas as pd
from xgboost import XGBRegressor
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
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train the model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb.predict(X_test)

# Evaluate the model
print('Mean squared error:', mean_squared_error(y_test, y_pred))
print('R-squared score:', r2_score(y_test, y_pred))

# Export the model
import joblib
joblib.dump(xgb, 'life_expectancy_xgb_model.pkl')
