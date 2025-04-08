import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset 
df = pd.read_csv('real_estate_dataset.csv')


X = df[['Num_Bedrooms', 'Square_Feet']]
y = df['Price']


model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X, y)


joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved using raw data!")
