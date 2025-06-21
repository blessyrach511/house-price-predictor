import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Example dataset
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500, 3000],
    'bhk': [2, 3, 3, 4, 4],
    'location': [1, 2, 1, 2, 3],  # Encoded location
    'price': [50, 75, 100, 130, 150]
})

X = data[['area', 'bhk', 'location']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
