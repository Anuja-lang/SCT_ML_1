import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Define file path
file_path = r'C:\Users\LENOVO\Desktop\machine\Housing.csv'

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found. Please ensure 'Housing.csv' is in the 'data/' folder.")
    print("If the file is in the project root, change the path to 'Housing.csv'.")
    exit(1)

try:
    # Load the dataset
    data = pd.read_csv(r'C:\Users\LENOVO\Desktop\machine\Housing.csv')

    # Verify required columns
    required_columns = ['area', 'bedrooms', 'bathrooms', 'price']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Error: Missing columns in dataset: {missing_columns}")
        print("Ensure you're using the 'Housing Prices Dataset' from Kaggle (yasserh/housing-prices-dataset).")
        exit(1)

    # Check for missing values
    if data[required_columns].isnull().any().any():
        print("Warning: Missing values detected. Filling with mean for numeric columns.")
        data = data.fillna(data[required_columns].mean())

    # Check for non-numeric data
    if not all(data[required_columns].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        print("Error: Non-numeric data found in feature columns. Please clean the dataset.")
        exit(1)

    # Handle outliers by clipping extreme values (e.g., top/bottom 1%)
    for col in ['area', 'price']:
        lower, upper = data[col].quantile([0.01, 0.99])
        data[col] = data[col].clip(lower, upper)

    # Select features and target
    X = data[['area', 'bedrooms', 'bathrooms']]
    y = data['price']

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print("Model Coefficients:")
    print(f"Area (square footage): {model.coef_[0]:.2f} (price per square foot)")
    print(f"Bedrooms: {model.coef_[1]:.2f}")
    print(f"Bathrooms: {model.coef_[2]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"\nMean Squared Error: {mse:,.2f}")
    print(f"R² Score: {r2:.2f}")

    # Example prediction for a house with 2000 sqft, 3 bedrooms, 2 bathrooms
    example_house = pd.DataFrame([[2000, 3, 2]], columns=['area', 'bedrooms', 'bathrooms'])
    predicted_price = model.predict(example_house)
    print(f"\nPredicted price for a 2000 sqft house with 3 bedrooms and 2 bathrooms: ${predicted_price[0]:,.2f}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your dataset and dependencies (pandas, numpy, scikit-learn).")

     