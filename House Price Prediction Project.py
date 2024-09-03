#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nest_asyncio
import uvicorn
from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fastapi.responses import StreamingResponse

nest_asyncio.apply()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
def predict():
    # Load datasets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Set the display option to avoid line breaks
    pd.set_option('display.expand_frame_repr', False)

    # Display the first few rows of the training dataset
    print("Training Data:")
    print(train_df.head())

    # Display the first few rows of the test dataset
    print("Test Data:")
    print(test_df.head())

    # Prepare data
    X = train_df[['year_built', 'square_feet', 'num_bedrooms', 'num_bathrooms', 'num_floors', 'garage_size', 'zestimate']]
    y = train_df['log_error']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Define models
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }

    # Train and evaluate models
    mse_results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_results[model_name] = mse
        print(f'{model_name} Mean Squared Error: {mse:.2f}')

    # Prepare test data
    X_test = test_df[['year_built', 'square_feet', 'num_bedrooms', 'num_bathrooms', 'num_floors', 'garage_size', 'zestimate']]
    X_test = scaler.transform(X_test)

    # Predict on test data with the best model (LinearRegression in this example)
    best_model = models['LinearRegression']
    test_predictions = best_model.predict(X_test)

    # Print test predictions to console
    print("Test Predictions:")
    print(test_predictions)

    # Add predictions to test dataframe
    test_df['predicted_log_error'] = test_predictions

    # Create plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(test_df['predicted_log_error'], kde=True, ax=axs[0])
    axs[0].set_title('Distribution of Predicted Log Error')

    sns.boxplot(x=test_df['predicted_log_error'], ax=axs[1])
    axs[1].set_title('Box Plot of Predicted Log Error')

    # Save plots to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)


# In[ ]:





# In[ ]:




