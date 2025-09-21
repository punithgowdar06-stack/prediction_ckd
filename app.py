import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained models
# Assuming the models are saved in a directory named 'trained_models'
# and the filenames are logistic_regression.joblib, svm.joblib, etc.

models = {
    'Logistic Regression': joblib.load('trained_models/logistic_regression.joblib'),
    'SVM': joblib.load('trained_models/svm.joblib'),
    'KNN': joblib.load('trained_models/knn.joblib'),
    'Random Forest': joblib.load('trained_models/random_forest.joblib'),
    'Naive Bayes': joblib.load('trained_models/naive_bayes.joblib')
}

# Assuming you have a scaler fitted on your training data
# It's crucial to use the same scaler that was used during training
# For this example, I'll create a dummy scaler.
# In a real scenario, you would save and load the fitted scaler as well.
# Since we don't have the original scaler object, we'll need to refit one
# on a dummy dataset with the same feature names or ideally load a saved scaler.
# For the purpose of making the app runnable with the existing notebook context,
# I'll use the X_train DataFrame available in the notebook's global scope to fit the scaler.
# In a production app, the scaler object should be saved and loaded.

# Access the X_train from the notebook's global scope
# This is a workaround for the Colab environment.
# In a standard script, you would load a pre-saved scaler.
if 'X_train' in globals():
    scaler = StandardScaler()
    scaler.fit(X_train) # Fit the scaler on the training data used in the notebook
else:
    st.error("X_train not found. Cannot fit the scaler. Please run the data preparation steps in the notebook.")
    st.stop() # Stop the app if scaler cannot be fitted


st.title('Chronic Kidney Disease Prediction')

st.sidebar.header('Input Features')

# Create input fields for features
def user_input_features():
    input_data = {}
    # Get feature names from the fitted scaler (assuming X_train was used)
    feature_names = X_train.columns if 'X_train' in globals() else []

    if not feature_names:
        st.warning("Feature names not available. Cannot display input fields.")
        return None

    for feature in feature_names:
        # You might need to adjust the default values and types based on your dataset
        # For simplicity, using number_input for all numerical features
        input_data[feature] = st.number_input(f'Enter {feature}', value=0.0) # Using 0.0 as a placeholder default

    return pd.DataFrame([input_data])

input_df = user_input_features()

# Select model
selected_model_name = st.sidebar.selectbox('Select Model', list(models.keys()))

# Make prediction
if st.sidebar.button('Predict'):
    if input_df is not None:
        # Preprocess the input data using the same scaler
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

        # Get the selected model
        model = models[selected_model_name]

        # Make prediction
        prediction = model.predict(input_scaled_df)
        prediction_proba = model.predict_proba(input_scaled_df)

        st.subheader('Prediction')
        diagnosis = 'Chronic Kidney Disease' if prediction[0] == 1 else 'No Chronic Kidney Disease'
        st.write(f'The predicted diagnosis is: **{diagnosis}**')

        st.subheader('Prediction Probability')
        st.write(f'Probability of No Chronic Kidney Disease (Class 0): {prediction_proba[0][0]:.4f}')
        st.write(f'Probability of Chronic Kidney Disease (Class 1): {prediction_proba[0][1]:.4f}')
