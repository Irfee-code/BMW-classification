import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Use pickle to load
import datetime
import xgboost # Required for loading the XGBoost model within the pipeline

# --- Configuration ---
st.set_page_config(layout="wide", page_title="BMW Sales Classification")
st.title("🚗 BMW Sales Classification Predictor")
st.markdown("Enter the car details below to predict if the sales classification will be 'High' or 'Low'.")

# --- Load Saved Objects ---
pickle_file = 'model_objects.pkl'  # Ensure this matches the save script filename
try:
    with open(pickle_file, 'rb') as f:
        loaded_objects = pickle.load(f)
    pipeline = loaded_objects['pipeline']
    le = loaded_objects['label_encoder']  # This was fitted on ['Low', 'High'] strings
    categorical_options = loaded_objects['categorical_options']
    numerical_features = loaded_objects['numerical_features']
    categorical_features = loaded_objects['categorical_features']
    st.success(f"Model and associated objects loaded successfully from '{pickle_file}'!")
except FileNotFoundError:
    st.error(f"Error: The file '{pickle_file}' was not found. Please run the `save_model.py` script first to create it.")
    st.stop() # Halt execution if file not found
except Exception as e:
    st.error(f"An error occurred while loading the pickle file: {e}")
    st.stop()

# --- User Input ---
st.sidebar.header("Input Car Features:")

input_data = {}

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Car Specifications")
    # Provide default empty list if key not found, prevent error
    input_data['Model'] = st.selectbox("Model", sorted(categorical_options.get('Model', ['N/A'])))
    # Use number_input for Year to calculate Car_Age
    year_input = st.number_input("Year", min_value=1990, max_value=datetime.datetime.now().year + 1, value=2020, step=1)
    input_data['Color'] = st.selectbox("Color", sorted(categorical_options.get('Color', ['N/A'])))
    input_data['Fuel_Type'] = st.selectbox("Fuel Type", sorted(categorical_options.get('Fuel_Type', ['N/A'])))
    input_data['Transmission'] = st.selectbox("Transmission", sorted(categorical_options.get('Transmission', ['N/A'])))

with col2:
    st.subheader("Performance & Condition")
    input_data['Engine_Size_L'] = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1, format="%.1f")
    input_data['Mileage_KM'] = st.number_input("Mileage (KM)", min_value=0, max_value=500000, value=50000, step=1000)
    input_data['Price_USD'] = st.number_input("Price (USD)", min_value=1000, max_value=500000, value=30000, step=1000)
    input_data['Region'] = st.selectbox("Region", sorted(categorical_options.get('Region', ['N/A'])))

# Calculate Car_Age based on Year input (this is a feature for the model)
current_year = datetime.datetime.now().year
input_data['Car_Age'] = current_year - year_input

# --- Prediction ---
if st.button("Predict Sales Classification", type="primary"):
    # Prepare input data as a DataFrame in the correct order
    # The order MUST match how the ColumnTransformer was fitted
    feature_order = categorical_features + numerical_features

    input_df_row = {}
    # Add categorical features first
    for feature in categorical_features:
        input_df_row[feature] = input_data[feature]
    # Add numerical features next
    for feature in numerical_features:
        # Special handling for Car_Age which was calculated
        if feature == 'Car_Age':
             input_df_row[feature] = input_data['Car_Age']
        else:
            input_df_row[feature] = input_data[feature]


    # Create the DataFrame row using the defined order
    input_df = pd.DataFrame([input_df_row], columns=feature_order)

    st.subheader("Input Data for Model:")
    st.dataframe(input_df)

    try:
        # Predict using the loaded pipeline (expects DataFrame input)
        prediction_encoded = pipeline.predict(input_df) # Predicts 0 or 1
        prediction_proba = pipeline.predict_proba(input_df) # Predicts probabilities [P(0), P(1)]

        # Decode the prediction (0 or 1) back to 'Low' or 'High'
        # The label encoder 'le' saved was fitted on the original strings
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        st.subheader("Prediction Result:")
        # Check the decoded string label
        if prediction_label == 'High':
            st.success(f"Predicted Sales Classification: **{prediction_label}** 🎉")
            st.write(f"Confidence (Probability of High): {prediction_proba[0][1]:.2f}") # Index 1 corresponds to 'High'
        else: # prediction_label == 'Low'
            st.warning(f"Predicted Sales Classification: **{prediction_label}**")
            st.write(f"Confidence (Probability of Low): {prediction_proba[0][0]:.2f}") # Index 0 corresponds to 'Low'

        # Display probabilities clearly
        st.write("Prediction Probabilities:")
        # Ensure le.classes_ maps correctly to the probabilities array indices
        proba_df = pd.DataFrame(prediction_proba, columns=le.classes_) # Should be ['Low', 'High'] if fitted correctly
        st.dataframe(proba_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check the input values and ensure the model loaded correctly.")
        # Print input_df columns for debugging if needed
        # st.write("Columns sent to pipeline:", input_df.columns.tolist())


st.markdown("---")
st.markdown("Developed based on BMW Sales Data Analysis.")
