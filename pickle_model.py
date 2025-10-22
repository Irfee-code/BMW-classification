import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Keep LabelEncoder import for saving
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime
import pickle # Import pickle library

print("--- Script Start: Saving Final Model ---")

# --- 1. Load Data ---
print("Loading data from 'BMW_Car_Sales_Classification.csv'...")
try:
    df = pd.read_csv('BMW_Car_Sales_Classification.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'BMW_Car_Sales_Classification.csv' not found. Please place it in the same directory.")
    exit() # Stop if data not found
except Exception as e:
    print(f"ERROR loading data: {e}")
    exit()

# --- 2. Feature Engineering ---
print("Performing feature engineering...")
current_year = datetime.datetime.now().year
df['Car_Age'] = current_year - df['Year']

# --- 3. Clean and Map Target Variable (y) ---
# Clean the text first (important for consistency)
df['Sales_Classification_Clean'] = df['Sales_Classification'].str.strip().str.capitalize()
y = df['Sales_Classification_Clean'].map({'High': 1, 'Low': 0})

# Check for mapping errors (NaNs)
if y.isnull().any():
    print("ERROR: Mapping failed for some values in 'Sales_Classification'.")
    print("Problematic values:", df[y.isnull()]['Sales_Classification'].unique())
    # Decide how to handle NaNs (e.g., drop rows)
    # For this script, we'll stop if there are errors
    exit()
else:
     print("Target variable mapped successfully ('High'=1, 'Low'=0).")


# --- 4. Define Features (X) ---
# Use the same feature set as in your tuning code
X = df.drop(['Sales_Classification', 'Sales_Classification_Clean', 'Sales_Volume', 'Year'], axis=1)
print("Features (X) and Target (y) defined.")

# --- 5. Calculate Imbalance Weight ---
counts = y.value_counts()
if 0 not in counts or 1 not in counts:
     raise ValueError("Target variable 'y' does not contain both 0 and 1 after mapping.")
scale_pos_weight = counts[0] / counts[1] # count_Low / count_High
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# --- 6. Define Feature Lists ---
categorical_features = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission']
numerical_features = ['Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Car_Age']
print("Feature lists defined.")

# --- 7. Create Preprocessor ---
# Use the same preprocessor definition
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])
print("Preprocessor created.")

# --- 8. Define the FINAL Model Pipeline ---
# Use the best parameters found during your RandomizedSearchCV
best_params_from_tuning = {
    'subsample': 0.7,
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.01,
    'colsample_bytree': 0.8
}
print(f"Using best parameters from tuning: {best_params_from_tuning}")

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        **best_params_from_tuning # Unpack the best parameters
    ))
])
print("Final pipeline defined.")

# --- 9. Train the Final Pipeline ---
# Train on the *entire* dataset (X, y)
print("Training the final pipeline on the full dataset...")
final_pipeline.fit(X, y)
print("Pipeline training complete.")

# --- 10. Prepare Objects for Saving ---
# Create a *new* LabelEncoder just for saving the class names ('Low', 'High')
# This is needed so the Streamlit app knows how to decode the 0/1 prediction
le_for_saving = LabelEncoder()
le_for_saving.fit(df['Sales_Classification_Clean'].dropna()) # Fit on original cleaned strings

# Get unique options for dropdowns in Streamlit
categorical_options = {col: X[col].unique().tolist() for col in categorical_features}
print("Categorical options extracted.")

# --- 11. Save Objects to Pickle File ---
save_path = 'model_objects.pkl'
objects_to_save = {
    'pipeline': final_pipeline,
    'label_encoder': le_for_saving, # Save the LE fitted on strings
    'categorical_options': categorical_options,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features
}

print(f"Saving objects to {save_path}...")
with open(save_path, 'wb') as f:
    pickle.dump(objects_to_save, f)

print("--- Objects successfully saved! ---")
print(f"File '{save_path}' created.")
print("You can now use this file with your Streamlit app.")
print("--- Script End ---")

