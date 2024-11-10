
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# Load the model and encoders
with open('model_penguin_65130701938.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Load your DataFrame
# Replace 'your_data.csv' with the actual file name or URL
df = pd.read_csv('penguins_size.csv')
df = df.drop('species', axis=1)

# Streamlit App
st.title('Penguin Species Prediction')

# # Define a session state to remember tab selections
# if 'tab_selected' not in st.session_state:
#     st.session_state.tab_selected = 0

# # Create tabs for prediction and visualization
# tabs = ['Predict KPIs', 'Visualize Data', 'Predict from CSV']
# selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# # Tab selection logic
# if selected_tab != st.session_state.tab_selected:
#     st.session_state.tab_selected = tabs.index(selected_tab)

# # Tab 1: Predict KPIs
# if st.session_state.tab_selected == 0:
#     st.header('Predict KPIs')

    # User Input Form
    st.header('Enter Penguin Characteristics:')
    island = st.selectbox('Island', island_encoder.classes_)
    culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0)
    culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0)
    flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0)
    body_mass_g = st.number_input('Body Mass (g)', min_value=0.0)
    sex = st.selectbox('Sex', sex_encoder.classes_)

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
  })

   # Categorical Data Encoding
    user_input['island'] = island_encoder.transform(user_input['island'])
    user_input['sex'] = sex_encoder.transform(user_input['sex'])

    # Predicting
    if st.button('Predict'):
      try:
          prediction = model.predict(user_input)
          predicted_species = species_encoder.inverse_transform(prediction)[0]
          st.success(f'Predicted Species: **{predicted_species}**')
      except Exception as e:
          st.error(f"Prediction failed: {e}")

    # Display Result
    st.subheader('Prediction Result:')
    st.write('KPIs_met_more_than_80:', prediction[0])




