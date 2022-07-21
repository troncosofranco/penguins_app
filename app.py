#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st
import joypy as joypy
from PIL import Image

#Head
st.title("Classification Penguins app")
st.write("This apps predicts the Penguin specie using ML classification algorithm")

penguin_logo = Image.open("penguins_logo.png")
st.image(penguin_logo, use_column_width=False)

#Load data
data = pd.read_csv('penguins.csv')

#data claned provided by dataprofessor
#https://github.com/dataprofessor/streamlit_freecodecamp/blob/main/app_8_classification_penguins/penguins_cleaned.csv
st.write(data.head())

st.subheader("Plotting the data")
#plotting the data
#Define features to plot
st.sidebar.subheader("Select features to plot")
features_x = ['species', 'island', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'] 
features_y = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
selected_x = st.sidebar.selectbox('Feature x', features_x)
selected_y = st.sidebar.selectbox('Feature y', features_y)



#Define features to predict
#Categorigal features
st.sidebar.subheader("Select features to predict species")
island_vector = data.island.unique()
sex_vector = data.sex.unique()

selected_island = st.sidebar.selectbox('Island', island_vector)
selected_sex = st.sidebar.selectbox('Gender', sex_vector)


#Numeric features
selected_bill_length = st.sidebar.slider('Bill length (mm)', 32.1,59.61,43.9) 
selected_bill_depth = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
selected_flipper_length = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
selected_body_mass = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
input_features = {'island': selected_island,
                'bill_length_mm': selected_bill_length,
                'bill_depth_mm': selected_bill_depth,
                'flipper_length_mm': selected_flipper_length,
                'body_mass_g': selected_body_mass,
                'sex': selected_sex}

input_df = pd.DataFrame(input_features, index=[0])

#encoding categorical features of the input
# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

########################################################
#Predict section
########################################################
#load model
import pickle
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

#Making predictions
prediction = load_clf.predict(df)

#Showing prediction probabilities
prediction_proba = load_clf.predict_proba(df)

########################################################
#Plot functions
########################################################

#violin plot
def violin_plot():
    fig, ax = plt.subplots()
    sns.violinplot(x=data[str(selected_x)], y=data[str(selected_y)], data=data)
    return st.pyplot(fig)

#distribution plot
def distribution():
    
    sns.FacetGrid(data, hue=str(selected_x), height=6,).map(sns.kdeplot, str(selected_y),shade=True).add_legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return st.pyplot()

#scatter plot
def scatter_plot():
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=data[str(selected_x)], y=data[str(selected_y)], hue='species')
    return st.pyplot(fig)

#gender distribution plot
def gender_distribution():
    data_gender = data.copy()
    data_gender["male"] = data_gender.apply(lambda row: row[str(selected_y)] if row["sex"] == "male" else np.nan, axis = 1)
    data_gender["female"] = data_gender.apply(lambda row: row[str(selected_y)] if row["sex"] == "female" else np.nan, axis = 1)
    fig, axes = joypy.joyplot(data_gender,  column=['female', 'male'],
                          by = "species",
                          ylim = 'own',
                          figsize = (12,8), 
                          legend = True)
    return st.pyplot(fig)


#Buttons
#Plot button
if st.button('Violin plot'):
    st.header('violin plot')
    violin_plot()

if st.button('Distribution'):
    st.header('Property distribution')
    distribution()

if st.button('Gender distribution'):
    st.header('Gender distribution')
    gender_distribution()

if st.button('Scatter plot'):
    st.header('Scatter plot')
    scatter_plot()

#show prediction
st.subheader("Prediction")
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write("The penguin is a ", penguins_species[prediction[0]], " specie.")

penguin_type = str(penguins_species[prediction[0]])

if penguin_type == "Adelie":
    image_type = Image.open("adelie.jpg")
elif penguin_type == "Chinstrap": 
    image_type = Image.open("chinstrap.jpg")
else:
    image_type = Image.open("gentoo.jpg")
    

st.image(image_type, use_column_width=False)


st.subheader("Prediction probability:")
results = {'Type':['Adelie', 'Chinstrap', 'Gentoo'],
        'Probability':[prediction_proba[0,0],prediction_proba[0,1], prediction_proba[0,2]]}
  
# Create DataFrame
df_results = pd.DataFrame(results)

# Plot probabilities
fig, ax = plt.subplots(figsize=(5,3))
sns.barplot(x=df_results.Type, y=df_results.Probability, color='goldenrod', ax=ax, label="Probability plot")
ax.set_xlabel("Type")
ax.set_ylabel("Probability")
st.pyplot(fig)


st.markdown("""
Credits:
* **Images:** [Google] (https://www.google.com)
* **Data Provided by:** [Dataprofessor](https://github.com/dataprofessor/streamlit_freecodecamp/tree/main/app_8_classification_penguins)
* **Code contribution:**
* [Dataprofessor] (https://github.com/dataprofessor/streamlit_freecodecamp/blob/main/app_8_classification_penguins/penguins-app.py)
* [Pratik Mukherjee] (https://www.kaggle.com/code/pratik1120/penguin-dataset-eda-classification-and-clustering/notebook)
""")