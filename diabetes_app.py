import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from PIL import Image

# Load the diabetes dataset
@st.cache_data()
def load_data():
    df = pd.read_csv('./data/diabetes.csv') 
    return df

# Load the pre-trained model and scaler from the pickle files
@st.cache_data()
def load_model():
    with open('./data/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the pre-fitted scaler from the pickle file
@st.cache_data()
def load_scaler():
    with open('./data/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Prediction function
def make_prediction(model, scaler, inputs):
    data = {
        "Pregnancies": inputs[0],
        "Glucose": inputs[1],
        "BloodPressure": inputs[2],
        "SkinThickness": inputs[3],
        "Insulin": inputs[4],
        "BMI": inputs[5],
        "DiabetesPedigreeFunction": inputs[6],
        "Age": inputs[7]
    }
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    df_scaled_np = np.array(df_scaled)
    prediction = model.predict(df_scaled_np)
    return prediction

# Sidebar for navigation
st.sidebar.title("Navigation")
tab_selection = st.sidebar.radio("Select a Tab", ["Prediction", "Data", "Model"])

if tab_selection == "Prediction":
    # Header Image
    header_image = Image.open('./data/high_res_header.png')
    st.image(header_image, use_column_width=True)

    # Page Title
    st.title("Diabetes Prediction Interface")
    st.markdown("Please enter all your relevant health metrics below and press the **Predict** button to get your prediction. If you don't have all information or are not sure just leave the current value in the field")

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=500, value=50)

    with col2:
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()

    # Prediction button
    if st.button("Predict"):
        # Make prediction
        inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        prediction = make_prediction(model, scaler, inputs)

        # Display the result with corresponding image
        if prediction == 0:  # No Diabetes
            result_image = Image.open('./data/high_res_positive.png')
            st.image(result_image, width=150)
            st.write("The model predicts that you are unlikely to have diabetes. Be aware that this does not constitute an official diagnosis!")
        else:  # Diabetes
            result_image = Image.open('./data/high_res_negative.png')
            st.image(result_image, width=150)
            st.write("The model predicts that you could have diabetes. Be aware that this does not constitute an official diagnosis!")

# Data Tab
elif tab_selection == "Data":
    st.title('Data Visualizations')

    df = load_data()

    # Display the first few rows of the dataset
    st.subheader('Dataset')
    st.write(df.head())

    # Display the Correlation Matrix (image from the notebook)
    st.subheader("Correlation Matrix")
    st.image('./data/Corr.Matrix.png', caption='Correlation Matrix', use_column_width=True)

    # Dropdowns for selecting x and y variables for pairplot
    st.subheader('Select Variables for Pairplot')
    
    # Dropdown for x and y axes
    x_axis = st.selectbox('Select X-axis Variable', df.columns)
    y_axis = st.selectbox('Select Y-axis Variable', df.columns)

    # Scatter plot based on selected variables
    st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df['Outcome'], palette='coolwarm', ax=ax)
    st.pyplot(fig)

# Model Tab
elif tab_selection == "Model":
    st.title('Model Evaluation')

    # Display the Confusion Matrix (image from the notebook)
    st.subheader("Confusion Matrix")
    st.image('./data/ConfusionMatrix.png', caption='Confusion Matrix', use_column_width=True)

    # Display the Classification Report as a table (from CSV)
    st.subheader("Classification Report")
    report_df = pd.read_csv('./data/classification_report.csv')  # Assuming you saved the report as CSV
    st.dataframe(report_df)

    # Display the ROC Curve (image from the notebook)
    st.subheader("ROC Curve")
    st.image('./data/ROCCurve.png', caption='ROC Curve', use_column_width=True)

