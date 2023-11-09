import streamlit as st
import pandas as pd
import joblib  

# Create a Streamlit app
st.title("Sweet People Prediction App")

# Define labels
labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Create input fields for each label
values = []
for label in labels:
    value = st.number_input(label, min_value=0, key=label)
    values.append(value)

# Button to make predictions
if st.button("Predict"):
   
    @st.cache_data()  
    def load_model():
        model = joblib.load('model.pkl')  
        return model

    model = load_model()

    # Create a DataFrame from the input values
    input_data = pd.DataFrame([values], columns=labels)

    # Use the model to make a prediction
    prediction = model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: You have diabetes.You are literally sweet")
    else:
        st.write("Prediction: You do not have diabetes.You are bitter")