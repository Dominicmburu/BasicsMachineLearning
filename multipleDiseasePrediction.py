import pickle
import streamlit as st
from streamlit_option_menu import option_menu

## loading the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkison_model = pickle.load(open('parkison.sav', 'rb'))

# diab_prediction_1 = diabetes_model.predict([[4,110,92,0,0,37.6,0.191,30]])
# diab_prediction_2 = heart_disease_model.predict([[60,0,3,150,240,0,1,171,0,0.9,2,0,2]])
# diab_prediction_3 = parkison_model.predict([[199.22800,209.51200,192.09100,0.00241,0.00001,0.00134,0.00138,0.00402,0.01015,0.08900,0.00504,0.00641,0.00762,0.01513,0.00167,30.94000,0.432439,0.742055,-7.682587,0.173319,2.103106,0.068501]])
# print(diab_prediction_1)

## sidebar for navigation
with st.sidebar:

    selected = option_menu('Multiple Disease Prediction System',
                           
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],

                            icons = ['activity','heart','person'],
                            default_index=0)

## Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    ## page title
    st.title('Diabetes Prediction')

    ## getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFuction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    ## code for Prediction
    diab_diagnosis = ''

    ## creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFuction, Age]])

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is Diabetic'

        else:
            diab_diagnosis = 'The person is Not Diabetic'
    
    st.success(diab_diagnosis)


if (selected == 'Heart Disease Prediction'):
    
    ## page title
    st.title('Heart Disease Prediction')

    ## getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversed defect')

    ## code for Prediction
    diab_diagnosis = ''

    ## creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        diab_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person has Heart Disease'

        else:
            diab_diagnosis = 'The person has no Heart Disease'
    
    st.success(diab_diagnosis)

if (selected == 'Parkinsons Prediction'):
    
    ## page title
    st.title('Parkinsons Prediction')

    ## getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col1:
        jitter_percentage = st.text_input('MDVP:Jitter(%)')

    with col2:
        jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col3:
        rap = st.text_input('MDVP:RAP')

    with col1:
        ppq = st.text_input('MDVP:PPQ')

    with col2:
        ddp = st.text_input('Jitter:DDP')

    with col3:
        shimmer = st.text_input('MDVP:Shimmer')

    with col1:
        shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col2:
        shimmer_apq3 = st.text_input('Shimmer:APQ3')

    with col3:
        Shimmer_APQ5 = st.text_input('Shimmer:APQ5')

    with col1:
        MDVP_APQ = st.text_input('MDVP:APQ')

    with col2:
        Shimmer_DDA = st.text_input('Shimmer:DDA')

    with col3:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col1:
        spread1 = st.text_input('spread1')

    with col2:
        spread2 = st.text_input('spread2')

    with col3:
        D2 = st.text_input('D2')

    with col1:
        PPE = st.text_input('PPE')

    ## code for Prediction
    diab_diagnosis = ''

    ## creating a button for Prediction

    if st.button('Parkinsons Test Result'):
        diab_prediction = parkison_model.predict([[fo, fhi, flo, jitter_percentage, jitter_Abs, rap, ppq, ddp, shimmer, shimmer_dB, shimmer_apq3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person has Parkinsons'

        else:
            diab_diagnosis = 'The person does Not have Parkinsons'
    
    st.success(diab_diagnosis)