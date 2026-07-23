import xgboost as xgb
import streamlit as st
import joblib
import pandas as pd

# @st.cache_resource: loads everythung from disk ONCE when the app starts,
# not every time a widget gets touched. models/scalers/dictionaries are
# exactly the kind of thing this decorator is meant for.
@st.cache_resource
def load_artefacts():
    baseline_model = joblib.load('stroke_model_baseline.pkl')
    tuned_model = joblib.load('stroke_model_tuned.pkl')
    scaler = joblib.load('stroke_scaler.pkl')
    feature_columns = joblib.load('stroke_feature_columns.pkl')
    encoding_maps = joblib.load('stroke_encoding_maps.pkl')
    return baseline_model, tuned_model, scaler, feature_columns, encoding_maps

# unpack all five in one line, same order the function returns them
baseline_model, tuned_model, scaler, feature_columns, encoding_maps= load_artefacts()

st.sidebar.title('Input Features')
age= st.sidebar.slider('Age', min_value=0.0, max_value=100.0, value=45.0)
avg_glucose_level = st.sidebar.slider('avg_glucose_level', min_value=55.0, max_value=272.0,  value= 106.0)
bmi = st.sidebar.slider('bmi', min_value = 10.30, max_value= 97.60 , value=28.50 )

# The one-hot encoded stuff that split columns
work_type= st.sidebar.selectbox('Work Type',
    ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])

smoking_status = st.sidebar.selectbox('Smoking Status',
   ['Unknown', 'formerly smoked', 'never smoked', 'smokes']  )

gender = st.selectbox('Gender', ['Female', 'Male'])
ever_married = st.selectbox('Married',['No', 'Yes'])
Residence_type = st.selectbox('Residence', ['Rural', 'Urban'])

model_choice = st.sidebar.selectbox('Model',['Baseline', 'Tuned'])

st.sidebar.subheader('Medical History')
hypertension =  int(st.sidebar.toggle('Hypertension'))
heart_disease =  int(st.sidebar.toggle('Heart Disease'))

if st.sidebar.button('Predict'):
    input_dict = {
        'gender':encoding_maps['gender'][gender],
        'age':age,
        'hypertension':hypertension,
        'heart_disease': heart_disease,
        'ever_married': encoding_maps['ever_married'][ever_married],
        
        'Residence_type': encoding_maps['Residence_type'][Residence_type],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi, }

    for col in feature_columns:
        if col.startswith('work_type_'):
            input_dict[col]= 1 if col == f'work_type_{work_type}' else 0
        elif col.startswith('smoking_status_'):
            input_dict[col] = 1 if col == f'smoking_status_{smoking_status}' else 0


    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns= feature_columns, fill_value=0)

    cols_to_scale = ['age', 'avg_glucose_level', 'bmi']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    selected_model = baseline_model if model_choice == 'Baseline' else tuned_model
    probability = float(selected_model.predict_proba(input_df)[0][1])

    with st.container(border=True):
        st.subheader('Prediction Result')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Predicted Stroke Risk', f'{probability:.2%}')
        with col2:
            if probability >= 0.5:
                st.error('Higher Risk')
            else:
                st.success('Lower Risk')
        
        st.progress(probability)
    st.divider()
