import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model, scaler, and label encoders
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define numerical and categorical columns (same as training, including MTRANS)
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']

# Define expected feature order (same as X_resampled during training)
feature_order = numerical_cols + categorical_cols

# Streamlit app
st.title('Prediksi Tingkat Obesitas')
st.write('Masukkan detail berikut untuk memprediksi tingkat obesitas:')

# Input fields
with st.form('prediction_form'):
    age = st.number_input('Usia', min_value=10, max_value=100, value=25)
    height = st.number_input('Tinggi Badan (m)', min_value=1.0, max_value=2.5, value=1.7, format="%.2f")
    weight = st.number_input('Berat Badan (kg)', min_value=30, max_value=200, value=70)
    fcvc = st.number_input('FCVC (Frekuensi Konsumsi Sayur)', min_value=1, max_value=3, value=2)
    ncp = st.number_input('NCP (Jumlah Makan Utama per Hari)', min_value=1, max_value=4, value=3)
    ch2o = st.number_input('CH2O (Asupan Air, liter)', min_value=1, max_value=3, value=2)
    faf = st.number_input('FAF (Frekuensi Aktivitas Fisik)', min_value=0, max_value=3, value=1)
    tue = st.number_input('TUE (Waktu Penggunaan Teknologi, jam)', min_value=0, max_value=10, value=1)

    gender = st.selectbox('Gender', ['Male', 'Female'])
    calc = st.selectbox('CALC (Alcohol Consumption)', ['no', 'Sometimes', 'Frequently', 'Always'])
    favc = st.selectbox('FAVC (High Caloric Food)', ['yes', 'no'])
    scc = st.selectbox('SCC (Calorie Monitoring)', ['yes', 'no'])
    smoke = st.selectbox('SMOKE (Smoking)', ['yes', 'no'])
    family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
    caec = st.selectbox('CAEC (Food Between Meals)', ['no', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.selectbox('MTRANS (Moda Transportasi)', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'], 
                         format_func=lambda x: x)

    submit = st.form_submit_button('Prediksi')

if submit:
    try:
        input_data = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'FCVC': fcvc,
            'NCP': ncp,
            'CH2O': ch2o,
            'FAF': faf,
            'TUE': tue,
            'Gender': gender,
            'CALC': calc,
            'FAVC': favc,
            'SCC': scc,
            'SMOKE': smoke,
            'family_history_with_overweight': family_history,
            'CAEC': caec,
            'MTRANS': mtrans
        }

        input_df = pd.DataFrame([input_data])[feature_order]

        # Debug: Tampilkan nama kolom input untuk memverifikasi
        st.write("Nama kolom input:", input_df.columns.tolist())
        st.write("Urutan fitur yang diharapkan:", feature_order)

        for col in categorical_cols:
            if input_df[col].iloc[0] not in label_encoders[col].classes_:
                st.error(f"Nilai tidak valid untuk {col}: {input_df[col].iloc[0]}. Harus salah satu dari {list(label_encoders[col].classes_)}")
                st.stop()
            input_df[col] = label_encoders[col].transform(input_df[col])

        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = model.predict(input_df)
        obesity_level = label_encoders['NObeyesdad'].inverse_transform(prediction)[0]

        st.success(f'Tingkat Obesitas yang Diprediksi: **{obesity_level}**')

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
