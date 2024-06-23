import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Statistik untuk kolom umur berdasarkan keterangan yang diberikan
mean_umur = 43.582741
std_umur = 18.650114

# Load model dan scaler
model = load_model('models/tbc_model.h5')
scaler = joblib.load('models/scaler.joblib')

# Dictionary untuk mapping nilai dropdown
jenis_kelamin_map = {
    0: 'Perempuan',
    1: 'Laki-laki'
}

foto_toraks_map = {
    0: 'Negatif',
    1: 'Positif'
}

status_hiv_map = {
    0: 'Negatif',
    1: 'Positif'
}

riwayat_diabetes_map = {
    0: 'Tidak',
    1: 'Ya'
}

hasil_tcm_map = {
    0: 'Negatif',
    1: 'Rif Sensitif',
    2: 'Rif Resisten'
}

lokasi_anatomi_map = {
    0: 'Paru',
    1: 'Ekstra Paru'
}

# Fungsi untuk normalisasi umur
def normalize_umur(umur):
    return (umur - mean_umur) / std_umur

# Fungsi untuk preprocessing data input
def preprocess_input(umur, jenis_kelamin, foto_toraks, status_hiv, riwayat_diabetes, hasil_tcm):
    data_baru = {
        'umur': [umur],
        'jenis_kelamin': [next(key for key, value in jenis_kelamin_map.items() if value == jenis_kelamin)],
        'foto_toraks': [next(key for key, value in foto_toraks_map.items() if value == foto_toraks)],
        'status_hiv': [next(key for key, value in status_hiv_map.items() if value == status_hiv)],
        'riwayat_diabetes': [next(key for key, value in riwayat_diabetes_map.items() if value == riwayat_diabetes)],
        'hasil_tcm': [next(key for key, value in hasil_tcm_map.items() if value == hasil_tcm)]
    }
    df_baru = pd.DataFrame(data_baru)
    df_baru['umur'] = normalize_umur(df_baru['umur'].values[0])
    return scaler.transform(df_baru)

# Fungsi untuk prediksi berdasarkan input pengguna
def predict(input_data):
    prediksi = model.predict(input_data)
    hasil_prediksi = 1 if prediksi[0] > 0.5 else 0
    return lokasi_anatomi_map[hasil_prediksi]

# Tampilan aplikasi dengan sidebar
st.sidebar.title('Prodiksi Lokasi Anatomi TBC')
pages = ['Data Understanding', 'Preprocessing', 'Modelling', 'Implementasi']
selected_page = st.sidebar.radio('', pages)

if selected_page == 'Data Understanding':
    st.title('Data Understanding')
    st.header("Informasi Data")
    st.subheader("Data Awal")
    data_awal = pd.read_csv("tbc_ori.csv")
    st.write(data_awal)
    st.subheader("dataset TBC asli memiliki 7 fitur :")
    st.markdown("""
    <ul>
        <li>Umur : berisi umur dari pasien yang mengalami TBC</li>
        <li>Jenis Kelamin : berisi jenis kelamin yang dimiliki pasien</li>
        <li>Kecamatan : informasi kecamatan asal pasien</li>
        <li>
            Foto Toraks : informasi tentang kondisi dada dari pasien
            <ul>
                <li>Positif : berarti bahwa hasil pemeriksaan sinar-X menunjukkan adanya temuan atau perubahan yang abnormal pada struktur di dalam dada</li>
                <li>Negatif :  hasil pemeriksaan sinar-X dikatakan "negatif" jika tidak ada temuan atau perubahan yang abnormal yang terlihat pada struktur di dalam dada.</li>
            </ul>    
        </li>
        <li>Status Hiv : informasi apakah pasien terinfeksi HIV atau tidak. Positif berarti terinfeksi dan Negatif berarti tidak terinfeksi</li>
        <li>Riwayat Diabetes : informasi apakah pasien pernah terkena diabetes atau tidak</li>
        <li>
            Hasil TCM : informasi yang merujuk kepada hasil uji kepekaan obat terhadap rifampisin, salah satu antibiotik yang penting dalam pengobatan TB.
            <ul>
                <li>Negatif :   hasil uji yang menunjukkan bahwa bakteri TB tidak tumbuh atau tidak dapat dideteksi dalam sampel tertentu, seperti tes kultur</li>
                <li>Rif Sensitif: Ini berarti bahwa bakteri Mycobacterium tuberculosis, penyebab TB, peka terhadap rifampisin.</li>
                <li>Rif Resisten: Ini berarti bahwa bakteri TB tidak peka terhadap rifampisin.</li>
            </ul>  
        </li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.subheader("dataset TBC memiliki 2 Kelas (Lokasi Anatomi):")
    st.write("mengacu pada lokasi anatomis di mana infeksi tuberkulosis dapat terjadi")
    st.markdown("""
    <ul>
        <li>Paru : Ini adalah organ utama yang terlibat dalam infeksi TBC. Tuberkulosis paru adalah bentuk paling umum dari penyakit ini, di mana bakteri Mycobacterium tuberculosis menginfeksi paru-paru. </li>
        <li>Ekstra Paru : infeksi TBC di luar paru-paru. Infeksi ini dapat menyerang organ atau jaringan lain di tubuh, seperti pleura (lapisan tipis yang melapisi paru-paru), kelenjar limfe, tulang, sendi, ginjal, otak, dan bagian lain dari tubuh.</li>
    </ul>
    """, unsafe_allow_html=True)

elif selected_page == 'Preprocessing':
    st.title('Preprocessing')
    st.write('Sebelum dataset kita latih dengan model, akan lebih baik apabila kita melakukan preprosessing terlebih dahulu untuk meningkatkan kulitas data kita, sehingga akurasi model juga akan ikut meningkat')

    st.subheader("Menghapus fitur Kecamatan")
    st.write("Pada dataset terdapat fitur yang tidak mempengaruhi keputusan dalam menentukan suatu pasien termasuk ke dalam jenis TBC paru atau Ekstra Paru. Fitur yang tidak terpakai yaitu kecamatan, maka akan lebih baik apabila fitur tersebut dihapus pada dataset")
    data_xkec = pd.read_csv("tbc_xkec.csv")
    st.write(data_xkec)

    st.subheader("Transformasi")
    st.write("model machine learning tidak dapat memproses data teks. Pada dataset TB yang kita miliki terdapat data berupa teks hal ini perlu untuk kita lakukakan pada transformasi dataset kita sehingga model dapat membaca dataset. pada transformasi kami melakukannya pada fitur jenis kelamin, foto toraks, status hiv, riwayat diabetes dan lokasi anatomi, dan kami juga melakukan tranformasi pada kolom kelas")
    st.markdown("""
    <b>Keterangan : </b>
    <ul>
        <li>
            Jenis Kelamin
            <ul>
                <li>P : 0</li>
                <li>L : 1</li>
            </ul>
        </li>
        <li>
            Foto Toraks
            <ul>
                <li>Negatif : 0</li>
                <li>Positif : 1</li>
            </ul>
        </li>
        <li>
            Status HIV
            <ul>
                <li>Negatif : 0</li>
                <li>Positif : 1</li>
            </ul>
        </li>
        <li>
            Riwayat Diabetes
            <ul>
                <li>Tidak : 0</li>
                <li>Ya : 1</li>
            </ul>
        </li>
        <li>
            Hasil TCM
            <ul>
                <li>Negatif : 0</li>
                <li>Rif Sensitif : 1</li>
                <li>Rif resisten : 2</li>
            </ul>
        </li>
        <li>
            Lokasi Anatomi
            <ul>
                <li>Paru : 0</li>
                <li>Ekstra Paru : 1</li>
            </ul>
        </li>
    </ul>
    """, unsafe_allow_html=True)

    st.subheader("Normalisasi")
    st.write("tujuan dilakukannya normalisasi pada data adalah supaya tidak terjadi overfiting pada model kita. sehingga peran normalisasi sangat berperngaruh dalam hasil prediksi data kita. untuk normalisasi saya menggunakan metode z-score")

    st.subheader("Hasil transformasi dan normalisasi")
    data_pre = pd.read_csv("tbc_preprocess1.csv")
    st.write(data_pre)

    st.subheader("Missing Values")
    st.write("Pada dataset kami terdapat banyak missing value, oleh karena itu perlu adanya imputasi terhadap missing value. Kami akan menggunakan metode KNN untuk melakukan imputasi terhadap missing value. Berikut adalah dataset yang sudah kami imputasi.")
    data_clean = pd.read_csv("tbc_clean.csv")
    st.write(data_clean)

elif selected_page == 'Modelling':
    st.title('Modelling')
    st.header('Jaringan Saraf Tiruan Backpropagation')

    st.subheader("Splitting Dataset")
    st.write("dataset kami bagi menjadi 80:20, 80% training dan 20% testing")
    train_df = pd.read_csv("xtrain.csv")
    test_df = pd.read_csv("xtest.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Train")
        st.write(train_df)
        st.write(train_df.shape)

    with col2:
        st.subheader("Data Test")
        st.write(test_df)
        st.write(test_df.shape)

    st.subheader("Informasi Model")
    st.write("Model jst backprop kami menggunakan 6 perceptron sesuai dengan jumlah inputan fitur, yang nantinya akan dimasukkan ke dalam hidden layer dengan fungsi aktifasi relu, kemudian akan diarahkan ke dalam 1 output layer dimana menggunakan fungsi aktivasi sigmoid dikarenakan label pada dataset kami kategorikal binary. kemudian untuk mengurangi gradien error dilakukan backprop yang akan kembali ke dalam hidden layer dan akan menuju ke output  layer. Proses backprop dilakukan sebanyak 32x per epochnya, dan kami menggunakan epoch sebanyak 50 epoch.")

    st.subheader("Evaluasi Model")
    st.markdown("""
    <ul>
        <li>Accuracy: 0.9746</li>            
        <li>Precision: 1.0000</li>            
        <li>F1 Score: 0.9524</li>            
        <li>Confusion Matrix:<br>
            [[142   0]<br>
            [  5  50]]        
    </ul>
    """, unsafe_allow_html=True)

elif selected_page == 'Implementasi':
    st.title('Implementasi')
    st.subheader('Prediksi Lokasi Anatomi TBC')

    # Input dari pengguna
    umur = st.number_input('Umur', min_value=1)
    jenis_kelamin = st.selectbox('Jenis Kelamin', options=list(jenis_kelamin_map.values()))
    foto_toraks = st.selectbox('Foto Toraks', options=list(foto_toraks_map.values()))
    status_hiv = st.selectbox('Status HIV', options=list(status_hiv_map.values()))
    riwayat_diabetes = st.selectbox('Riwayat Diabetes', options=list(riwayat_diabetes_map.values()))
    hasil_tcm = st.selectbox('Hasil TCM', options=list(hasil_tcm_map.values()))

    # Prediksi berdasarkan input pengguna
    if st.button('Prediksi'):
        input_data = preprocess_input(umur, jenis_kelamin, foto_toraks, status_hiv, riwayat_diabetes, hasil_tcm)
        hasil_prediksi = predict(input_data)
        st.write(f'Hasil Prediksi: {hasil_prediksi}')

# Untuk menjalankan aplikasi Streamlit, gunakan perintah di terminal:
# streamlit run app.py
