import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os

# ==========================
# Load model Random Forest
# ==========================
MODEL_PATH = "model/model_rf_hargapenawaran.joblib"
if os.path.exists(MODEL_PATH):
    model_rf = joblib.load(MODEL_PATH)
    st.session_state["model_rf"] = model_rf
else:
    st.warning("⚠️ Model belum tersedia!")

# ==========================
# Sidebar Navigasi
# ==========================
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Beranda", "Evaluasi Model", "Model Dasar Prediksi", "Prediksi"]
)

# ==========================
# Halaman 1: Beranda
# ==========================
if page == "Beranda":
    st.title("🏠 Beranda")
    st.markdown("""
    Selamat datang di aplikasi Prediksi Harga Penawaran.  
    Gunakan modul **Evaluasi Model** untuk mengecek performa model.  
    Gunakan modul **Prediksi** untuk menghitung harga baru dari input variabel.
    """)

# ==========================
# Halaman 2: Evaluasi Model
# ==========================
elif page == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    uploaded_file = st.file_uploader(
        "Upload dataset CSV/XLSX dengan kolom target 'HARGAPENAWARAN'", 
        type=["csv","xlsx"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
        
        if "HARGAPENAWARAN" not in df.columns:
            st.error("Kolom target 'HARGAPENAWARAN' tidak ditemukan!")
        else:
            target = "HARGAPENAWARAN"
            X = df.drop(columns=[target])
            y = df[target]
            X = X.loc[:, X.nunique() > 1]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=77
            )

            # Daftar semua model
            all_models = {
                'Linear Regression': LinearRegression(),
                'Support Vector Regression': SVR(),
                'K-Nearest Neighbor': KNeighborsRegressor(),
                'Elastic Net': ElasticNet(random_state=77),
                'Passive Aggressive Regressor': PassiveAggressiveRegressor(random_state=77),
                'Random Forest': RandomForestRegressor(random_state=77),
                'Gradient Boosting': GradientBoostingRegressor(random_state=77),
                'LightGBM': LGBMRegressor(random_state=77),
                'XGBoost': xgb.XGBRegressor(random_state=77, verbosity=0)
            }

            # Pilih model yang ingin dievaluasi
            selected_models = st.multiselect(
                "Pilih model untuk dievaluasi:",
                options=list(all_models.keys()),
                default=["Random Forest", "Linear Regression"]
            )

            # DataFrame hasil evaluasi
            results = pd.DataFrame(columns=[
                'Model',
                'R2_in_sample', 'R2_out_sample',
                'MSE_in_sample', 'MSE_out_sample',
                'RMSE_in_sample', 'RMSE_out_sample',
                'MAE_in_sample', 'MAE_out_sample',
                'MAPE_in_sample', 'MAPE_out_sample'
            ])

            # Training & evaluasi model terpilih
            for name in selected_models:
                model = all_models[name]
                try:
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred  = model.predict(X_test)

                    mse_in  = mean_squared_error(y_train, y_train_pred)
                    rmse_in = np.sqrt(mse_in)
                    mae_in  = mean_absolute_error(y_train, y_train_pred)
                    mape_in = mean_absolute_percentage_error(y_train, y_train_pred)
                    r2_in   = r2_score(y_train, y_train_pred)

                    mse_out  = mean_squared_error(y_test, y_test_pred)
                    rmse_out = np.sqrt(mse_out)
                    mae_out  = mean_absolute_error(y_test, y_test_pred)
                    mape_out = mean_absolute_percentage_error(y_test, y_test_pred)
                    r2_out   = r2_score(y_test, y_test_pred)

                    results = pd.concat([results, pd.DataFrame([{
                        'Model': name,
                        'R2_in_sample': r2_in,
                        'R2_out_sample': r2_out,
                        'MSE_in_sample': mse_in,
                        'MSE_out_sample': mse_out,
                        'RMSE_in_sample': rmse_in,
                        'RMSE_out_sample': rmse_out,
                        'MAE_in_sample': mae_in,
                        'MAE_out_sample': mae_out,
                        'MAPE_in_sample': mape_in,
                        'MAPE_out_sample': mape_out
                    }])], ignore_index=True)
                except Exception as e:
                    st.warning(f"⚠️ Model {name} gagal dijalankan: {e}")

            # Tampilkan hasil sorted by RMSE_out_sample
            results = results.sort_values(by='RMSE_out_sample')
            st.subheader("Hasil Evaluasi Model")
            st.dataframe(results)

# ==========================
# Halaman 3: Model Dasar Prediksi (GitHub)
# ==========================
elif page == "Model Dasar Prediksi":
    st.title("📈 Model Dasar Prediksi")

    # Ganti <username> dan <repo> dengan akun GitHub dan repo-mu
    GITHUB_URL = "https://raw.githubusercontent.com/matinbook-learningmachine/PrediksiSewaRuang/main/datamodelprediksi.xlsx"


    try:
        df_eval = pd.read_excel(GITHUB_URL)
        st.success(f"📥 Dataset berhasil dimuat dari GitHub! ({df_eval.shape[0]} baris x {df_eval.shape[1]} kolom)")
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file dari GitHub: {e}")
        st.stop()

    if "HARGAPENAWARAN" not in df_eval.columns:
        st.error("Kolom target 'HARGAPENAWARAN' tidak ditemukan!")
        st.stop()

    # Persiapan data
    target = "HARGAPENAWARAN"
    X = df_eval.drop(columns=[target])
    y = df_eval[target]

    X = X.loc[:, X.nunique() > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=77
    )

    # Daftar model lengkap
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR(),
        'K-Nearest Neighbor': KNeighborsRegressor(),
        'Elastic Net': ElasticNet(random_state=77),
        'Passive Aggressive Regressor': PassiveAggressiveRegressor(random_state=77),
        'Random Forest': RandomForestRegressor(random_state=77),
        'Gradient Boosting': GradientBoostingRegressor(random_state=77),
        'LightGBM': LGBMRegressor(random_state=77),
        'XGBoost': xgb.XGBRegressor(random_state=77, verbosity=0)
    }

    results_eval = pd.DataFrame(columns=[
        'Model',
        'R2_in_sample', 'R2_out_sample',
        'MSE_in_sample', 'MSE_out_sample',
        'RMSE_in_sample', 'RMSE_out_sample',
        'MAE_in_sample', 'MAE_out_sample',
        'MAPE_in_sample', 'MAPE_out_sample'
    ])

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            mse_in  = mean_squared_error(y_train, y_train_pred)
            rmse_in = np.sqrt(mse_in)
            mae_in  = mean_absolute_error(y_train, y_train_pred)
            mape_in = mean_absolute_percentage_error(y_train, y_train_pred)
            r2_in   = r2_score(y_train, y_train_pred)

            mse_out  = mean_squared_error(y_test, y_test_pred)
            rmse_out = np.sqrt(mse_out)
            mae_out  = mean_absolute_error(y_test, y_test_pred)
            mape_out = mean_absolute_percentage_error(y_test, y_test_pred)
            r2_out   = r2_score(y_test, y_test_pred)

            results_eval = pd.concat([results_eval, pd.DataFrame([{
                'Model': name,
                'R2_in_sample': r2_in,
                'R2_out_sample': r2_out,
                'MSE_in_sample': mse_in,
                'MSE_out_sample': mse_out,
                'RMSE_in_sample': rmse_in,
                'RMSE_out_sample': rmse_out,
                'MAE_in_sample': mae_in,
                'MAE_out_sample': mae_out,
                'MAPE_in_sample': mape_in,
                'MAPE_out_sample': mape_out
            }])], ignore_index=True)
        except Exception as e:
            st.warning(f"⚠️ Model {name} gagal dijalankan: {e}")

    results_eval = results_eval.sort_values(by='RMSE_out_sample')
    st.subheader("Hasil Evaluasi Dasar Semua Model")
    st.dataframe(results_eval)


# ==========================
# Halaman 4: Prediksi
# ==========================
elif page == "Prediksi":
    st.title("💡 Prediksi Harga")
    
    if "model_rf" not in st.session_state:
        st.warning("⚠️ Model belum tersedia!")
    else:
        model_rf = st.session_state["model_rf"]

        # Load feature columns
        FEATURE_PATH = "model/feature_columns.joblib"
        if os.path.exists(FEATURE_PATH):
            feature_cols = joblib.load(FEATURE_PATH)
        else:
            st.warning("⚠️ Feature columns belum tersedia! Pastikan file 'feature_columns.joblib' ada di folder model/")
            st.stop()

        # Input variabel otomatis
        st.subheader("Input Variabel")
        input_data = {}
        for col in feature_cols:
            val = st.number_input(f"{col}", value=0.0)
            input_data[col] = val
        input_df = pd.DataFrame([input_data])

        # Prediksi harga
        if st.button("Prediksi Harga"):
            try:
                pred_harga = model_rf.predict(input_df)[0]
                st.success(f"💰 Prediksi Harga: {pred_harga:,.2f}")

                # Top-5 similarity
                scaler = StandardScaler()
                X_dummy = pd.DataFrame(np.random.rand(100, len(feature_cols)), columns=feature_cols)
                X_scaled = scaler.fit_transform(X_dummy)
                input_scaled = scaler.transform(input_df)
                sim_matrix = cosine_similarity(X_scaled, input_scaled)
                top5_idx = np.argsort(sim_matrix[:,0])[::-1][:5]
                st.subheader("Top-5 Similar Data Points (Index & Score)")
                top5_df = pd.DataFrame({
                    "Index": top5_idx,
                    "Similarity": sim_matrix[top5_idx,0]
                })
                st.dataframe(top5_df)
            except Exception as e:
                st.error(f"⚠️ Terjadi error saat prediksi: {e}")


