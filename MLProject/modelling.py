import pandas as pd
import sys
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================================================
# 1. SETUP PATH & LOAD DATA (LOAD PROCESSED CSVs)
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# data folder inside this project
data_dir = os.path.join(current_dir, 'heartdisease_preprocessing')
train_csv = os.path.join(data_dir, 'train_processed.csv')
test_csv = os.path.join(data_dir, 'test_processed.csv')

# Validate files exist
if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    sys.exit(f"File train/test CSV tidak ditemukan di: {data_dir}")

print(f"Loading train data from: {train_csv}")
print(f"Loading test data from : {test_csv}")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Dataset sudah diproses (one-hot, scaling, dst.) — target di kolom 'num'
target_col = 'num'
if target_col not in df_train.columns or target_col not in df_test.columns:
    sys.exit(f"Kolom target '{target_col}' tidak ditemukan di CSV")

X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col].astype(int)
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col].astype(int)

# ==============================================================================
# 2. EXPERIMENT SETUP
# ==============================================================================
mlflow.set_experiment("Eksperimen_Heart_Disease_Basic")

# ==============================================================================
# 3. TRAINING DENGAN PURE AUTOLOG (Syarat Basic)
# ==============================================================================
# Kita aktifkan autolog TANPA parameter log_models=False.
# Biarkan dia default (True), jadi dia otomatis simpan model, param, metric.
mlflow.sklearn.autolog()

print("Mulai training model basic (Pure Autolog)...")

with mlflow.start_run(run_name="Basic_Training_Autolog_Only"):
    
    # Inisialisasi Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Proses Training (Fit)
    # Di sinilah MAGIC autolog bekerja. Dia akan otomatis mencatat:
    # - Parameters (n_estimators, random_state, dll)
    # - Metrics (Training score)
    # - Artifacts (Model.pkl, conda.yaml, MLmodel)
    rf.fit(X_train, y_train)
    
    # Evaluasi (Opsional, hanya untuk print di terminal)
    # Autolog biasanya otomatis menghitung metrics training, 
    # tapi untuk test metrics kadang perlu manual atau eval_and_log_metrics.
    # Namun untuk syarat 'Basic', kode di atas sudah cukup memicu autolog.
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")
    
    print("Selesai. Semua log ditangani otomatis oleh MLflow Autolog.")
