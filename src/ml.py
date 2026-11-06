"""Huấn luyện Random Forest trên dữ liệu Pima thật, KHÔNG sinh nhãn giả.

Đây là mô hình phân loại dữ liệu nghiên cứu. Xác suất không phải nguy cơ
cá nhân được kiểm định lâm sàng; Pima không đại diện cho mọi nhóm dân số.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
LABELS = {"Pregnancies": "Số lần mang thai", "Glucose": "Đường huyết (mg/dL)", "BloodPressure": "Huyết áp tâm trương (mmHg)",
          "SkinThickness": "Độ dày nếp da (mm)", "Insulin": "Insulin huyết thanh (µU/mL)", "BMI": "BMI (kg/m²)",
          "DiabetesPedigreeFunction": "Chỉ số tiền sử gia đình (DPF)", "Age": "Tuổi"}
DEFAULTS = {"Pregnancies": 2, "Glucose": 110, "BloodPressure": 72, "SkinThickness": 23, "Insulin": 80,
            "BMI": 28.0, "DiabetesPedigreeFunction": 0.45, "Age": 35}
# Một số trường bằng 0 trong Pima là giá trị thiếu, không có nghĩa là đo được 0.
ZERO_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
DATA_SOURCE = "Pima Indians Diabetes (768 bản ghi; bộ dữ liệu học thuật, không phải dữ liệu lâm sàng đại diện)"

def load_data(path=None):
    """Đọc CSV thật; lỗi rõ ràng nếu dữ liệu chưa có / không đúng định dạng."""
    path = Path(path) if path else DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy python scripts/download_data.py hoặc tải CSV Pima lên giao diện.")
    return validate_data(pd.read_csv(path))

def validate_data(df):
    missing = sorted(set(FEATURES + ["Outcome"]) - set(df.columns))
    if missing:
        raise ValueError(f"Thiếu cột CSV Pima: {', '.join(missing)}")
    d = df[FEATURES + ["Outcome"]].copy()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Outcome"])
    if len(d) < 40 or set(d.Outcome.unique()) != {0, 1}:
        raise ValueError("Cần ≥40 hàng và nhãn Outcome gồm cả 0 và 1.")
    if (d[FEATURES].notna().sum(axis=1) == 0).any():
        raise ValueError("Có bản ghi không chứa chỉ số hợp lệ.")
    d["Outcome"] = d["Outcome"].astype(int)
    for f in ZERO_MISSING:
        d.loc[d[f] == 0, f] = np.nan
    return d.reset_index(drop=True)

def train_rf(n_trees=200, seed=42, dataframe=None):
    """Chia stratified train/test trước khi impute để tránh rò rỉ dữ liệu."""
    d = validate_data(dataframe) if dataframe is not None else load_data()
    Xtr, Xte, ytr, yte = train_test_split(d[FEATURES], d["Outcome"], test_size=0.25, random_state=seed,
                                           stratify=d["Outcome"])
    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("rf", RandomForestClassifier(n_estimators=n_trees, max_depth=8, min_samples_leaf=3,
                                                    random_state=seed, n_jobs=1))])
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    return dict(model=model, source=DATA_SOURCE if dataframe is None else "Pima CSV do người dùng tải lên",
                auc=float(roc_auc_score(yte, proba)), acc=float(accuracy_score(yte, proba >= 0.5)),
                f1=float(f1_score(yte, proba >= 0.5)),
                ranges={f: (float(d[f].min()), float(d[f].max())) for f in FEATURES},
                medians={f: float(d[f].median()) for f in FEATURES}, n=len(d), n_test=len(Xte))

def predict_risk(bundle, values):
    """Chỉ suy luận khi đủ 8 chỉ số; không âm thầm điền giá trị cho người dùng."""
    if bundle is None:
        raise ValueError("Mô hình chưa huấn luyện trên bộ dữ liệu Pima thật.")
    missing = [f for f in FEATURES if f not in values or values[f] is None]
    if missing:
        raise ValueError("Thiếu chỉ số: " + ", ".join(missing))
    row = {f: float(values[f]) for f in FEATURES}
    if any(not np.isfinite(v) or v < 0 for v in row.values()):
        raise ValueError("Tất cả chỉ số phải là số hữu hạn không âm.")
    for f in ZERO_MISSING:
        if row[f] == 0:
            row[f] = np.nan  # 0 là ký hiệu thiếu trong tập dữ liệu Pima
    p = float(bundle["model"].predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1])
    warn = [f"{LABELS[f]} = {row[f]} ngoài khoảng dữ liệu huấn luyện {bundle['ranges'][f][0]:g}–{bundle['ranges'][f][1]:g}"
            for f in FEATURES if np.isfinite(row[f]) and not bundle["ranges"][f][0] <= row[f] <= bundle["ranges"][f][1]]
    for f in ZERO_MISSING:
        if np.isnan(row[f]):
            warn.append(f"{LABELS[f]} = 0: được xem là thiếu và thay bằng trung vị tập huấn luyện")
    importances = bundle["model"].named_steps["rf"].feature_importances_
    return dict(prob=p, level="Dưới 0,5" if p < 0.5 else "Từ 0,5 trở lên", warnings=warn,
                importance=dict(zip(FEATURES, (float(x) for x in importances))), inputs=row)
