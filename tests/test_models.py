"""Fixture tự tạo chỉ nằm trong TESTS; không được dùng làm kết quả UI."""
import numpy as np
import pandas as pd
import pytest
from src import ml, lstm


def fake_pima(n=120):
    rng=np.random.default_rng(42)
    data={f:rng.uniform(1,100,n) for f in ml.FEATURES}
    data["Pregnancies"]=rng.integers(0,6,n)
    data["Outcome"]=np.array([0,1]*(n//2))
    return pd.DataFrame(data)

def test_load_data_requires_real_file(tmp_path):
    with pytest.raises(FileNotFoundError): ml.load_data(tmp_path/"missing.csv")

def test_validation_rejects_missing_col():
    with pytest.raises(ValueError): ml.validate_data(fake_pima().drop(columns=["Glucose"]))

def test_validation_zero_missing():
    f=fake_pima()
    f.loc[0,"Insulin"]=0
    assert np.isnan(ml.validate_data(f).loc[0,"Insulin"])

def test_rf_train_and_prediction():
    bundle=ml.train_rf(n_trees=12,dataframe=fake_pima())
    assert bundle["n_test"]==30
    assert 0<=bundle["auc"]<=1
    p=ml.predict_risk(bundle,dict(ml.DEFAULTS))
    assert 0<=p["prob"]<=1
    assert len(p["importance"])==8

def test_rf_requires_all_inputs():
    bundle=ml.train_rf(n_trees=8,dataframe=fake_pima())
    with pytest.raises(ValueError,match="Thiếu chỉ số"):
        ml.predict_risk(bundle,{"Glucose":140})

def test_csv_parse_stock():
    dates=pd.bdate_range("2024-01-01",periods=120)
    df=pd.DataFrame({"Date":dates,"Close":np.linspace(170,210,120)})
    series=lstm.prices_from_csv(df)
    assert len(series)==120 and series.iloc[-1]==210

def test_stock_rejects_too_short():
    with pytest.raises(ValueError,match="100 phiên"):
        lstm.prices_from_csv(pd.DataFrame({"Date":["2025-01-01"],"Close":[10]}))

@pytest.mark.skipif(not lstm.HAS_TORCH,reason="Torch chưa cài")
def test_lstm_train_holdout():
    n=120
    series=pd.Series(180+np.linspace(0,24,n)+np.sin(np.arange(n)/7),index=pd.bdate_range("2025-01-01",periods=n))
    result=lstm.train_forecast(series,horizon=3,window=12,epochs=1,hidden=8)
    assert result["backend"]=="LSTM (PyTorch)"
    assert len(result["forecast"])==3 and result["n_test"]>0
    assert np.isfinite(result["rmse"])
