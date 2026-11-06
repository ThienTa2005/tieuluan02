"""LSTM dự báo AAPL: dữ liệu giá đóng cửa thực từ yfinance hoặc CSV.

Không sinh giá giả và không tự thay LSTM bằng Ridge dưới cùng một tên.
Chuẩn hoá bằng tập TRAIN để tránh rò rỉ giá từ tương lai.
"""
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

STOCK_PATH = Path(__file__).resolve().parents[1] / "data" / "aapl.csv"

def prices_from_csv(df):
    """Nhận CSV có Date/Close hoặc date/close; làm sạch và sắp ngày."""
    dates = next((c for c in df.columns if c.lower().strip() in ("date", "datetime")), None)
    close = next((c for c in df.columns if c.lower().strip() in ("close", "adj close")), None)
    if dates is None or close is None:
        raise ValueError("CSV giá cần có cột Date và Close (hoặc Adj Close).")
    dt = pd.to_datetime(df[dates], errors="coerce", utc=True).dt.tz_localize(None)
    vals = pd.to_numeric(df[close], errors="coerce")
    s = pd.Series(vals.to_numpy(dtype=float), index=dt, name="AAPL").dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s > 0]
    if len(s) < 100:
        raise ValueError("Cần ít nhất 100 phiên giá hợp lệ để train/test LSTM.")
    return s

def get_prices(ticker="AAPL", years=3):
    """Ưu tiên CSV cục bộ; nếu chưa có thử Yahoo Finance; cuối cùng báo thiếu."""
    if STOCK_PATH.is_file():
        try: return prices_from_csv(pd.read_csv(STOCK_PATH)), "CSV giá lịch sử thật (data/aapl.csv)"
        except (ValueError, KeyError): pass
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{years}y", progress=False, auto_adjust=True, threads=False)
        if df is not None and len(df) >= 100:
            close = df["Close"]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            tmp = pd.DataFrame({"Date": pd.to_datetime(close.index), "Close": close.to_numpy(dtype=float)})
            return prices_from_csv(tmp), "Yahoo Finance (lấy qua yfinance; dữ liệu thị trường thực)"
    except Exception:
        pass
    return None, "Chưa có giá thật: tải data/aapl.csv hoặc sử dụng tính năng upload CSV"

def _windows(x, w):
    if len(x) <= w: raise ValueError("Không đủ dữ liệu so với chiều dài cửa sổ.")
    return np.stack([x[i:i+w] for i in range(len(x)-w)]), x[w:]

if HAS_TORCH:
    class Net(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

def train_forecast(prices, horizon=7, window=30, epochs=15, hidden=32, seed=0):
    """Đánh giá holdout thời gian 15% cuối; dự báo đệ quy horizon phiên."""
    if not HAS_TORCH:
        raise RuntimeError("Chưa cài PyTorch; LSTM không khả dụng.")
    if prices is None: raise ValueError("Chưa có lịch sử giá AAPL thật.")
    if not 1 <= horizon <= 60 or not 5 <= window <= 120 or not 1 <= epochs <= 100:
        raise ValueError("Tham số horizon/window/epochs ngoài giới hạn.")
    v = np.asarray(prices, dtype=np.float64)
    if len(v) < max(100, window + 30) or not np.all(np.isfinite(v)) or (v <= 0).any():
        raise ValueError("Cần >=100 giá đóng cửa thực hợp lệ và đủ dài so với window.")
    Xraw, yraw = _windows(v, window)
    cut = int(len(Xraw) * 0.85)
    # Thống kê chỉ dùng các giá có mặt trong giai đoạn train, không biết tương lai.
    train_vals = v[:cut + window]
    lo, hi = float(train_vals.min()), float(train_vals.max())
    scale = max(hi - lo, 1e-9)
    Xtr = (Xraw[:cut] - lo)/scale; ytr = (yraw[:cut] - lo)/scale
    Xte = (Xraw[cut:] - lo)/scale; yte = yraw[cut:]
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    net = Net(hidden); opt = torch.optim.Adam(net.parameters(), lr=0.005); lossf = nn.MSELoss()
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    xt, yt = to_t(Xtr).unsqueeze(-1), to_t(ytr).unsqueeze(-1)
    hist = []
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(xt)); total = 0.
        for i in range(0, len(xt), 64):
            b = perm[i:i+64]
            opt.zero_grad(); loss = lossf(net(xt[b]), yt[b]); loss.backward(); opt.step()
            total += float(loss.item()) * len(b)
        hist.append(total / len(xt))
    net.eval()
    with torch.no_grad():
        pte = net(to_t(Xte).unsqueeze(-1)).squeeze(-1).numpy() * scale + lo
        seq = list((v[-window:] - lo) / scale); fc = []
        for _ in range(horizon):
            val = float(net(to_t(np.array(seq[-window:])).view(1, -1, 1)).item())
            fc.append(val); seq.append(val)
    rmse = float(np.sqrt(np.mean((pte - yte)**2)))
    pred = np.array(fc)*scale + lo
    dates = pd.bdate_range(pd.Timestamp(prices.index[-1]) + pd.Timedelta(days=1), periods=horizon)
    last = float(v[-1])
    # Biên sai số minh hoạ từ RMSE, KHÔNG phải khoảng tin cậy đã hiệu chuẩn.
    heuristic = rmse*np.sqrt(np.arange(1, horizon+1))
    return dict(backend="LSTM (PyTorch)", forecast=pd.Series(pred, index=dates),
                upper=pd.Series(pred+heuristic, index=dates), lower=pd.Series(pred-heuristic, index=dates),
                rmse=rmse, last=last, change_pct=float((pred[-1]/last-1)*100), loss_hist=hist,
                horizon=horizon, window=window, n_train=cut, n_test=len(Xte))
