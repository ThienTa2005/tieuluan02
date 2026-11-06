"""Tải bộ dữ liệu Pima từ bản mirror công khai, kiểm tra schema và số hàng.

Chạy: python scripts/download_data.py
Nguồn: https://github.com/Zue77/Pima-Indians-Diabetes-Dataset/blob/main/diabetes.csv
Bộ Pima gốc có 768 dòng; nếu tải lỗi không tự sinh CSV giả.
"""
from pathlib import Path
from urllib.request import urlopen, Request
import io
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/Zue77/Pima-Indians-Diabetes-Dataset/main/diabetes.csv"
EXPECTED = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

def main():
    try:
        req = Request(URL, headers={"User-Agent": "Tiểu luận 02 educational project"})
        with urlopen(req, timeout=25) as response: content = response.read(500_000)
        frame = pd.read_csv(io.BytesIO(content))
        if list(frame.columns) != EXPECTED or len(frame) != 768 or set(frame.Outcome.unique()) != {0, 1}:
            raise ValueError(f"Dataset trả về sai cấu trúc: {frame.shape}")
        path = ROOT / "data" / "diabetes.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Đã tải {len(frame)} bản ghi Pima thật → {path}")
    except Exception as exc:
        print(f"Không tải được dữ liệu: {exc}. Có thể đặt CSV thật vào data/diabetes.csv hoặc upload trên giao diện.", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__": main()
