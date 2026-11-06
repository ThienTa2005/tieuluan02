# Dữ liệu đầu vào

- `diabetes.csv`: Pima Indians Diabetes, 768 bản ghi/9 cột; tải bằng `python scripts/download_data.py` hoặc upload trên tab Y tế. Không tạo dữ liệu giả.
- `aapl.csv`: giá AAPL thật theo thời gian có cột `Date,Close`; không bắt buộc nếu yfinance lấy thành công. Cũng có thể upload trên tab Chứng khoán.

Hai file CSV không được dựng ngẫu nhiên và không có sẵn trong gói. Nếu không có nguồn thực, dashboard báo trạng thái thiếu dữ liệu và không tính kết quả ML/LSTM.
