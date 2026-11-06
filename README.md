# TIỂU LUẬN 02 — Knowledge Graph + Hybrid GraphRAG + Multi-Agent

> Dashboard Streamlit phục vụ **3 module chính**: **Y tế tiểu đường**, **Phân tích AAPL** và **Benchmark**. Hai trang kỹ thuật **Knowledge Graph** và **Multi-Agent** trình bày cấu phần để giảng viên có thể kiểm chứng. Ứng dụng có chế độ offline **trích xuất từ tài liệu**, khác với gọi LLM thật.

**URL public:** _Chưa triển khai; điền URL Streamlit Cloud/Render sau khi deploy._  
**GitHub repo:** _Chưa có; điền liên kết sau khi push mã nguồn._

## 1. Chức năng và vị trí mã nguồn

| # | Yêu cầu | Triển khai / nơi xem |
|---|---|---|
| 1 | 3 module dashboard | `app.py`, thanh điều hướng Y tế / Chứng khoán / Benchmark (+ 2 trang kỹ thuật) |
| 2 | KG y tế + Apple, Cypher, NetworkX, đa chặng | `src/kg.py`, tab **Knowledge Graph**; Neo4j tùy chọn |
| 3 | Vector RAG, GraphRAG, Hybrid | `src/vector.py`, `src/rag.py` (xếp hạng hợp nhất với trọng số α) |
| 4 | Chatbot tiểu đường có chứng cứ | `src/corpus.py`, `src/llm.py`, tab Y tế → Chatbot |
| 5 | Clinical ML Random Forest Pima | `src/ml.py`, Y tế → Random Forest; chỉ học từ CSV Pima thật |
| 6 | AAPL chuỗi cung ứng/sự kiện | `src/kg.py`, `src/corpus.py`, Chứng khoán → Chatbot/Sự kiện |
| 7 | Dự báo LSTM AAPL | `src/lstm.py`, Chứng khoán → LSTM; dữ liệu giá thực từ Yahoo hoặc CSV |
| 8 | Orchestrator 4 công cụ | `src/agent.py`: KG Query, Vector Search, Clinical ML Predictor, Stock LSTM Forecaster |
| 9 | ReAct nhật ký, nguồn, đồ thị con | trang Multi-Agent → 4 tab, xuất JSON |
| 10 | Benchmark 4 kiến trúc RAG | `src/benchmark.py`, 8 câu chuẩn + 1 câu thêm; CSV/JSON |
| 11 | Điều chỉnh truy xuất, nhập câu hỏi | Sidebar top-k, hops, α, lựa chọn phương pháp; nút chạy / xuất kết quả |

> **Phân biệt thuật ngữ:** Agent là **ReAct theo policy luật** (các bước Thought/Action/Observation xác định), không phải 4 tác tử LLM tự lập kế hoạch độc lập. Vector mặc định là **TF-IDF n-gram từ vựng**, không phải embedding semantic; có cấu hình embedding đa ngôn ngữ tùy chọn. Trong môi trường không có API key, chatbot là **extractive QA**, không phải LLM thực.

## 2. Cấu trúc

```text
app.py                        # Dashboard 6 trang (3 chính + 3 kỹ thuật/tổng quan)
src/corpus.py                 # 22 đoạn tài liệu tóm lược + URL nguồn từng đoạn
src/kg.py                     # 57 nút, 75 cạnh, mini-Cypher, Neo4j, multi-hop
src/vector.py                 # Vector TF-IDF / multilingual embedding tùy chọn
src/rag.py                    # Vector / Graph / Hybrid Graph-Vector RAG
src/llm.py                    # Gọi Anthropic thật hoặc trả lời trích xuất offline
src/ml.py                     # Huấn luyện Random Forest với CSV Pima thật
src/lstm.py                   # LSTM PyTorch với giá thực, holdout theo thời gian
src/agent.py                  # Điều phối 4 công cụ và nhật ký ReAct
src/benchmark.py              # 8 câu hỏi có nhãn và các chỉ số proxy
src/viz.py                    # Đồ thị NetworkX tương tác qua Plotly
scripts/download_data.py      # Tải & kiểm tra bộ Pima công khai 768 dòng
scripts/sync_neo4j.py         # Đồng bộ KG lên Neo4j (cần quyền WRITE)
tests/                       # Kiểm thử tự động lõi hệ thống
Dockerfile, render.yaml       # Deploy Render qua Docker
.streamlit/config.toml        # Giao diện Streamlit
```

## 3. Chạy thử trên máy (Python 3.11 khuyến nghị)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
streamlit run app.py
```

Mở `http://localhost:8501`. Lệnh tải Pima yêu cầu mạng và chỉ lấy **bản mirror có 768 hàng**; script kiểm tra đủ 9 cột và hai lớp 0/1. Nếu máy không tải được, tải từ [Pima public mirror](https://github.com/Zue77/Pima-Indians-Diabetes-Dataset/blob/main/diabetes.csv), đặt đúng tên `data/diabetes.csv` **hoặc upload CSV trên tab Y tế**. **Không** thay bằng dữ liệu ngẫu nhiên để trình bày accuracy.

### Giá AAPL thật

- Ứng dụng tự thử lấy 3 năm giá qua `yfinance` (đã trong `requirements.txt`). Yahoo Finance có thể chặn truy cập ở vài môi trường.
- Khi Yahoo không trả giá, **upload CSV có `Date` và `Close`** vào tab Chứng khoán hoặc đặt file ở `data/aapl.csv`. Ít nhất 100 phiên dữ liệu hợp lệ. Không tự sinh chuỗi giá giả.
- LSTM huấn luyện theo thứ tự thời gian: đầu 85% **cửa sổ** là train, 15% cuối là test. Min/max chỉ dùng từ train. RMSE là kiểm tra nội bộ, không khẳng định hiệu quả giao dịch. Dải ±RMSE×√h chỉ là *minh họa*, không phải 95% CI.

### Các bước demo cho giảng viên

1. **Y tế:** hỏi “Metformin chống chỉ định khi nào?”, mở danh sách nguồn và đồ thị con. Chuyển Vector RAG → GraphRAG → Hybrid và thay α để so sánh.
2. **Knowledge Graph:** chạy mini-Cypher `MATCH (a:Drug)-[r:CONTRAINDICATED_IN]->(b) RETURN a.name, type(r), b.name` và kiểm tra đường 2–3 chặng.
3. **Y tế ML:** có Pima CSV thật → nhập 8 số đo → xem P(Outcome=1), AUC/F1/Accuracy và biến quan trọng. Chỉ là phân loại học thuật.
4. **AAPL:** có lịch sử giá thật → mở LSTM → xem giá, dự báo, loss, RMSE, xuất CSV.
5. **Multi-Agent:** nhập “Dự báo AAPL tuần tới và thuế quan tác động đến Apple?” → kiểm tra đúng 4 công cụ đăng ký và trace; nếu thiếu dữ liệu giá phải có lý do trong observation.
6. **Benchmark:** chọn 8 câu + các phương pháp, xuất CSV/JSON, trình bày rõ các điểm **SKIP/NA**.

## 4. Embedding ngữ nghĩa, LLM và Neo4j (tùy chọn)

**Embedding:**

```bash
pip install -r requirements-semantic.txt
# Windows PowerShell:
$env:EMBEDDING_BACKEND="semantic"
# Linux/macOS: export EMBEDDING_BACKEND=semantic
streamlit run app.py
```

Mặc định để `EMBEDDING_BACKEND=tfidf` cho gói RAM thấp. Nếu chọn `semantic`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` sẽ cần tải model lần đầu; nếu tải thất bại, UI **báo lỗi**, không âm thầm tự nhận là semantic. `auto` mới cho phép fallback về TF-IDF có nhãn rõ.

**LLM thật và Neo4j:**

```bash
pip install -r requirements-optional.txt
# Cấu hình qua env hoặc Streamlit Secrets
# ANTHROPIC_API_KEY=...  (không commit key)
# LLM_MODEL=claude-sonnet-4-5
# NEO4J_URI=neo4j+s://...
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=...
# Nạp KG: python -m scripts.sync_neo4j (cần quyền WRITE một lần)
```

Khi `NEO4J_URI` được thiết lập, tab Cypher chuyển sang đọc trên Neo4j; tài khoản dùng trong app nên chỉ có quyền READ. Nếu không có Neo4j, mini-Cypher chạy trên NetworkX và chỉ hỗ trợ tập con `MATCH ... RETURN ... [LIMIT]`. KG vẫn lưu ở NetworkX phục vụ thuật toán RAG.

> API LLM thật là **tùy chọn**. Không có API key, app vẫn hỏi đáp bằng trích xuất văn bản, nhưng baseline “LLM thuần” được **SKIP** trong benchmark, không dùng mô phỏng để giả đo kết quả.

## 5. Deploy công khai

### A. Streamlit Community Cloud
1. Push toàn bộ **nội dung thư mục `tieuluan02`** lên repo GitHub (không push các secret).
2. Mở https://share.streamlit.io → **Create app** → repo và nhánh chính → `app.py`.
3. Đặt Python 3.11 (xem `.python-version`) nếu nền tảng cho chọn, cài từ `requirements.txt`.
4. Tùy chọn nhập **Secrets** trong dashboard: `ANTHROPIC_API_KEY = "..."` và/hoặc `NEO4J_*`.
5. Nếu thiếu Pima, upload CSV trong ứng dụng hoặc chạy downloader tại máy rồi commit CSV (kiểm tra quyền sử dụng dữ liệu). Nếu Yahoo bị chặn, upload AAPL CSV.
6. Copy URL được cấp và ghi vào mục đầu README. Nên dùng bản TF-IDF mặc định cho Cloud vì model embedding và LSTM tiêu tốn RAM.

### B. Render qua Docker
1. Push repo GitHub. Kết nối Render, chọn **Blueprint** với `render.yaml` hoặc Web Service chọn Dockerfile.
2. `render.yaml` mặc định `plan: starter` vì PyTorch và Streamlit có thể vượt giới hạn gói free. Điều chỉnh plan theo tài nguyên tài khoản.
3. Cấu hình các biến môi trường tùy chọn trong Dashboard; không đặt key trong GitHub.
4. Chờ endpoint `/_stcore/health` báo thành công; mở trang được cấp. Cấu hình file CSV / upload tương tự mục A.

**Không có URL/GitHub công khai sẵn trong ZIP**: hai liên kết chỉ xuất hiện sau khi triển khai trên tài khoản GitHub, Render hoặc Streamlit của bạn. Không ghi URL giả.

## 6. Đánh giá khoa học & giới hạn

- **Dữ liệu tri thức:** 57 nút, 75 cạnh, 22 tài liệu **tóm lược**. Nguồn tham khảo có link ở `src/corpus.py`, không tự động thu thập/cập nhật tài liệu gốc. Một số mối quan hệ Apple là khái quát lịch sử/giả thuyết tác động, không phải dự báo sự kiện tương lai.
- **Random Forest Pima:** tiền xử lý 0→missing cho Glucose/BloodPressure/SkinThickness/Insulin/BMI, impute median chỉ học trên train; chia stratified 75/25 seed 42; không huấn luyện trên dữ liệu giả. Vì mẫu Pima có tính chọn lọc, không dùng để chẩn đoán cá nhân.
- **LSTM:** dữ liệu thị trường thực có thể thay đổi mỗi lần tải; kết quả thay đổi theo giai đoạn giá, số epoch và máy chạy. Dùng holdout theo thời gian, không cung cấp lời khuyên mua/bán.
- **Benchmark:** 8 câu có nhãn từ khóa/thực thể do nhóm quy định. `quality` là keyword recall trong câu trả lời, `multihop` là độ phủ thực thể trong ngữ cảnh của **câu gán nhãn ≥2 chặng**, `grounding` là overlap từ vựng với ngữ cảnh (proxy yếu, đặc biệt cao ở chế độ extractive); `latency` là thời gian thực thi. Không gọi đây là factual accuracy/hallucination rate.
- **Công cụ agent:** ReAct theo luật, các Thought là lời giải thích tuyến xử lý đã lập trình, không phải reasoning ẩn của LLM.
- **Dữ liệu Pima:** nguồn mirror [GitHub Pima CSV](https://github.com/Zue77/Pima-Indians-Diabetes-Dataset/blob/main/diabetes.csv); cần kiểm tra giấy phép/nguồn phù hợp khi chia sẻ công khai.
- **Rủi ro và bảo mật:** Tránh đưa thông tin sức khỏe có định danh, khóa API và thông tin đầu tư riêng vào dịch vụ bên thứ ba. Neo4j dùng tài khoản READ cho runtime.

## 7. Kiểm thử tự động

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q app.py src scripts
```

Test tập trung vào KG/Cypher, kết quả RAG, trích nguồn, Agent thiếu dữ liệu, Benchmark không giả lập LLM, kiểm tra phân chia dữ liệu RF và LSTM trên **fixture do test tự tạo** (fixture này không được dùng cho ứng dụng thực tế).

### Chạy Docker local

```bash
docker compose up --build
# http://localhost:8501
```

Có thể gắn CSV thật vào `data/` (docker-compose đã mount thư mục này read-only). Trong dashboard, upload vẫn sử dụng phiên Streamlit.

File `CHANGELOG.md` liệt kê các thay đổi so với ZIP gốc; `.github/workflows/tests.yml` kiểm tra tự động sau khi push GitHub.
