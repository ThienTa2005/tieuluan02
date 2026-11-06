# Rà soát và bổ sung từ bản tieuluan02.zip đầu tiên

**Trạng thái:** Mã nguồn và kiểm thử Python; chưa xác nhận full UI Streamlit, Docker build hoặc deploy trên server thật.

1. Giữ 57 nút, 75 cạnh, 22 bản tóm lược có nguồn trong Knowledge Graph, bổ sung URL truy cập tài liệu gốc.
2. Giữ mini-Cypher + NetworkX; thêm giới hạn chỉ đọc cho truy vấn Neo4j trên giao diện, script đồng bộ KG bằng Neo4j riêng.
3. Giữ Vector RAG, GraphRAG, Hybrid; bổ sung embedding multilingual Sentence-Transformers tùy chọn và hiển thị rõ khi dùng TF-IDF dự phòng.
4. Thay Random Forest tự sinh dữ liệu mô phỏng thành đọc/tải/upload Pima thật; tiền xử lý thiếu dữ liệu, stratified holdout và cấm âm thầm điền input chưa được cung cấp trong Agent.
5. Thay dữ liệu giá AAPL mô phỏng thành Yahoo Finance/CSV thực; LSTM PyTorch thực, không dùng Ridge để giả kết quả LSTM; chuẩn hóa trên tập train, holdout thời gian.
6. Sửa benchmark: baseline LLM thuần SKIP khi không có API; câu tự nhập không giả có đáp án vàng; metric đặt tên là proxy, chỉ đo multi-hop ở câu gán nhãn ≥2 chặng.
7. Sửa câu trả lời có mã nguồn [M#]/[S#] xuyên suốt KG, RAG, Agent; xuất JSON/CSV.
8. Giao diện thêm upload Pima/AAPL, trạng thái lỗi rõ ràng, link nguồn và hướng dẫn trong dashboard.
9. Bổ sung Dockerfile, docker-compose, Render Blueprint (starter), GitHub Actions CI, README và bộ 30 test.
10. Không tạo URL GitHub/Render/Streamlit giả, không đóng gói API key; người dùng phải triển khai trên tài khoản của mình.
