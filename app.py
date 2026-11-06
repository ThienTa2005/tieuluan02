"""Tiểu luận 02: Knowledge Graph, GraphRAG, Agent và Benchmark.

Các đầu vào ML/thị trường phải lấy từ dữ liệu có thật. Không dùng số liệu
mô phỏng làm kết quả trong UI. Giao diện chính bằng Streamlit.
"""
from io import BytesIO
import hashlib
import json
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.kg import build_graph, run_cypher, multi_hop_paths, expand_subgraph
from src.corpus import DOCS
from src.vector import VectorIndex
from src import rag, agent, benchmark, llm
from src.ml import train_rf, predict_risk, FEATURES, LABELS, DEFAULTS, DATA_PATH
from src.lstm import get_prices, prices_from_csv, train_forecast, HAS_TORCH
from src.viz import plot_graph

st.set_page_config(page_title="Tiểu luận 02 | Knowledge Graph & Multi-Agent", page_icon="🧠", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.7rem;padding-bottom:3rem}
[data-testid="stMetric"]{border:1px solid #bfc8d655;border-radius:11px;padding:12px}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Đang dựng đồ thị và chỉ mục vector...")
def load_knowledge():
    return build_graph(), VectorIndex(DOCS)

@st.cache_resource(show_spinner="Đang huấn luyện Random Forest trên Pima...")
def load_rf():
    try: return train_rf()
    except FileNotFoundError: return None

@st.cache_resource(show_spinner="Đang huấn luyện CSV Pima được upload...")
def rf_from_bytes(content):
    return train_rf(dataframe=pd.read_csv(BytesIO(content)))

@st.cache_data(ttl=3600, show_spinner="Đang lấy giá AAPL thực...")
def load_prices():
    return get_prices("AAPL")

@st.cache_resource(show_spinner="Đang huấn luyện LSTM và kiểm tra RMSE...")
def forecast_cached(prices, horizon, window, epochs):
    return train_forecast(prices, horizon=horizon, window=window, epochs=epochs)

G, VI = load_knowledge()
S = st.session_state
RF = S["rf_uploaded"] if "rf_uploaded" in S else load_rf()
if "prices_uploaded" in S:
    PRICES, PRICE_SRC = S["prices_uploaded"], "CSV AAPL thật do người dùng upload"
else:
    PRICES, PRICE_SRC = load_prices()
S.setdefault("lstm_window", 30)
S.setdefault("lstm_epochs", 15)

def forecast_for_agent(horizon):
    if PRICES is None or not HAS_TORCH: raise ValueError("Chưa có PyTorch hoặc giá AAPL thật.")
    return forecast_cached(PRICES, horizon, S["lstm_window"], S["lstm_epochs"])

CTX = dict(G=G, vi=VI, docs=DOCS, rf=RF,
           forecast=forecast_for_agent if PRICES is not None and HAS_TORCH else None)

with st.sidebar:
    st.title("🧠 Tiểu luận 02")
    page = st.radio("Điều hướng", ["🏠 Tổng quan", "🩺 Y tế tiểu đường", "📈 Chứng khoán AAPL", "🕸️ Knowledge Graph", "🤖 Multi-Agent", "📊 Benchmark"])
    st.divider()
    st.subheader("Tuỳ chọn RAG")
    METHOD = st.selectbox("Phương pháp", list(rag.METHODS) + ["Agentic Hybrid RAG"], index=2)
    TOPK = st.slider("Top-k", 1, 8, 4)
    HOPS = st.slider("Số chặng đồ thị", 1, 4, 2)
    ALPHA = st.slider("α – trọng số vector", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"Vector: {VI.detail}")
    st.caption("LLM: " + ("API đã cấu hình" if llm.has_llm() else "Trích xuất offline (không phải LLM thật)"))
    st.caption(f"Pima: {'đã huấn luyện dữ liệu thật' if RF is not None else 'chưa có dữ liệu'}")
    st.caption(f"LSTM: {'sẵn sàng' if PRICES is not None and HAS_TORCH else 'chưa đủ giá hoặc PyTorch'}")


def run_method(question, domain=None):
    """Dùng đúng tham số truy xuất sidebar với phương pháp được chọn."""
    if METHOD == "Agentic Hybrid RAG":
        return agent.run(CTX, question, k=TOPK, hops=HOPS, alpha=ALPHA, domain=domain)
    return rag.answer(CTX, METHOD, question, k=TOPK, hops=HOPS, alpha=ALPHA, domain=domain)


def show_passages(passages):
    for p in passages:
        score = p.get("fused", p.get("score", 0))
        st.markdown(f"**[{p['id']}] {p['title']}** · [{p['source']}]({p['url']}) · điểm `{score:.3f}`")
        st.caption(p["text"])


def show_result(result, key):
    st.markdown(result["answer"])
    st.caption(f"{result['method']} · {result['latency']*1000:.0f} ms")
    with st.expander(f"📚 Nguồn & liên kết kiểm chứng ({len(result['passages'])})"):
        show_passages(result["passages"])
    if result["facts"]:
        with st.expander(f"🔗 Quan hệ KG ({len(result['facts'])})"):
            st.write("\n".join(f"- {f}" for f in result["facts"]))
    if result["nodes"]:
        with st.expander("🕸️ Đồ thị con tương ứng"):
            st.plotly_chart(plot_graph(G, result["nodes"], result["edges"], result["seeds"], height=430),
                            use_container_width=True, key=f"subgraph_{key}")


def chat_panel(name, domain, examples):
    history = S.setdefault(name, [])
    columns = st.columns(len(examples))
    for col, ex in zip(columns, examples):
        if col.button(ex, use_container_width=True, key=f"{name}_example_{ex}"):
            S[f"{name}_pending"] = ex
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user": st.write(msg["content"])
            else: show_result(msg["result"], f"{name}_{i}")
    question = st.chat_input("Hỏi Knowledge Graph/RAG/Agent...", key=f"{name}_chat") or S.pop(f"{name}_pending", None)
    if question:
        history.append(dict(role="user", content=question))
        try:
            result = run_method(question, domain)
            history.append(dict(role="assistant", result=result))
        except Exception as exc:
            st.error(f"Không xử lý được câu hỏi: {exc}")
        st.rerun()


def upload_csv(kind):
    """Upload CSV Pima/giá thực. Lưu bản huấn luyện trên phiên làm việc."""
    if kind == "pima":
        file = st.file_uploader("Tải CSV Pima thật (9 cột chuẩn)", type="csv", key="pima_upload")
    else:
        file = st.file_uploader("Tải lịch sử giá AAPL thật (Date, Close)", type="csv", key="aapl_upload")
    if file is None: return
    raw = file.getvalue(); digest = hashlib.sha256(raw).hexdigest()
    if S.get(f"{kind}_digest") == digest: return
    try:
        if kind == "pima":
            S["rf_uploaded"] = rf_from_bytes(raw)
        else:
            S["prices_uploaded"] = prices_from_csv(pd.read_csv(BytesIO(raw)))
        S[f"{kind}_digest"] = digest
        st.rerun()
    except Exception as exc:
        st.error(f"CSV không hợp lệ: {exc}")


# ====================== TỔNG QUAN ======================
if page == "🏠 Tổng quan":
    st.title("Knowledge Graph · Hybrid GraphRAG · Multi-Agent")
    st.write("Dashboard nghiên cứu kiến thức bệnh tiểu đường, chuỗi cung ứng và dự báo AAPL.")
    metrics = st.columns(5)
    metrics[0].metric("Thực thể KG", G.number_of_nodes())
    metrics[1].metric("Quan hệ KG", G.number_of_edges())
    metrics[2].metric("Tài liệu có nguồn", len(DOCS))
    metrics[3].metric("Mô hình Diabetes", "RF" if RF is not None else "Chờ dữ liệu")
    metrics[4].metric("Công cụ Agent", len(agent.TOOLS))
    st.info("Dữ liệu nguồn là bản tóm lược học thuật. Nếu chưa có Pima/AAPL thật, hai mô hình sẽ không tự tạo dữ liệu giả.")
    st.markdown("**Ba module chính:** Y tế tiểu đường · Chứng khoán AAPL · Benchmark. Các trang KG và Multi-Agent cho phép kiểm tra riêng các thành phần kiến trúc.")
    st.plotly_chart(plot_graph(G, list(G.nodes), height=600, show_labels=False), use_container_width=True)

# ====================== Y TẾ ======================
elif page == "🩺 Y tế tiểu đường":
    st.title("🩺 Y tế tiểu đường")
    st.caption("Mục đích nghiên cứu và hỗ trợ truy vấn tài liệu; không chẩn đoán hoặc khuyến nghị dùng thuốc.")
    tab_chat, tab_ml = st.tabs(["💬 Chatbot có dẫn nguồn", "🧪 Random Forest trên Pima"])
    with tab_chat:
        chat_panel("medical_chat", "medical", ["Triệu chứng tiểu đường type 2?", "Metformin chống chỉ định khi nào?", "Tiểu đường có biến chứng thận?"])
    with tab_ml:
        upload_csv("pima")
        if RF is None:
            st.warning("Chưa có dữ liệu Pima thật. Chạy `python scripts/download_data.py` để tạo `data/diabetes.csv`, rồi chạy lại ứng dụng; hoặc upload CSV ngay bên trên.")
            st.markdown("[Mở nguồn dữ liệu Pima (768 dòng)](https://github.com/Zue77/Pima-Indians-Diabetes-Dataset/blob/main/diabetes.csv)")
        else:
            st.success(f"Đang dùng: {RF['source']} | {RF['n']} mẫu | tập test {RF['n_test']} mẫu | Accuracy {RF['acc']:.3f} · F1 {RF['f1']:.3f} · AUC {RF['auc']:.3f}")
            st.caption("Bộ Pima là dữ liệu nghiên cứu cho phụ nữ thuộc quần thể cụ thể, không xác thực mô hình cho mọi bệnh nhân.")
            cols = st.columns(4); values = {}
            for i, f in enumerate(FEATURES):
                values[f] = cols[i % 4].number_input(LABELS[f], min_value=0.0, value=float(DEFAULTS[f]),
                                                     step=0.1 if f in ("BMI", "DiabetesPedigreeFunction") else 1.0,
                                                     key=f"rf_{f}")
            if st.button("Chạy phân loại trên 8 chỉ số", type="primary"):
                try:
                    result = predict_risk(RF, values)
                    st.metric("P(Outcome=1) theo Random Forest", f"{result['prob']:.1%}")
                    importance = pd.Series(result["importance"]).rename(index=LABELS).sort_values()
                    st.plotly_chart(px.bar(importance, orientation="h", title="Feature importance (RF)"), use_container_width=True)
                    for note in result["warnings"]: st.warning(note)
                    st.warning("Xác suất phân loại mô hình không phải chẩn đoán hoặc xác suất mắc bệnh cá nhân được hiệu chuẩn lâm sàng.")
                except ValueError as exc:
                    st.error(str(exc))

# ====================== CHỨNG KHOÁN ======================
elif page == "📈 Chứng khoán AAPL":
    st.title("📈 Phân tích AAPL")
    tab_forecast, tab_chat, tab_events = st.tabs(["🔮 LSTM & chuỗi giá thực", "💬 Trợ lý chuỗi cung ứng", "⚡ Tác động sự kiện"])
    with tab_forecast:
        upload_csv("aapl")
        st.caption(f"Nguồn dữ liệu: {PRICE_SRC}")
        if PRICES is None:
            st.warning("Không có chuỗi AAPL thật (không truy cập được Yahoo Finance và chưa upload CSV). Tải CSV Date/Close từ nguồn thị trường có quyền sử dụng rồi upload ở trên.")
        elif not HAS_TORCH:
            st.error("Chưa cài PyTorch nên không thể chạy LSTM.")
        else:
            c = st.columns(4)
            horizon = c[0].slider("Số phiên dự báo", 1, 30, 7)
            S["lstm_window"] = c[1].slider("Window", 10, 60, S["lstm_window"])
            S["lstm_epochs"] = c[2].slider("Epoch", 1, 50, S["lstm_epochs"])
            show_days = c[3].selectbox("Hiển thị phiên", [90, 180, 365, 756], index=1)
            try:
                result = forecast_for_agent(horizon)
                m = st.columns(4)
                m[0].metric("Giá cuối (USD)", f"{result['last']:.2f}")
                m[1].metric(f"Dự báo +{horizon} phiên", f"{result['forecast'].iloc[-1]:.2f}", f"{result['change_pct']:+.1f}%")
                m[2].metric("Holdout RMSE (USD)", f"{result['rmse']:.2f}")
                m[3].metric("Tập kiểm tra", f"{result['n_test']} phiên")
                fig = go.Figure()
                series = PRICES.iloc[-show_days:]
                fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Giá lịch sử thực"))
                fig.add_trace(go.Scatter(x=result["forecast"].index, y=result["forecast"].values,
                                         name="Dự báo LSTM", line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=list(result["upper"].index)+list(result["lower"].index[::-1]),
                                         y=list(result["upper"])+list(result["lower"][::-1]), fill="toself",
                                         fillcolor="rgba(90,120,220,0.14)", line=dict(width=0), name="Dải sai số minh hoạ"))
                fig.update_layout(height=450, yaxis_title="USD", legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(px.line(x=range(1, len(result["loss_hist"])+1), y=result["loss_hist"],
                                        labels=dict(x="Epoch", y="Train MSE"), title="Đường loss LSTM"), use_container_width=True)
                st.download_button("Tải kết quả CSV", result["forecast"].rename("Predicted_Close").to_csv().encode("utf-8"),
                                   "aapl_lstm_forecast.csv", mime="text/csv")
                st.caption("Dải sai số = ±RMSE×√h mang tính minh họa, KHÔNG phải khoảng tin cậy 95% đã hiệu chuẩn. Dự báo không phải tư vấn đầu tư.")
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    with tab_chat:
        chat_panel("stock_chat", "stock", ["Nhà cung cấp chính của Apple?", "TSMC sản xuất gì cho Apple?", "Thuế quan Mỹ-Trung tác động thế nào?"])
    with tab_events:
        events = [n for n,d in G.nodes(data=True) if d["type"] == "Event"]
        event = st.selectbox("Sự kiện", events)
        rows = [dict(Đối_tượng=v, Tác_động=d["note"], Mã_nguồn=d["source"]) for _,v,d in G.out_edges(event, data=True) if d["rel"] == "AFFECTS"]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        nodes, edges = expand_subgraph(G, [event], 2)
        st.plotly_chart(plot_graph(G, nodes, edges, [event], height=420), use_container_width=True)

# ====================== KNOWLEDGE GRAPH ======================
elif page == "🕸️ Knowledge Graph":
    st.title("🕸️ Khám phá Knowledge Graph")
    t1, t2, t3, t4 = st.tabs(["Đồ thị", "Cypher (đọc)", "NetworkX", "Đường đa chặng"])
    with t1:
        domain = st.radio("Miền", ["Tất cả", "medical", "stock"], horizontal=True)
        nodes = [n for n,d in G.nodes(data=True) if domain == "Tất cả" or d["domain"] == domain]
        st.plotly_chart(plot_graph(G, nodes, height=620, show_labels=len(nodes) < 35), use_container_width=True)
        st.caption(f"{len(nodes)} nút · {G.subgraph(nodes).number_of_edges()} cạnh")
    with t2:
        st.caption("NetworkX hỗ trợ mini-Cypher MATCH ... RETURN; kết nối Neo4j đọc qua NEO4J_URI nếu có.")
        examples = {
            "Metformin chống chỉ định": "MATCH (a:Drug)-[r:CONTRAINDICATED_IN]->(b) RETURN a.name, type(r), b.name LIMIT 20",
            "Nhà cung cấp Apple": "MATCH (a:Company)-[r:SUPPLIES]->(b:Company {name:'Apple Inc.'}) RETURN a.name, type(r), b.name",
            "Triệu chứng type 2": "MATCH (a:Disease {name:'Tiểu đường type 2'})-[r:HAS_SYMPTOM]->(b) RETURN a.name, type(r), b.name",
        }
        choice = st.selectbox("Truy vấn mẫu", list(examples))
        query = st.text_area("Cypher", examples[choice], height=100)
        if st.button("Chạy Cypher", type="primary"):
            try:
                frame = pd.DataFrame(run_cypher(G, query))
                st.dataframe(frame, use_container_width=True)
                st.download_button("CSV", frame.to_csv(index=False).encode("utf-8"), "cypher_results.csv")
            except Exception as exc: st.error(f"Truy vấn không thực thi: {exc}")
    with t3:
        node = st.selectbox("Chọn thực thể", sorted(G.nodes), index=sorted(G.nodes).index("Metformin"))
        st.write(f"{node}: bậc vào {G.in_degree(node)}, bậc ra {G.out_degree(node)}")
        rows = [dict(Hướng="→", Quan_hệ=d["rel"], Đích=v, Nguồn=d["source"]) for _,v,d in G.out_edges(node, data=True)]
        rows += [dict(Hướng="←", Quan_hệ=d["rel"], Đích=u, Nguồn=d["source"]) for u,_,d in G.in_edges(node, data=True)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        rank = pd.Series(nx.pagerank(nx.DiGraph(G))).sort_values(ascending=False).head(10)
        st.plotly_chart(px.bar(rank[::-1], orientation="h", title="Top 10 PageRank"), use_container_width=True)
    with t4:
        names = sorted(G.nodes); cols = st.columns(3)
        source = cols[0].selectbox("Nguồn", names, index=names.index("Metformin"))
        dest = cols[1].selectbox("Đích", names, index=names.index("Bệnh thận đái tháo đường"))
        hops = cols[2].slider("Chặng tối đa", 1, 5, 3)
        paths = multi_hop_paths(G, source, dest, hops, 10)
        if not paths: st.info("Chưa có đường nối trong giới hạn chặng.")
        for n,p in paths: st.markdown(f"**{n} chặng:** {p}")

# ====================== AGENT ======================
elif page == "🤖 Multi-Agent":
    st.title("🤖 Agent Orchestrator (ReAct theo luật)")
    st.caption("KG Query · Vector Search · Clinical ML Predictor · Stock LSTM Forecaster. Có trace hành động/quan sát, không giả là bốn tác tử LLM độc lập.")
    examples = ["Tiểu đường type 2 có biến chứng thận, Metformin cần thận trọng khi nào?",
                "Dự báo AAPL tuần tới và thuế quan Mỹ-Trung tác động đến Foxconn ra sao?"]
    cols = st.columns(2)
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True): S["agent_pending"] = example
    question = st.chat_input("Nhập câu hỏi yêu cầu nhiều công cụ...") or S.pop("agent_pending", None)
    if question:
        try: S["agent_last"] = (question, agent.run(CTX, question, TOPK, HOPS, ALPHA))
        except Exception as exc: st.error(f"Agent lỗi: {exc}")
    if "agent_last" in S:
        question, result = S["agent_last"]
        st.chat_message("user").write(question)
        with st.chat_message("assistant"): st.markdown(result["answer"])
        t1,t2,t3,t4 = st.tabs(["🧾 Nhật ký ReAct", "🔧 Kết quả công cụ", "📚 Nguồn", "🕸️ Đồ thị con"])
        with t1:
            for step in result["trace"]:
                with st.expander(f"Bước {step['step']}: {step['action']} · {step['ms']} ms", expanded=True):
                    st.markdown(f"**Thought:** {step['thought']}")
                    st.markdown(f"**Action:** {step['action']}")
                    st.markdown(f"**Observation:** {step['observation']}")
        with t2:
            if result["ml"]: st.metric("P(Outcome=1) RF", f"{result['ml']['prob']:.1%}")
            if result["forecast"]: st.metric("AAPL dự báo phiên cuối", f"${result['forecast']['forecast'].iloc[-1]:.2f}")
            if not result["ml"] and not result["forecast"]: st.info("Không chạy ML/LSTM hoặc dữ liệu chưa đầy đủ; xem Nhật ký ReAct để biết nguyên nhân.")
            st.write("\n".join(f"- {f}" for f in result["facts"][:15]))
        with t3: show_passages(result["passages"])
        with t4: st.plotly_chart(plot_graph(G, result["nodes"], result["edges"], result["seeds"], height=520), use_container_width=True)
        dump = dict(question=question, answer=result["answer"], trace=result["trace"],
                    sources=[dict(id=p["id"], url=p["url"], source=p["source"]) for p in result["passages"]],
                    facts=result["facts"])
        st.download_button("Xuất nhật ký JSON", json.dumps(dump, ensure_ascii=False, indent=2).encode(),
                           "agent_trace.json", mime="application/json")

# ====================== BENCHMARK ======================
elif page == "📊 Benchmark":
    st.title("📊 Benchmark các kiến trúc truy xuất")
    st.caption("Chỉ số proxy tái lập được, không thay thế đánh giá bằng chuyên gia; thời gian là trên môi trường đang chạy.")
    methods = st.multiselect("Phương pháp so sánh", benchmark.ALL_METHODS, default=benchmark.ALL_METHODS)
    questions = st.multiselect("Bộ câu hỏi có nhãn", [q["q"] for q in benchmark.QUESTIONS],
                               default=[q["q"] for q in benchmark.QUESTIONS])
    custom = st.text_input("Câu hỏi tự nhập (không có nhãn chuẩn, chỉ đo grounding/thời gian)")
    if not llm.has_llm(): st.info("LLM thuần túy sẽ ghi SKIP (không có ANTHROPIC_API_KEY), KHÔNG được tính thành baseline thực.")
    if st.button("Chạy benchmark", type="primary"):
        items = [it for it in benchmark.QUESTIONS if it["q"] in questions]
        if custom: items.append(dict(q=custom, kw=[], ent=[], hops=1))
        if methods and items:
            progress = st.progress(0.0)
            try: S["bench"] = benchmark.run_benchmark(CTX, methods, items, TOPK, HOPS, ALPHA, progress.progress)
            except Exception as exc: st.error(f"Benchmark thất bại: {exc}")
            finally: progress.empty()
        else: st.warning("Chọn ít nhất một phương pháp và một câu hỏi.")
    if "bench" in S:
        frame = S["bench"]
        agg = frame.groupby("method", as_index=False)[["quality", "multihop", "grounding", "latency_s"]].mean().round(3)
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.markdown("**Chú giải:** quality = tỷ lệ từ khóa đáp án chuẩn có trong câu trả lời; multihop = độ phủ thực thể trên câu gán nhãn ≥2 chặng; grounding = độ trùng từ câu trả lời/ngữ cảnh truy xuất; latency = thời gian thực thi. Ô trống nghĩa là không áp dụng hoặc chưa đo.")
        col1,col2=st.columns(2)
        scores=agg.melt("method", ["quality", "multihop", "grounding"], var_name="Chỉ số", value_name="Điểm")
        col1.plotly_chart(px.bar(scores.dropna(), x="method", y="Điểm", color="Chỉ số", barmode="group", title="Các chỉ số proxy"), use_container_width=True)
        col2.plotly_chart(px.bar(agg.dropna(subset=["latency_s"]), x="method", y="latency_s", title="Độ trễ (giây)"), use_container_width=True)
        with st.expander("Chi tiết từng câu hỏi và trạng thái đo"):
            st.dataframe(frame, use_container_width=True, hide_index=True)
        a,b=st.columns(2)
        a.download_button("Xuất kết quả CSV", frame.to_csv(index=False).encode("utf-8-sig"), "benchmark.csv", "text/csv")
        b.download_button("Xuất kết quả JSON", frame.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                          "benchmark.json", "application/json")
