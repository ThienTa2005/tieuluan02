"""Kiểm thử RAG, tích hợp Agent, chống hiện tượng trả lời giả."""
import numpy as np
from src.kg import build_graph
from src.corpus import DOCS
from src.vector import VectorIndex
from src import rag, agent, benchmark, llm


def context(): return dict(G=build_graph(), vi=VectorIndex(DOCS, backend="tfidf"), docs=DOCS, rf=None, forecast=None)

def test_vector_search():
    vi = VectorIndex(DOCS, backend="tfidf")
    result = vi.search("metformin suy thận", 4, domain="medical")
    assert len(result)==4 and all(d["domain"]=="medical" for d in result)
    assert result[0]["id"] == "M4"

def test_vector_index_label():
    vi=VectorIndex(DOCS, backend="tfidf")
    assert "không phải embedding" in vi.detail

def test_rag_all_methods():
    ctx = context()
    for name in rag.METHODS:
        result=rag.answer(ctx,name,"Metformin chống chỉ định khi nào?",k=3,domain="medical")
        assert result["method"]==name
        assert result["passages"] and len(result["passages"])<=3
        assert all(p["domain"]=="medical" and p["url"] for p in result["passages"])
        assert result["answer"]

def test_hybrid_alpha():
    ctx=context()
    a=rag.hybrid_rag(ctx,"Metformin suy thận",alpha=1.0,domain="medical")
    b=rag.hybrid_rag(ctx,"Metformin suy thận",alpha=0.0,domain="medical")
    assert len(a["passages"])==4 and len(b["passages"])==4
    assert all(0 <= p["score"] <= 1.01 for p in a["passages"])

def test_kg_cites_document_id():
    result=rag.graph_rag(context(),"Metformin eGFR 30",domain="medical")
    assert any("[M4]" in f for f in result["facts"])

def test_agent_has_four_tools():
    assert set(agent.TOOLS)=={"KG Query","Vector Search","Clinical ML Predictor","Stock LSTM Forecaster"}

def test_agent_ml_no_fake():
    o=agent.tool_clinical_ml(context(),"Glucose 160, BMI 35, 55 tuổi")
    assert o["result"] is None
    assert "chưa có CSV" in o["obs"]

def test_agent_lstm_no_fake():
    o=agent.tool_stock_lstm(context(),"Dự báo AAPL")
    assert o["result"] is None
    assert "giá AAPL thực" in o["obs"]

def test_agent_react_trace():
    result=agent.run(context(),"Dự báo AAPL tuần tới và thuế quan Mỹ-Trung",domain="stock")
    assert "Stock LSTM Forecaster" in result["tools_used"]
    assert "Vector Search" in result["tools_used"]
    assert result["trace"][-1]["action"]=="Final Answer"
    assert all("observation" in x and "thought" in x for x in result["trace"])

def test_agent_parse_all_features():
    q="Pregnancies 3, Glucose 150, BloodPressure 75, SkinThickness 20, Insulin 120, BMI 31, DPF 0.6, age 45 tuoi"
    parsed=agent.parse_clinical(q)
    # Hàm nhận các nhãn tiếng Anh riêng, Age 45 tuoi vẫn hỗ trợ.
    assert len(parsed)==8 and parsed["Glucose"]==150

def test_benchmark_skip_llm_without_key(monkeypatch):
    monkeypatch.setattr(llm,"has_llm",lambda:False)
    row=benchmark.evaluate(context(),"LLM thuần túy",benchmark.QUESTIONS[0])
    assert np.isnan(row["quality"]) and row["status"].startswith("SKIP")

def test_benchmark_custom_has_nan_quality():
    row=benchmark.evaluate(context(),"Vector RAG",dict(q="Metformin là gì?",kw=[],ent=[],hops=1))
    assert np.isnan(row["quality"]) and np.isnan(row["multihop"])

def test_benchmark_run_exportable():
    frame=benchmark.run_benchmark(context(),["GraphRAG","Vector RAG"],benchmark.QUESTIONS[:2])
    assert len(frame)==4
    assert '"method"' in frame.to_json(orient="records")
