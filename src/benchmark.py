"""Benchmark nội bộ có nhãn: keyword recall, entity coverage, lexical support, latency.

Đây là PROXY tự thiết kế, không phải phép chấm đúng-sai bởi chuyên gia. Baseline
LLM thuần chỉ đo khi có API key thật; custom question không có gold -> NA.
"""
import re
import time
import numpy as np
import pandas as pd
from .kg import norm
from . import llm, rag, agent

QUESTIONS = [
    dict(q="Triệu chứng của tiểu đường type 2 là gì?", kw=["khat", "tieu nhieu", "met"], ent=["Khát nước nhiều", "Tiểu nhiều"], hops=1),
    dict(q="Metformin chống chỉ định trong trường hợp nào?", kw=["egfr", "suy than", "30"], ent=["Suy thận nặng (eGFR<30)"], hops=1),
    dict(q="Bệnh nhân tiểu đường có biến chứng thận thì thuốc nào chống chỉ định hoặc cần thận trọng?", kw=["metformin", "suy than"], ent=["Bệnh thận đái tháo đường", "Suy thận nặng (eGFR<30)", "Metformin"], hops=2),
    dict(q="Thuốc nào có thể gây hạ đường huyết?", kw=["insulin", "gliclazide"], ent=["Hạ đường huyết", "Insulin", "Gliclazide"], hops=1),
    dict(q="Ngưỡng HbA1c để chẩn đoán tiểu đường là bao nhiêu?", kw=["6.5"], ent=["HbA1c"], hops=1),
    dict(q="Những nhà cung cấp chính của Apple là ai?", kw=["tsmc", "foxconn"], ent=["TSMC", "Foxconn"], hops=1),
    dict(q="Thiếu hụt chip ảnh hưởng thế nào đến Apple và nhà cung cấp TSMC?", kw=["chip", "tsmc"], ent=["Thiếu hụt chip", "TSMC", "Apple Inc."], hops=2),
    dict(q="Thuế quan Mỹ-Trung tác động đến Foxconn và iPhone ra sao?", kw=["foxconn", "iphone"], ent=["Thuế quan Mỹ-Trung", "Foxconn", "iPhone"], hops=2),
]
ALL_METHODS = ["LLM thuần túy", "Vector RAG", "GraphRAG", "Hybrid Graph-Vector RAG", "Agentic Hybrid RAG"]

def _tok(s):
    return {w for w in re.findall(r"\w+", norm(s)) if len(w)>3}

def evaluate(ctx, method, item, k=4, hops=2, alpha=0.5):
    t0 = time.perf_counter(); question = item["q"]
    # Không gán chất lượng cho câu trả lời mô phỏng khi không có LLM thật.
    if method == "LLM thuần túy" and not llm.has_llm():
        return dict(method=method, question=question, quality=np.nan, multihop=np.nan,
                    grounding=np.nan, latency_s=np.nan, status="SKIP: thiếu ANTHROPIC_API_KEY")
    if method == "LLM thuần túy":
        ans = llm.generate_plain(question)
        context = ""; blob = norm(ans)
    else:
        result = agent.run(ctx, question, k=k, hops=hops, alpha=alpha) if method == "Agentic Hybrid RAG" else \
            rag.answer(ctx, method, question, k=k, hops=hops, alpha=alpha)
        ans = result["answer"]
        context = " ".join(result["facts"] + [p["text"] for p in result["passages"]])
        blob = norm(" ".join(result["nodes"]) + " " + context)
    duration = time.perf_counter()-t0
    keywords = item.get("kw") or []
    entities = item.get("ent") or []
    quality = sum(kw in norm(ans) for kw in keywords)/len(keywords) if keywords else np.nan
    # Chỉ áp dụng multi-hop với bộ câu hỏi gán nhãn ít nhất 2 chặng.
    multihop = sum(norm(e) in blob for e in entities)/len(entities) if entities and item.get("hops", 1)>=2 else np.nan
    grounding = len(_tok(ans) & _tok(context))/max(len(_tok(ans)), 1) if context else np.nan
    return dict(method=method, question=question, quality=quality, multihop=multihop,
                grounding=grounding, latency_s=duration, status="OK")

def run_benchmark(ctx, methods, items, k=4, hops=2, alpha=0.5, progress=None):
    if not methods or not items:
        return pd.DataFrame(columns=["method","question","quality","multihop","grounding","latency_s","status"])
    records = []
    for i, item in enumerate(items):
        for method in methods:
            records.append(evaluate(ctx, method, item, k=k, hops=max(hops, item.get("hops", 1)), alpha=alpha))
        if progress: progress((i+1)/len(items))
    return pd.DataFrame(records)
