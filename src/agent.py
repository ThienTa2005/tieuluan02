"""Agent Orchestrator dạng ReAct có nhật ký Thought → Action → Observation.

Đây là policy điều phối bằng luật (không phải 4 LLM agent độc lập).
Điều phối đúng bốn công cụ: KG, Vector, Clinical RF và Stock LSTM.
"""
import re
import time
from .kg import link_entities, expand_subgraph, multi_hop_paths, norm
from .rag import detect_rels, _facts, _edge_docs
from .ml import FEATURES, LABELS, predict_risk
from . import llm

PATTERNS = {
    "Pregnancies": r"(?:so lan mang thai|pregnancies|mang thai)\D{0,12}(\d+(?:[.,]\d+)?)",
    "Glucose": r"(?:glucose|duong huyet|duong mau)\D{0,12}(\d+(?:[.,]\d+)?)",
    "BloodPressure": r"(?:huyet ap|bloodpressure)\D{0,12}(\d+(?:[.,]\d+)?)",
    "SkinThickness": r"(?:skin thickness|skinthickness|nep da)\D{0,12}(\d+(?:[.,]\d+)?)",
    "Insulin": r"insulin\D{0,12}(\d+(?:[.,]\d+)?)",
    "BMI": r"\bbmi\D{0,8}(\d+(?:[.,]\d+)?)",
    "DiabetesPedigreeFunction": r"(?:dpf|diabetespedigreefunction)\D{0,8}(\d+(?:[.,]\d+)?)",
    "Age": r"(?:\btuoi\D{0,6}(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*tuoi\b)",
}

def parse_clinical(question):
    """Tách 8 giá trị Pima; không suy diễn các trường chưa được nhập."""
    q = norm(question)
    values = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            val = next((v for v in m.groups() if v is not None), None)
            if val is not None: values[key] = float(val.replace(",", "."))
    return values

def tool_kg_query(ctx, q, hops=2, domain=None, **_):
    G = ctx["G"]
    seeds = [s for s in link_entities(G, q) if domain is None or G.nodes[s]["domain"] == domain]
    pref = detect_rels(q)
    nodes, edges = expand_subgraph(G, seeds, hops, pref, max_edges=25)
    paths = multi_hop_paths(G, seeds[0], seeds[1], cutoff=hops+2, limit=2) if len(seeds) >= 2 else []
    facts = _facts(G, edges, pref)
    obs = f"Thực thể: {seeds or 'không có'} | {len(nodes)} nút, {len(edges)} cạnh" + "".join(
        f"\nĐường đi ({n} chặng): {p}" for n,p in paths)
    return dict(obs=obs, facts=facts, nodes=nodes, edges=edges, seeds=seeds, bonus=_edge_docs(G, edges, pref))

def tool_vector_search(ctx, q, k=4, domain=None, **_):
    hits = ctx["vi"].search(q, k, domain)
    return dict(obs="; ".join(f"{p['id']} ({p['score']:.2f})" for p in hits), passages=hits)

def tool_clinical_ml(ctx, q, **_):
    values = parse_clinical(q)
    missing = [f for f in FEATURES if f not in values]
    if ctx.get("rf") is None:
        return dict(obs="Không chạy Random Forest: chưa có CSV Pima thực để huấn luyện.", text="Chưa có mô hình tiểu đường được huấn luyện trên dữ liệu thực.", result=None)
    if missing:
        missing_text = ", ".join(LABELS[f] for f in missing)
        msg = f"Không ước tính: câu hỏi mới có {len(values)}/8 chỉ số Pima; thiếu {missing_text}. Vui lòng cung cấp đầy đủ ở tab Y tế."
        return dict(obs=msg, text=msg, result=None)
    try:
        r = predict_risk(ctx["rf"], values)
        txt = (f"Random Forest phân loại mẫu Pima: xác suất lớp Outcome=1 là {r['prob']:.1%}. "
               "Đây không phải dự đoán nguy cơ cá nhân hoặc chẩn đoán bệnh.")
        return dict(obs=txt, text=txt, result=r)
    except ValueError as exc:
        return dict(obs=str(exc), text=f"Không thể ước tính: {exc}", result=None)

def tool_stock_lstm(ctx, q, horizon=7, **_):
    if ctx.get("forecast") is None:
        return dict(obs="Không chạy LSTM: chưa có chuỗi giá AAPL thực.", text="Cần tải giá AAPL thật để chạy LSTM.", result=None)
    try:
        r = ctx["forecast"](horizon)
        msg = (f"{r['backend']} dự báo AAPL sau {horizon} phiên {r['forecast'].iloc[-1]:.2f} USD, "
               f"RMSE kiểm tra {r['rmse']:.2f} USD. Không phải khuyến nghị đầu tư.")
        return dict(obs=msg, text=msg, result=r)
    except (ValueError, RuntimeError) as exc:
        return dict(obs=f"LSTM không khả dụng: {exc}", text=f"LSTM không khả dụng: {exc}", result=None)

TOOLS = {"KG Query": tool_kg_query, "Vector Search": tool_vector_search,
         "Clinical ML Predictor": tool_clinical_ml, "Stock LSTM Forecaster": tool_stock_lstm}

def plan(question):
    """Lập kế hoạch theo luật để có thể chạy offline và giải thích được."""
    nq = norm(question); out = []
    if parse_clinical(question) and re.search(r"nguy co|du doan|uoc tinh|risk|mac tieu duong", nq):
        out.append(("Câu hỏi có các chỉ số và yêu cầu phân loại → kiểm tra đầu vào Pima trước khi gọi ML.", "Clinical ML Predictor"))
    if re.search(r"\b(aapl|apple)\b", nq) and re.search(r"du bao|forecast|lstm|gia co phieu|gia tuong lai|tuan toi", nq):
        out.append(("Câu hỏi yêu cầu dự báo AAPL → dùng chuỗi giá thực và LSTM nếu đã cấu hình.", "Stock LSTM Forecaster"))
    out.append(("Tìm thực thể và các quan hệ đa chặng có cấu trúc.", "KG Query"))
    out.append(("Tìm bằng chứng văn bản có nguồn và điểm tương đồng vector.", "Vector Search"))
    return out

def run(ctx, q, k=4, hops=2, alpha=0.5, domain=None, horizon=7):
    """Thực hiện các bước có thể giải thích, hợp nhất và sinh câu trả lời."""
    t0 = time.perf_counter(); trace = []; outs = {}
    for i, (thought, name) in enumerate(plan(q), start=1):
        t1 = time.perf_counter()
        result = TOOLS[name](ctx, q, k=k, hops=hops, domain=domain, horizon=horizon)
        outs[name] = result
        trace.append(dict(step=i, thought=thought, action=name,
                          observation=result["obs"], ms=round((time.perf_counter()-t1)*1000, 1)))
    kg, vs = outs["KG Query"], outs["Vector Search"]
    vmax = max([p["score"] for p in vs["passages"]] + [1e-9])
    candidates = {p["id"]: dict(p, fused=alpha*p["score"]/vmax) for p in vs["passages"]}
    graph_bonus = {}
    for doc in ctx["docs"]:
        if domain and doc["domain"] != domain: continue
        graph_bonus[doc["id"]] = len(set(kg["nodes"]) & set(doc["entities"])) + kg["bonus"].get(doc["id"], 0)
    gmax = max([1] + list(graph_bonus.values()))
    for doc in ctx["docs"]:
        if domain and doc["domain"] != domain: continue
        g = graph_bonus[doc["id"]]
        if g:
            c = candidates.setdefault(doc["id"], dict(doc, score=0., fused=0.))
            c["fused"] += (1-alpha)*g/gmax
    passages = sorted(candidates.values(), key=lambda x: -x["fused"])[:k]
    extra = [outs[name]["text"] for name in ("Clinical ML Predictor", "Stock LSTM Forecaster") if name in outs]
    answer = llm.generate(q, kg["facts"], passages, extra)
    trace.append(dict(step=len(trace)+1, thought="Hợp nhất chứng cứ từ KG/Vector và kết quả công cụ (nếu có).",
                      action="Final Answer", observation=f"{len(passages)} nguồn, {len(kg['facts'])} bộ ba",
                      ms=round((time.perf_counter()-t0)*1000, 1)))
    return dict(method="Agentic Hybrid RAG", answer=answer, trace=trace, passages=passages, facts=kg["facts"],
                nodes=kg["nodes"], edges=kg["edges"], seeds=kg["seeds"],
                tools_used=[s["action"] for s in trace[:-1]],
                ml=outs.get("Clinical ML Predictor", {}).get("result"),
                forecast=outs.get("Stock LSTM Forecaster", {}).get("result"),
                latency=time.perf_counter()-t0)
