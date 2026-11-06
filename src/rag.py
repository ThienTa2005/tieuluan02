"""Ba phương pháp truy xuất: Vector RAG, GraphRAG, Hybrid Graph-Vector RAG."""
import time
from .kg import link_entities, expand_subgraph, fact_str, norm
from . import llm

# Ý định câu hỏi -> quan hệ ưu tiên trong đồ thị
INTENT = {"trieu chung": "HAS_SYMPTOM", "dau hieu": "HAS_SYMPTOM", "thuoc": "TREATS", "dieu tri": "TREATS",
          "bien chung": "CAUSES_COMPLICATION", "chong chi dinh": "CONTRAINDICATED_IN", "than trong": "CAUTION_IN",
          "tac dung phu": "SIDE_EFFECT", "chan doan": "DIAGNOSED_BY", "nguy co": "RISK_FOR", "gay": "SIDE_EFFECT",
          "nha cung cap": "SUPPLIES", "cung cap": "SUPPLIES", "doi thu": "COMPETES_WITH", "canh tranh": "COMPETES_WITH",
          "tac dong": "AFFECTS", "anh huong": "AFFECTS", "san pham": "PRODUCES"}

def detect_rels(q):
    nq = norm(q); return tuple({r for k, r in INTENT.items() if k in nq})

def _edge_docs(G, edges, prefer):
    """Điểm cộng cho tài liệu là nguồn của các cạnh có quan hệ phù hợp ý định câu hỏi."""
    bonus = {}
    for u, v, rel in edges:
        for d in G.get_edge_data(u, v).values():
            if d["rel"] == rel: bonus[d["source"]] = bonus.get(d["source"], 0) + (3 if rel in prefer else 1)
    return bonus

def _facts(G, edges, prefer=()):
    out = []
    edges = sorted(edges, key=lambda e: e[2] not in prefer)   # sự kiện đúng ý định lên trước
    for u, v, rel in edges:
        for d in G.get_edge_data(u, v).values():
            if d["rel"] == rel: out.append(f"{fact_str(G, u, v, d)} [{d['source']}]"); break
    return out

def _result(method, passages, facts=(), nodes=(), edges=(), seeds=()):
    return dict(method=method, passages=passages, facts=list(facts), nodes=list(nodes), edges=list(edges), seeds=list(seeds))

def vector_rag(ctx, q, k=4, domain=None, **_):
    return _result("Vector RAG", ctx["vi"].search(q, k, domain))

def graph_rag(ctx, q, k=4, hops=2, domain=None, **_):
    G = ctx["G"]; seeds = link_entities(G, q)
    pref = detect_rels(q)
    nodes, edges = expand_subgraph(G, seeds, hops, pref, max_edges=25)
    ns = set(nodes); docs = []; bonus = _edge_docs(G, edges, pref)
    for d in ctx["docs"]:
        if domain and d["domain"] != domain: continue
        ov = len(ns & set(d["entities"])) + 2 * len(set(seeds) & set(d["entities"])) + bonus.get(d["id"], 0)
        if ov: docs.append({**d, "score": float(ov)})
    docs = sorted(docs, key=lambda x: -x["score"])[:k]
    return _result("GraphRAG", docs, _facts(G, edges, pref), nodes, edges, seeds)

def hybrid_rag(ctx, q, k=4, hops=2, alpha=0.5, domain=None, **_):
    """Hybrid: điểm = alpha * ngữ nghĩa (vector) + (1-alpha) * liên quan đồ thị. Hạt giống đồ thị lấy từ câu hỏi + top tài liệu vector."""
    G, vi = ctx["G"], ctx["vi"]
    vs = vi.scores(q); vmax = max(vs.max(), 1e-9)
    top = vs.argsort()[::-1][:3]
    seeds = link_entities(G, q)
    if len(seeds) < 1:   # câu hỏi không có thực thể -> mượn thực thể từ tài liệu vector tốt nhất
        for i in top[:1]:
            for e in ctx["docs"][i]["entities"][:2]:
                if e not in seeds: seeds.append(e)
    pref = detect_rels(q)
    nodes, edges = expand_subgraph(G, seeds, hops, pref, max_edges=25)
    ns, sd = set(nodes), set(seeds); cand = []; bonus = _edge_docs(G, edges, pref)
    gmax = 1
    for i, d in enumerate(ctx["docs"]):
        if domain and d["domain"] != domain: continue
        g = len(ns & set(d["entities"])) + 2 * len(sd & set(d["entities"])) + bonus.get(d["id"], 0); gmax = max(gmax, g)
        cand.append((i, g))
    scored = [{**ctx["docs"][i], "score": alpha * float(vs[i] / vmax) + (1 - alpha) * g / gmax, "vec": float(vs[i]), "graph": g} for i, g in cand]
    scored = sorted(scored, key=lambda x: -x["score"])[:k]
    return _result("Hybrid Graph-Vector RAG", scored, _facts(G, edges, pref), nodes, edges, seeds)

METHODS = {"Vector RAG": vector_rag, "GraphRAG": graph_rag, "Hybrid Graph-Vector RAG": hybrid_rag}

def answer(ctx, method, q, k=4, hops=2, alpha=0.5, domain=None):
    """Truy xuất + sinh câu trả lời + đo thời gian."""
    t0 = time.perf_counter()
    r = METHODS[method](ctx, q, k=k, hops=hops, alpha=alpha, domain=domain)
    r["answer"] = llm.generate(q, r["facts"], r["passages"])
    r["latency"] = time.perf_counter() - t0
    return r
