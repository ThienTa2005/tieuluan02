"""Knowledge Graph: xây dựng đồ thị (NetworkX), truy vấn Cypher (mini-parser + Neo4j tuỳ chọn), tìm kiếm đa chặng."""
import os, re, unicodedata
import networkx as nx

def norm(s: str) -> str:
    """Chuẩn hoá chuỗi: chữ thường, bỏ dấu tiếng Việt -> so khớp không phụ thuộc dấu."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")

# (tên, loại, miền, danh sách alias)
NODES = [
 ("Tiểu đường type 2","Disease","medical",["đái tháo đường type 2","t2dm","tiểu đường tuýp 2","đái tháo đường tuýp 2","tiểu đường"]),
 ("Tiểu đường type 1","Disease","medical",["đái tháo đường type 1","t1dm","tiểu đường tuýp 1"]),
 ("Tiểu đường thai kỳ","Disease","medical",["đái tháo đường thai kỳ"]),
 ("Tiền tiểu đường","Disease","medical",["tiền đái tháo đường"]),
 ("Khát nước nhiều","Symptom","medical",["khát nước","khát"]),
 ("Tiểu nhiều","Symptom","medical",["đi tiểu nhiều"]),
 ("Sụt cân không rõ nguyên nhân","Symptom","medical",["sụt cân","giảm cân"]),
 ("Mệt mỏi","Symptom","medical",[]),
 ("Nhìn mờ","Symptom","medical",["mờ mắt"]),
 ("Vết thương lâu lành","Symptom","medical",[]),
 ("Metformin","Drug","medical",[]),
 ("Insulin","Drug","medical",[]),
 ("Gliclazide","Drug","medical",["sulfonylurea"]),
 ("Empagliflozin","Drug","medical",["sglt2"]),
 ("Sitagliptin","Drug","medical",["dpp-4","dpp4"]),
 ("Liraglutide","Drug","medical",["glp-1","glp1"]),
 ("Bệnh võng mạc đái tháo đường","Complication","medical",["võng mạc"]),
 ("Bệnh thận đái tháo đường","Complication","medical",["biến chứng thận","bệnh thận","thận"]),
 ("Bệnh thần kinh ngoại biên","Complication","medical",["thần kinh"]),
 ("Bệnh tim mạch","Complication","medical",["tim mạch"]),
 ("Nhiễm toan ceton","Complication","medical",["dka","toan ceton"]),
 ("Hạ đường huyết","Complication","medical",["hạ glucose","hạ đường máu"]),
 ("Nhiễm toan lactic","Complication","medical",["toan lactic"]),
 ("Suy thận nặng (eGFR<30)","Condition","medical",["suy thận","egfr"]),
 ("Suy gan nặng","Condition","medical",["suy gan"]),
 ("Viêm tuỵ","Condition","medical",["viêm tụy"]),
 ("Ung thư tuyến giáp thể tuỷ","Condition","medical",["ung thư tuyến giáp","tuyến giáp"]),
 ("Mang thai","Condition","medical",["thai kỳ","có thai"]),
 ("Béo phì","RiskFactor","medical",["thừa cân","bmi cao"]),
 ("Tuổi từ 45","RiskFactor","medical",["tuổi cao"]),
 ("Tiền sử gia đình","RiskFactor","medical",["di truyền"]),
 ("Ít vận động","RiskFactor","medical",["lười vận động"]),
 ("Tăng huyết áp","RiskFactor","medical",["huyết áp cao"]),
 ("HbA1c","Test","medical",[]),
 ("Đường huyết lúc đói","Test","medical",["glucose đói","fpg","đường huyết đói"]),
 ("Nghiệm pháp dung nạp glucose","Test","medical",["ogtt"]),
 # --- chứng khoán ---
 ("Apple Inc.","Company","stock",["aapl","apple"]),
 ("TSMC","Company","stock",["tsm"]),
 ("Foxconn","Company","stock",["hon hai"]),
 ("Samsung Electronics","Company","stock",["samsung"]),
 ("Qualcomm","Company","stock",[]),
 ("Broadcom","Company","stock",[]),
 ("Corning","Company","stock",[]),
 ("Alphabet","Company","stock",["google"]),
 ("Microsoft","Company","stock",[]),
 ("iPhone","Product","stock",[]),
 ("Mac","Product","stock",["macbook"]),
 ("iPad","Product","stock",[]),
 ("Wearables","Product","stock",["apple watch","airpods"]),
 ("Dịch vụ","Product","stock",["services","app store"]),
 ("Thuế quan Mỹ-Trung","Event","stock",["thuế quan","tariff"]),
 ("Thiếu hụt chip","Event","stock",["chip shortage","thiếu chip"]),
 ("Ra mắt Apple Intelligence","Event","stock",["apple intelligence"]),
 ("Fed tăng lãi suất","Event","stock",["lãi suất","fed"]),
 ("Luật DMA của EU","Event","stock",["dma"]),
 ("Bán dẫn","Sector","stock",["semiconductor"]),
 ("Smartphone","Sector","stock",["điện thoại thông minh"]),
]

# (nguồn, quan hệ, đích, tài liệu nguồn, ghi chú)
EDGES = [
 *[("Tiểu đường type 2","HAS_SYMPTOM",s,"M2","") for s in ["Khát nước nhiều","Tiểu nhiều","Mệt mỏi","Nhìn mờ","Vết thương lâu lành"]],
 *[("Tiểu đường type 1","HAS_SYMPTOM",s,"M3","") for s in ["Khát nước nhiều","Tiểu nhiều","Sụt cân không rõ nguyên nhân","Mệt mỏi"]],
 *[(r,"RISK_FOR","Tiểu đường type 2","M12","") for r in ["Béo phì","Tuổi từ 45","Tiền sử gia đình","Ít vận động","Tăng huyết áp"]],
 ("Ít vận động","LEADS_TO","Béo phì","M13",""),
 *[("Tiểu đường type 2","DIAGNOSED_BY",t,"M1","") for t in ["HbA1c","Đường huyết lúc đói","Nghiệm pháp dung nạp glucose"]],
 ("Tiền tiểu đường","DIAGNOSED_BY","HbA1c","M1","HbA1c 5.7-6.4%"),
 ("Tiền tiểu đường","LEADS_TO","Tiểu đường type 2","M13",""),
 *[("Tiểu đường type 2","CAUSES_COMPLICATION",c,"M10","") for c in ["Bệnh võng mạc đái tháo đường","Bệnh thận đái tháo đường","Bệnh thần kinh ngoại biên","Bệnh tim mạch"]],
 ("Bệnh thận đái tháo đường","LEADS_TO","Suy thận nặng (eGFR<30)","M10","bệnh thận tiến triển"),
 ("Metformin","TREATS","Tiểu đường type 2","M4","thuốc đầu tay"),
 ("Insulin","TREATS","Tiểu đường type 1","M9",""),
 ("Insulin","TREATS","Tiểu đường type 2","M9",""),
 ("Insulin","TREATS","Tiểu đường thai kỳ","M14",""),
 ("Gliclazide","TREATS","Tiểu đường type 2","M7",""),
 ("Empagliflozin","TREATS","Tiểu đường type 2","M5",""),
 ("Sitagliptin","TREATS","Tiểu đường type 2","M8",""),
 ("Liraglutide","TREATS","Tiểu đường type 2","M6",""),
 ("Insulin","SIDE_EFFECT","Hạ đường huyết","M9",""),
 ("Gliclazide","SIDE_EFFECT","Hạ đường huyết","M7",""),
 ("Metformin","SIDE_EFFECT","Nhiễm toan lactic","M4","hiếm"),
 ("Empagliflozin","SIDE_EFFECT","Nhiễm toan ceton","M5","bình đường huyết"),
 ("Metformin","CONTRAINDICATED_IN","Suy thận nặng (eGFR<30)","M4","eGFR<30"),
 ("Metformin","CAUTION_IN","Suy gan nặng","M4","tránh dùng theo nhãn thuốc"),
 ("Liraglutide","CONTRAINDICATED_IN","Ung thư tuyến giáp thể tuỷ","M6",""),
 ("Liraglutide","CAUTION_IN","Viêm tuỵ","M6","thận trọng"),
 ("Sitagliptin","CAUTION_IN","Viêm tuỵ","M8","thận trọng"),
 ("Sitagliptin","CAUTION_IN","Suy thận nặng (eGFR<30)","M8","chỉnh liều"),
 ("Gliclazide","CAUTION_IN","Suy thận nặng (eGFR<30)","M7","thận trọng"),
 ("Empagliflozin","CAUTION_IN","Mang thai","M5","không khuyến cáo"),
 ("Empagliflozin","PROTECTS","Bệnh tim mạch","M5",""),
 ("Empagliflozin","PROTECTS","Bệnh thận đái tháo đường","M5",""),
 # chứng khoán
 ("TSMC","SUPPLIES","Apple Inc.","S1","chip A/M-series"),
 ("Foxconn","SUPPLIES","Apple Inc.","S1","lắp ráp iPhone"),
 ("Samsung Electronics","SUPPLIES","Apple Inc.","S1","màn hình OLED"),
 ("Qualcomm","SUPPLIES","Apple Inc.","S1","modem 5G"),
 ("Broadcom","SUPPLIES","Apple Inc.","S1","linh kiện RF"),
 ("Corning","SUPPLIES","Apple Inc.","S1","kính"),
 ("Samsung Electronics","COMPETES_WITH","Apple Inc.","S3",""),
 ("Alphabet","COMPETES_WITH","Apple Inc.","S3",""),
 ("Microsoft","COMPETES_WITH","Apple Inc.","S3",""),
 *[("Apple Inc.","PRODUCES",p,"S2","") for p in ["iPhone","Mac","iPad","Wearables","Dịch vụ"]],
 ("TSMC","IN_SECTOR","Bán dẫn","S5",""),
 ("Qualcomm","IN_SECTOR","Bán dẫn","S1",""),
 ("iPhone","IN_SECTOR","Smartphone","S3",""),
 ("Samsung Electronics","IN_SECTOR","Smartphone","S3",""),
 ("Thuế quan Mỹ-Trung","AFFECTS","Foxconn","S4","tiêu cực"),
 ("Thuế quan Mỹ-Trung","AFFECTS","iPhone","S4","tiêu cực"),
 ("Thuế quan Mỹ-Trung","AFFECTS","Apple Inc.","S4","tiêu cực"),
 ("Thiếu hụt chip","AFFECTS","TSMC","S5","tiêu cực"),
 ("Thiếu hụt chip","AFFECTS","Mac","S5","tiêu cực"),
 ("Thiếu hụt chip","AFFECTS","Apple Inc.","S5","tiêu cực"),
 ("Ra mắt Apple Intelligence","AFFECTS","iPhone","S6","tích cực"),
 ("Ra mắt Apple Intelligence","AFFECTS","Apple Inc.","S6","tích cực"),
 ("Fed tăng lãi suất","AFFECTS","Apple Inc.","S7","tiêu cực"),
 ("Luật DMA của EU","AFFECTS","Dịch vụ","S8","tiêu cực"),
]

def build_graph() -> nx.MultiDiGraph:
    """Dựng đồ thị có hướng đa cạnh từ danh sách NODES/EDGES."""
    G = nx.MultiDiGraph()
    for name, typ, dom, al in NODES:
        G.add_node(name, type=typ, domain=dom, aliases=al)
    for a, r, b, src, note in EDGES:
        assert a in G and b in G, (a, b)
        G.add_edge(a, b, rel=r, source=src, note=note)
    return G

# ---------------- Liên kết thực thể & truy vấn ----------------
def link_entities(G, text: str):
    """Tìm các nút được nhắc tới trong câu hỏi (khớp tên/alias, không phân biệt dấu)."""
    nt = norm(text); hits = []
    for n, d in G.nodes(data=True):
        names = [n, re.sub(r"\s*\(.*?\)", "", n)] + d.get("aliases", [])
        for a in names:
            na = norm(a)
            if len(na) >= 2 and re.search(r"(?<!\w)" + re.escape(na) + r"(?!\w)", nt):
                hits.append((n, na)); break
    # bỏ nút có alias là chuỗi con của alias dài hơn thuộc nút khác (vd "tiểu đường" vs "tiểu đường type 1")
    return [n for n, a in hits if not any(a != a2 and a in a2 and n != n2 for n2, a2 in hits)]

def fact_str(G, u, v, d):
    note = f" ({d['note']})" if d.get("note") else ""
    return f"{u} —[{d['rel']}]→ {v}{note}"

def expand_subgraph(G, seeds, hops=2, prefer_rels=(), max_edges=40):
    """Mở rộng đa chặng (BFS vô hướng) quanh các nút hạt giống; ưu tiên quan hệ theo ý định câu hỏi."""
    if not seeds: return [], []
    seen = set(seeds); frontier = list(seeds); edges = {}
    for h in range(1, hops + 1):
        nxt = []
        for n in frontier:
            for u, v, k, d in list(G.out_edges(n, keys=True, data=True)) + list(G.in_edges(n, keys=True, data=True)):
                m = v if u == n else u
                score = -h + (2 if d["rel"] in prefer_rels else 0) + (1 if n in seeds else 0)
                edges[(u, v, k)] = max(score, edges.get((u, v, k), -99))
                if m not in seen: seen.add(m); nxt.append(m)
        frontier = nxt
    top = sorted(edges, key=lambda e: -edges[e])[:max_edges]
    nodes = set(seeds)
    for u, v, k in top: nodes.update([u, v])
    elist = [(u, v, G.edges[u, v, k]["rel"]) for u, v, k in top]
    return sorted(nodes), elist

def multi_hop_paths(G, a, b, cutoff=4, limit=5):
    """Tìm đường đi đa chặng giữa 2 nút (bỏ qua hướng cạnh), trả về [(số chặng, chuỗi quan hệ)]."""
    try:
        U = nx.Graph(G.to_undirected())
        paths = []
        for p in nx.all_simple_paths(U, a, b, cutoff=cutoff):
            parts = [p[0]]
            for x, y in zip(p, p[1:]):
                if G.has_edge(x, y): d = next(iter(G.get_edge_data(x, y).values())); parts.append(f"—[{d['rel']}]→ {y}")
                else: d = next(iter(G.get_edge_data(y, x).values())); parts.append(f"←[{d['rel']}]— {y}")
            paths.append((len(p) - 1, " ".join(parts)))
            if len(paths) >= 200: break
        return sorted(paths)[:limit]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

# ---------------- Mini Cypher trên NetworkX ----------------
_CY = re.compile(
 r"MATCH\s*\((\w+)(?::(\w+))?(?:\s*\{name:\s*'([^']+)'\})?\)\s*-\[(\w+)?(?::(\w+))?\]->\s*\((\w+)(?::(\w+))?(?:\s*\{name:\s*'([^']+)'\})?\)"
 r"(?:\s*WHERE\s+(\w+)\.name\s+CONTAINS\s+'([^']+)')?\s*RETURN\s+(.+?)(?:\s+LIMIT\s+(\d+))?\s*;?\s*$", re.I | re.S)

def run_cypher(G, query: str):
    """Chạy tập con Cypher: MATCH (a:Type {name:'x'})-[r:REL]->(b:Type) [WHERE a.name CONTAINS 'x'] RETURN a.name, type(r), b.name [LIMIT n].
    Nếu cấu hình biến môi trường NEO4J_URI thì chạy trên Neo4j thật."""
    if os.getenv("NEO4J_URI"):
        return run_neo4j(query)
    m = _CY.match(query.strip())
    if not m:
        raise ValueError("Cú pháp chưa hỗ trợ. Ví dụ: MATCH (a:Drug)-[r:TREATS]->(b:Disease) RETURN a.name, type(r), b.name LIMIT 20")
    va, ta, na, vr, rel, vb, tb, nb, wv, wt, ret, lim = m.groups()
    out, seen = [], set()
    for x, y, d in G.edges(data=True):          # x -> y là cạnh có hướng; a=x, b=y
        if rel and d["rel"] != rel: continue
        if ta and G.nodes[x]["type"] != ta: continue
        if tb and G.nodes[y]["type"] != tb: continue
        if na and x != na: continue
        if nb and y != nb: continue
        env = {va: x, vb: y}
        if wv and wt.lower() not in env.get(wv, "").lower(): continue
        row = {}
        for item in [i.strip() for i in ret.split(",")]:
            mm = re.match(r"(\w+)\.(name|type)$", item)
            if mm and mm.group(1) in env:
                row[item] = env[mm.group(1)] if mm.group(2) == "name" else G.nodes[env[mm.group(1)]]["type"]
            elif vr and item == f"type({vr})": row[item] = d["rel"]
            else: row[item] = None
        key = tuple(row.items())
        if key not in seen: seen.add(key); out.append(row)
    return out[: int(lim) if lim else 100]

def run_neo4j(query):
    """Chỉ cho phép truy vấn MATCH đọc dữ liệu từ giao diện công khai.

    Bộ lọc này dành cho dashboard học tập, không thay thế phân quyền chỉ-đọc
    trong Neo4j. Nên cấp tài khoản READ cho ứng dụng khi deploy.
    """
    query = query.strip()
    if not re.match(r"^MATCH\s", query, flags=re.I) or not re.search(r"\bRETURN\b", query, re.I):
        raise ValueError("Chỉ hỗ trợ MATCH ... RETURN (chế độ đọc).")
    if re.search(r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP|LOAD|CALL|REMOVE|FOREACH|GRANT|DENY|REVOKE|UNWIND)\b|;", query, re.I):
        raise ValueError("Truy vấn ghi hoặc thực thi thủ tục không được phép.")
    from neo4j import GraphDatabase, READ_ACCESS
    with GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))) as drv:
        with drv.session(default_access_mode=READ_ACCESS) as session:
            return [dict(row) for row in session.run(query).data()]

def load_to_neo4j(G):
    """Nạp bộ ba KG bằng MERGE, chạy qua script CLI riêng với tài khoản ghi."""
    from neo4j import GraphDatabase
    types = {d["type"] for _,d in G.nodes(data=True)}
    relations = {d["rel"] for _,_,d in G.edges(data=True)}
    # Các label và relationship chỉ lấy từ danh sách KG nội bộ, không từ user.
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) for value in types | relations):
        raise ValueError("Label/relationship không hợp lệ.")
    with GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))) as drv:
        with drv.session() as session:
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
            for name, data in G.nodes(data=True):
                session.run(f"MERGE (x:Entity {{name:$name}}) SET x:{data['type']}, x.domain=$domain", name=name, domain=data["domain"])
            for u, v, data in G.edges(data=True):
                session.run(f"MATCH (a:Entity {{name:$u}}),(b:Entity {{name:$v}}) MERGE (a)-[r:{data['rel']} {{source:$source}}]->(b) SET r.note=$note",
                            u=u, v=v, source=data["source"], note=data.get("note", ""))
