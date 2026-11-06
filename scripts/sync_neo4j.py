"""Nạp NetworkX KG vào Neo4j khi có quyền WRITE (không chạy tự động)."""
import os
from src.kg import build_graph, load_to_neo4j
if __name__ == "__main__":
    if not os.getenv("NEO4J_URI") or not os.getenv("NEO4J_PASSWORD"):
        raise SystemExit("Cần đặt NEO4J_URI và NEO4J_PASSWORD trước khi nạp.")
    load_to_neo4j(build_graph())
    print("Đã đồng bộ Knowledge Graph lên Neo4j")
