"""Kiểm tra KG, truy vấn mini-Cypher và đường đi nhiều chặng."""
import pytest
from src.kg import build_graph, run_cypher, link_entities, multi_hop_paths, expand_subgraph, norm
from src.corpus import DOCS

@pytest.fixture(scope="module")
def graph(): return build_graph()

def test_graph_size(graph):
    assert graph.number_of_nodes() == 57
    assert graph.number_of_edges() == 75

def test_graph_documents_existing(graph):
    known = {d["id"] for d in DOCS}
    assert len(DOCS) == 22
    assert all(edge["source"] in known for _,_,edge in graph.edges(data=True))

def test_every_doc_has_source_url():
    assert all(d["url"].startswith("https://") and d["source"] for d in DOCS)

def test_entity_matching_diacritics(graph):
    assert "Metformin" in link_entities(graph, "metformin gây ảnh hưởng gì?")
    assert "Tiểu đường type 2" in link_entities(graph, "tieu duong type 2")

def test_norm(graph):
    assert norm("Đường huyết") == "duong huyet"

def test_cypher_select(graph):
    results = run_cypher(graph, "MATCH (a:Company)-[r:SUPPLIES]->(b:Company {name:'Apple Inc.'}) RETURN a.name, type(r), b.name LIMIT 10")
    assert len(results) >= 3
    assert any(x["a.name"] == "TSMC" for x in results)

def test_cypher_reject_write(graph):
    with pytest.raises(ValueError): run_cypher(graph, "MATCH (n) DETACH DELETE n")

def test_multi_hop(graph):
    paths = multi_hop_paths(graph,"Metformin","Bệnh thận đái tháo đường",cutoff=4)
    assert paths and min(n for n,_ in paths) >= 2

def test_expand_subgraph(graph):
    nodes, edges = expand_subgraph(graph,["Metformin"],2,max_edges=20)
    assert "Metformin" in nodes and 0 < len(edges) <= 20
