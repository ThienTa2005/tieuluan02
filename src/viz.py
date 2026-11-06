"""Vẽ đồ thị tương tác bằng Plotly (kéo/zoom/hover) từ NetworkX."""
import networkx as nx
import plotly.graph_objects as go

COLORS = {"Disease": "#e45756", "Symptom": "#f58518", "Drug": "#4c78a8", "Complication": "#b279a2", "Condition": "#9d755d",
          "RiskFactor": "#eeca3b", "Test": "#54a24b", "Company": "#2e86de", "Product": "#17becf", "Event": "#d62728",
          "Sector": "#7f7f7f"}

def plot_graph(G, nodes=None, edges=None, highlight=(), height=560, show_labels=True):
    """Vẽ đồ thị. `edges` là danh sách (u, v, rel); nếu None thì lấy mọi cạnh giữa `nodes`."""
    H = nx.DiGraph()
    if edges is None:
        nodes = list(nodes or G.nodes)
        for u, v, d in G.subgraph(nodes).edges(data=True): H.add_edge(u, v, rel=d["rel"])
    else:
        for u, v, r in edges: H.add_edge(u, v, rel=r)
    for n in (nodes or []): H.add_node(n)
    if H.number_of_nodes() == 0: return go.Figure().update_layout(height=200, annotations=[dict(text="Không có đồ thị con", showarrow=False)])
    pos = nx.spring_layout(H, seed=7, k=1.1 / max(H.number_of_nodes(), 1) ** 0.5, iterations=80)
    ex, ey, mx, my, mt = [], [], [], [], []
    for u, v, d in H.edges(data=True):
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
        mx.append((pos[u][0] + pos[v][0]) / 2); my.append((pos[u][1] + pos[v][1]) / 2); mt.append(d["rel"])
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#9aa4b2"), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers+text" if show_labels and len(mt) <= 30 else "markers", text=mt, textfont=dict(size=8, color="#6b7280"),
                             marker=dict(size=4, color="#9aa4b2"), hovertext=mt, hoverinfo="text", showlegend=False, textposition="top center"))
    types = {}
    for n in H.nodes: types.setdefault(G.nodes[n]["type"] if n in G else "Other", []).append(n)
    for t, ns in types.items():
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=t,
                                 text=ns if show_labels else None, textposition="bottom center", textfont=dict(size=10),
                                 marker=dict(size=[26 if n in highlight else 16 for n in ns], color=COLORS.get(t, "#999"),
                                             line=dict(width=[3 if n in highlight else 1 for n in ns], color="#111")),
                                 hovertext=[f"{n} ({t})" for n in ns], hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.05), plot_bgcolor="rgba(0,0,0,0)")
    return fig
