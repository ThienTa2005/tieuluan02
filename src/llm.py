"""Sinh câu trả lời có dẫn nguồn [M4], [S1] bằng Claude hoặc trích xuất offline."""
import os
import re
from .kg import norm

SYSTEM = ("Trợ lý học thuật tiếng Việt: chỉ dựa vào dữ liệu trong NGỮ CẢNH, "
          "dẫn nguồn theo mã tài liệu [M1], [M4], [S1] đúng với dữ liệu được trích. "
          "Nếu không đủ bằng chứng, nói rõ chưa có dữ liệu. "
          "Không tự tư vấn phác đồ cá nhân, chẩn đoán y tế hoặc khuyến nghị giao dịch.")

def _key():
    k = os.getenv("ANTHROPIC_API_KEY")
    if not k:
        try:
            import streamlit as st
            k = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    return k or None

def has_llm(): return bool(_key())

def _call(system, user, max_tokens=700):
    import anthropic
    client = anthropic.Anthropic(api_key=_key())
    output = client.messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-4-5"),
                                    max_tokens=max_tokens, system=system,
                                    messages=[{"role": "user", "content": user}])
    return "\n".join(chunk.text for chunk in output.content if getattr(chunk, "type", "") == "text")

def best_sentences(question, passages):
    qt = set(re.findall(r"\w+", norm(question)))
    for p in passages:
        candidates = re.split(r"(?<=[.;])\s+", p["text"])
        best = max(candidates, key=lambda s: len(qt & set(re.findall(r"\w+", norm(s))))) if candidates else p["text"]
        yield p["id"], best

def generate(question, facts, passages, extra=None):
    extra = extra or []
    ctx = "\n".join(["# Kết quả công cụ"] + [f"- {x}" for x in extra] +
                    ["# Sự kiện đồ thị (trích nguồn trong ngoặc vuông)"] + [f"- {f}" for f in facts] +
                    ["# Tài liệu có nguồn"] +
                    [f"[{p['id']}] {p['title']} ({p['source']}) {p['text']}" for p in passages])
    note = ""
    if has_llm():
        try: return _call(SYSTEM, f"NGỮ CẢNH:\n{ctx}\n\nCÂU HỎI: {question}")
        except Exception as exc: note = f"\n\n_LLM hiện không khả dụng ({type(exc).__name__}); dùng trích xuất offline._"
    chunks = []
    if extra: chunks.append("**Kết quả công cụ**\n" + "\n".join(f"- {e}" for e in extra))
    if facts: chunks.append("**Sự kiện từ đồ thị**\n" + "\n".join(f"- {f}" for f in facts[:8]))
    if passages:
        chunks.append("**Đoạn liên quan từ tài liệu (chế độ offline)**\n" +
                      "\n".join(f"- {s} [{pid}]" for pid,s in best_sentences(question, passages[:3])))
    if not chunks: chunks.append("Chưa tìm thấy bằng chứng trong kho dữ liệu; không thể kết luận.")
    return "\n\n".join(chunks) + note

def generate_plain(question):
    if not has_llm():
        raise RuntimeError("LLM thuần túy cần ANTHROPIC_API_KEY thật để benchmark.")
    return _call("Bạn là trợ lý học thuật tiếng Việt. Trả lời ngắn gọn.", question, max_tokens=400)
