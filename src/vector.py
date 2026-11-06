"""Chỉ mục Vector: embedding đa ngôn ngữ (tuỳ chọn) hoặc TF-IDF dự phòng.

Embedding đa ngôn ngữ cần cài requirements-semantic.txt; cấu hình
EMBEDDING_BACKEND=semantic (bắt buộc), hoặc auto (thử và fallback rõ ràng).
TF-IDF là vector hoá từ vựng, KHÔNG được ghi thành 'semantic embedding'.
"""
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .kg import norm

class VectorIndex:
    def __init__(self, docs, backend=None):
        self.docs = docs
        self.backend = (backend or os.getenv("EMBEDDING_BACKEND", "tfidf")).lower()
        texts = [d["title"] + ". " + d["text"] for d in docs]
        self.model = None
        self.detail = "TF-IDF ký tự (truy xuất từ vựng, không phải embedding ngữ nghĩa)"
        if self.backend in ("semantic", "auto"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
                self.X = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
                self.detail = "Embedding đa ngôn ngữ (Sentence-Transformers, cosine similarity)"
            except Exception as exc:
                if self.backend == "semantic":
                    raise RuntimeError("Không khởi tạo được embedding ngữ nghĩa. Cài requirements-semantic.txt và kiểm tra mạng/cache.") from exc
        if self.model is None:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), sublinear_tf=True)
            self.X = self.vec.fit_transform([norm(t) for t in texts])

    def scores(self, query):
        if self.model is not None:
            v = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            return np.asarray((v @ self.X.T)[0], dtype=float)
        return cosine_similarity(self.vec.transform([norm(query)]), self.X)[0]

    def search(self, query, k=4, domain=None):
        scores = self.scores(query)
        order = np.argsort(-scores)
        found = []
        for i in order:
            if domain is not None and self.docs[i]["domain"] != domain: continue
            found.append({**self.docs[i], "score": float(scores[i])})
            if len(found) >= k: break
        return found
