"""
Чистое Python-векторное хранилище на TF-IDF.
Без внешних зависимостей — использует JSON для хранения и TF-IDF + косинусное сходство для поиска.
"""
import json
import math
import os
import re
import uuid
from collections import Counter
from typing import Dict, List, Optional, Any


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'don', 'now', 'и', 'в', 'на', 'не', 'что', 'это', 'как', 'по', 'для',
            'от', 'из', 'к', 'с', 'но', 'а', 'о', 'у', 'ли', 'же', 'ни', 'бы',
            'все', 'его', 'её', 'их', 'при', 'до', 'за', 'над', 'под', 'без'}
    return [t for t in tokens if t not in stop and len(t) > 1]


class LocalVectorStore:
    def __init__(
        self,
        collection_name: str = "code_knowledge",
        persist_directory: str = "data/knowledge",
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_directory
        self.db_path = os.path.join(persist_directory, f"{collection_name}.json")
        self.documents: List[Dict[str, Any]] = []
        self.idf: Dict[str, float] = {}
        self._dirty = False

        os.makedirs(persist_directory, exist_ok=True)
        self._load()

        print(f"✅ Векторное хранилище готово (TF-IDF, чистый Python)")
        print(f"   Коллекция: {collection_name} | Документов: {len(self.documents)}")

    # ── Persistence ────────────────────────────────────────────────
    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.documents = data.get('documents', [])
                self.idf = data.get('idf', {})
            except Exception:
                self.documents = []
                self.idf = {}

    def save(self):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'idf': self.idf,
            }, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def _rebuild_idf(self):
        n = len(self.documents)
        if n == 0:
            self.idf = {}
            return
        doc_freq: Counter = Counter()
        for doc in self.documents:
            tokens = set(_tokenize(doc.get('text', '')))
            for t in tokens:
                doc_freq[t] += 1
        self.idf = {}
        for term, df in doc_freq.items():
            self.idf[term] = math.log((n + 1) / (df + 1)) + 1

    # ── TF-IDF helpers ────────────────────────────────────────────
    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        vec = {}
        for term, count in tf.items():
            idf = self.idf.get(term, math.log(len(self.documents) + 2))
            vec[term] = (count / total) * idf
        return vec

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Public API ────────────────────────────────────────────────
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for i, text in enumerate(texts):
            self.documents.append({
                'id': ids[i],
                'text': text,
                'metadata': metadatas[i],
            })

        self._rebuild_idf()
        self._dirty = True
        self.save()
        return ids

    def search(
        self,
        query: str,
        n_results: int = 3,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        self._rebuild_idf()
        q_vec = self._tfidf_vector(query)

        scored = []
        for doc in self.documents:
            if where:
                meta = doc.get('metadata', {})
                match = all(meta.get(k) == v for k, v in where.items())
                if not match:
                    continue
            d_vec = self._tfidf_vector(doc.get('text', ''))
            sim = self._cosine(q_vec, d_vec)
            scored.append((sim, doc))

        scored.sort(key=lambda x: -x[0])
        results = []
        for sim, doc in scored[:n_results]:
            results.append({
                'id': doc['id'],
                'text': doc['text'],
                'metadata': doc.get('metadata', {}),
                'distance': round(1.0 - sim, 4),
            })
        return results

    def delete_document(self, doc_id: str) -> bool:
        before = len(self.documents)
        self.documents = [d for d in self.documents if d.get('id') != doc_id]
        if len(self.documents) < before:
            self._rebuild_idf()
            self.save()
            return True
        return False

    def count(self) -> int:
        return len(self.documents)

    def clear(self):
        self.documents = []
        self.idf = {}
        self.save()
