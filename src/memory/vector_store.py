"""
Локальное векторное хранилище на основе ChromaDB.
"""
import os
import uuid
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class LocalVectorStore:
    def __init__(
        self,
        collection_name: str = "code_knowledge",
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        print(f"✅ Векторное хранилище готово. Коллекция: {collection_name}")
        print(f"   Количество документов: {self.collection.count()}")

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        embeddings = self.embedding_model.encode(texts).tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return ids

    def search(
        self,
        query: str,
        n_results: int = 3,
        where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'id': results['ids'][0][i],
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i]
                })

        return documents

    def delete_document(self, doc_id: str) -> bool:
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except:
            return False

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)