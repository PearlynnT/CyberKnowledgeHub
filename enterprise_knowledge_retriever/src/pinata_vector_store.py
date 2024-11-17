import json
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from enterprise_knowledge_retriever.src.pinata_client import PinataClient

class PinataVectorStore(VectorStore):
    def __init__(self, embeddings: Embeddings, pinata_client: PinataClient):
        self._embeddings = embeddings
        self.pinata_client = pinata_client

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: Embeddings):
        self._embeddings = value

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        embeddings = self.embeddings.embed_documents(texts)
        documents = []
        cids = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            doc = {
                'text': text,
                'embedding': embedding,
                'metadata': metadata
            }
            documents.append(doc)

            # Convert the document to JSON
            json_content = json.dumps(doc)

            # Upload the JSON to Pinata
            try:
                response = self.pinata_client.pin_json_to_ipfs(json_content, f"document_{i}")
                cid = response['IpfsHash']
                cids.append(cid)
            except Exception as e:
                print(f"Error uploading document {i} to Pinata: {str(e)}")
                cids.append(None)

        return cids

    # def similarity_search(self, query: str, k: int = 4) -> List[Document]:
    #     query_embedding = self.embeddings.embed_query(query)
        
    #     # Retrieve all documents from Pinata
    #     all_docs = self.pinata_client.list_files()
        
    #     # Compute similarities and sort
    #     similarities = []
    #     for doc in all_docs:
    #         doc_content = self.pinata_client.get_file_content(doc['ipfs_pin_hash'])
    #         doc_data = json.loads(doc_content)
    #         similarity = self.compute_similarity(query_embedding, doc_data['embedding'])
    #         similarities.append((similarity, doc_data))
        
    #     similarities.sort(key=lambda x: x[0], reverse=True)
    #     print(similarities)
        
    #     # Return top k documents
    #     return [Document(page_content=doc['text'], metadata=doc['metadata']) for _, doc in similarities[:k]]

    def _select_relevance_score_fn(self):
        return lambda x: x

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, score_threshold: float = None
    ) -> List[Tuple[Document, float]]:
        query_embedding = self.embeddings.embed_query(query)
        all_docs = self.pinata_client.list_files()
        
        similarities = []
        for doc in all_docs:
            try:
                doc_content = self.pinata_client.get_file_content(doc['ipfs_pin_hash'])
                doc_data = json.loads(doc_content)
                similarity = self.compute_similarity(query_embedding, doc_data['embedding'])
                if score_threshold is None or similarity >= score_threshold:
                    similarities.append((Document(page_content=doc_data['text'], metadata=doc_data['metadata']), similarity))
            except json.JSONDecodeError:
                print(f"Error decoding JSON for document {doc['ipfs_pin_hash']}")
            except KeyError as e:
                print(f"Missing key in document {doc['ipfs_pin_hash']}: {str(e)}")
            except Exception as e:
                print(f"Error processing document {doc['ipfs_pin_hash']}: {str(e)}")
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        results = self.similarity_search_with_relevance_scores(query, k)
        return [doc for doc, _ in results]

    def as_retriever(self, search_type: str = "similarity", **kwargs):
        if search_type == "similarity":
            return VectorStoreRetriever(vectorstore=self, search_kwargs=kwargs)
        elif search_type == "similarity_score_threshold":
            return VectorStoreRetriever(
                vectorstore=self,
                search_type="similarity_score_threshold",
                search_kwargs=kwargs
            )
        else:
            raise ValueError(f"search_type {search_type} not supported")

    # def similarity_search(self, query: str, k: int = 4) -> List[Document]:
    #     try:
    #         query_embedding = self.embeddings.embed_query(query)
    #         all_docs = self.pinata_client.list_files()
    #         print(f"Retrieved {len(all_docs)} documents from Pinata")
            
    #         similarities = []
    #         for doc in all_docs:
    #             try:
    #                 doc_content = self.pinata_client.get_file_content(doc['ipfs_pin_hash'])
    #                 doc_data = json.loads(doc_content)
    #                 similarity = self.compute_similarity(query_embedding, doc_data['embedding'])
    #                 similarities.append((similarity, doc_data))
    #             except Exception as e:
    #                 print(f"Error processing document {doc['ipfs_pin_hash']}: {str(e)}")
            
    #         similarities.sort(key=lambda x: x[0], reverse=True)
    #         print(f"Computed similarities for {len(similarities)} documents")
            
    #         return [Document(page_content=doc['text'], metadata=doc['metadata']) for _, doc in similarities[:k]]
    #     except Exception as e:
    #         print(f"Error in similarity_search: {str(e)}")
    #         raise

    # @staticmethod
    # def compute_similarity(vec1, vec2):
    #     # Implement cosine similarity or other similarity measure
    #     # For simplicity, we'll use dot product here
    #     return sum(a * b for a, b in zip(vec1, vec2))

    @staticmethod
    def compute_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: List[Dict[str, Any]] = None, **kwargs) -> "PinataVectorStore":
        """Create a PinataVectorStore from a list of texts."""
        pinata_client = kwargs.get('pinata_client', PinataClient())
        instance = cls(embedding, pinata_client)
        instance.add_texts(texts, metadatas)
        return instance

from langchain.schema import BaseRetriever

class VectorStoreRetriever(BaseRetriever):
    def __init__(self, vectorstore: PinataVectorStore, search_type="similarity", search_kwargs=None):
        self._vectorstore = vectorstore
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}

    @property
    def vectorstore(self) -> PinataVectorStore:
        return self._vectorstore

    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            return self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
            return [doc for doc, _ in docs_and_scores]
        else:
            raise ValueError(f"search_type {self.search_type} not supported")

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)