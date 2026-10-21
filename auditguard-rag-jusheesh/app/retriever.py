import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
from app.embeddings import embed_query

class HybridRetriever:
    """
    Hybrid Retriever combining Sparse Lexical Search (BM25) and Dense Vector Cosine Similarity
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, lance_store: Any = None, doc_id: str = ""):
        self.chunks = chunks
        self.embeddings = embeddings  # Shape: (N, 384), pre-normalized unit vectors
        self.lance_store = lance_store
        self.doc_id = doc_id
        
        self.node_id_to_indices: Dict[str, List[int]] = {}
        for c_idx, c in enumerate(chunks):
            nid = c.get("ast_node_id")
            if nid:
                self.node_id_to_indices.setdefault(nid, []).append(c_idx)
            for linked_id in c.get("linked_node_ids", []):
                self.node_id_to_indices.setdefault(linked_id, []).append(c_idx)

        if not chunks:
            self.bm25 = None
            return

        tokenized_corpus = [c["child_text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 4, k_rrf: int = 60, q_vec: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        if not self.chunks or len(self.chunks) == 0 or self.bm25 is None:
            return []

        # 1. Sparse BM25 Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        candidate_window = min(len(self.chunks), max(50, top_k * 8))
        bm25_rank_indices = np.argsort(bm25_scores)[::-1][:candidate_window]

        # 2. Dense Vector Dot Product Search (reuse q_vec if passed across multi-doc vault!)
        if q_vec is None:
            q_vec = embed_query(query)
        dense_scores = np.dot(self.embeddings, q_vec)
        dense_rank_indices = np.argsort(dense_scores)[::-1][:candidate_window]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_map = {}

        for rank, idx in enumerate(bm25_rank_indices):
            if bm25_scores[idx] > 0:
                rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (k_rrf + rank))

        for rank, idx in enumerate(dense_rank_indices):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (k_rrf + rank))

        # 4. Integrate Persistent LanceDB Vector Index Hits
        if self.lance_store and self.doc_id:
            try:
                lance_hits = self.lance_store.search(self.doc_id, q_vec, top_k=top_k * 2)
                for l_rank, hit in enumerate(lance_hits):
                    hit_id = hit.get("id")
                    if hit_id in self.node_id_to_indices:
                        for c_idx in self.node_id_to_indices[hit_id]:
                            rrf_map[c_idx] = rrf_map.get(c_idx, 0.0) + (1.0 / (k_rrf + l_rank))
            except Exception:
                pass

        sorted_indices = sorted(rrf_map.keys(), key=lambda i: rrf_map[i], reverse=True)

        # Deduplicate parent paragraphs ONLY within the same filename!
        seen_parents = set()
        results = []
        citation_num = 1

        for idx in sorted_indices:
            chunk = self.chunks[idx].copy()
            parent = chunk.get("parent_text", "")
            fname = chunk.get("filename", "")
            dedup_key = (fname, parent)
            if dedup_key in seen_parents:
                continue
            seen_parents.add(dedup_key)

            chunk["doc_id"] = self.doc_id
            chunk["citation_id"] = citation_num
            chunk["dense_score"] = float(dense_scores[idx])
            chunk["bm25_score"] = float(bm25_scores[idx])
            chunk["rrf_score"] = float(rrf_map[idx])
            results.append(chunk)
            citation_num += 1

            if len(results) >= top_k:
                break

        return results

    def escalate_page_scan(
        self,
        query: str,
        candidate_pages: List[int],
        ast: Any = None,
        min_threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        pages_to_check = candidate_pages
        if not pages_to_check:
            if ast and hasattr(ast, "get_all_pages"):
                pages_to_check = ast.get_all_pages()
            else:
                pages_to_check = sorted(list({c.get("page", 1) for c in self.chunks}))

        if not pages_to_check:
            return []

        import re
        q_vec = embed_query(query)
        stop_words = {"what", "is", "the", "are", "how", "for", "and", "with", "does", "can", "in", "of", "to", "a"}
        raw_tokens = [re.sub(r'[^a-zA-Z0-9]', '', t) for t in query.lower().split()]
        query_tokens = [t for t in raw_tokens if len(t) > 2 and t not in stop_words]
        if not query_tokens:
            query_tokens = [t for t in raw_tokens if len(t) >= 2]

        best_page = None
        best_score = 0.0
        best_dense = 0.0
        best_lexical = 0.0
        best_text = ""

        for p in pages_to_check:
            if ast and hasattr(ast, "get_page_text"):
                page_text = ast.get_page_text(p)
            else:
                page_chunks = [c["parent_text"] for c in self.chunks if c.get("page") == p]
                page_text = "\n\n".join(dict.fromkeys(page_chunks))

            if not page_text.strip():
                continue

            page_dense_scores = [
                float(np.dot(self.embeddings[i], q_vec))
                for i, c in enumerate(self.chunks)
                if c.get("page") == p
            ]
            max_dense = max(page_dense_scores) if page_dense_scores else 0.0

            page_lower = page_text.lower()
            matching_tokens = sum(1 for t in query_tokens if re.search(r'\b' + re.escape(t) + r'\b', page_lower))
            lexical_ratio = matching_tokens / max(1, len(query_tokens))

            pos_dense = max(0.0, max_dense)
            if lexical_ratio >= 0.50:
                composite_score = round(max(pos_dense, 0.40 + (lexical_ratio * 0.30)), 3)
            else:
                composite_score = round((pos_dense * 0.55 + lexical_ratio * 0.45), 3)

            if composite_score > best_score:
                best_score = composite_score
                best_page = p
                best_text = page_text
                best_dense = max_dense
                best_lexical = lexical_ratio

        if best_page is not None and best_score >= min_threshold:
            filename = ast.title if (ast and hasattr(ast, "title")) else self.chunks[0].get("filename", "Document")
            return [{
                "citation_id": 1,
                "doc_id": self.doc_id,
                "child_text": f"Whole-Page Scan (Page {best_page}): {best_text[:280]}...",
                "parent_text": best_text,
                "filename": filename,
                "page": best_page,
                "dense_score": round(best_dense, 3),
                "lexical_recall": round(best_lexical, 3),
                "bm25_score": round(best_lexical, 3),
                "page_relevance": round(best_score, 3),
                "rrf_score": round(best_score, 3),
                "escalated_from": "whole_page_scan"
            }]

        return []