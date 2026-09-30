"""Barrier 1 (retrieval): measures P_ret(rho) = Recall@k of the adversarial
passage after obfuscation, for BM25 (sparse) and dense retrievers (Contriever, E5).

Design (Sec. IV-C of the paper): for each query q with a crafted adversarial
passage d_adv (clean at rho=0, retrievable by construction following Chang et
al. [7]), inject d_adv into the corpus, run the retriever, and record the rank of
d_adv. Recall@k = P(rank(d_adv) <= k) over the query set.
"""
import math
from collections import Counter
import numpy as np
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer
    import torch
except Exception:  # allow import on machines without the dense stack
    SentenceTransformer = None


def _tok(s):
    return s.lower().split()


class BM25Retriever:
    """BM25 over the distractor corpus. The adversarial passage is scored with
    the corpus's own IDF/avgdl statistics, which is exact up to the (negligible
    for N >= 1000) change one extra document makes to IDF. For tiny corpora
    (the demo) we refit exactly with the passage included."""
    name = "BM25"

    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.bm25 = BM25Okapi([_tok(d) for d in self.corpus])
        n = len(self.corpus)
        self._unseen_idf = math.log((n + 0.5) / 1.5)  # idf of a term only d_adv has
        self._qcache = {}

    def _adv_score(self, q_tokens, adv_passage):
        d = _tok(adv_passage)
        tf, dl = Counter(d), len(d)
        k1, b, avgdl = self.bm25.k1, self.bm25.b, self.bm25.avgdl
        s = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if f:
                idf = self.bm25.idf.get(t, self._unseen_idf)
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        return s

    def rank_of(self, query, adv_passage):
        if len(self.corpus) < 1000:  # exact refit for the demo corpus
            docs = self.corpus + [adv_passage]
            scores = BM25Okapi([_tok(d) for d in docs]).get_scores(_tok(query))
            return int(np.sum(scores[:-1] > scores[-1])) + 1
        if query not in self._qcache:
            self._qcache[query] = self.bm25.get_scores(_tok(query))
        adv = self._adv_score(_tok(query), adv_passage)
        return int(np.sum(self._qcache[query] > adv)) + 1  # 1-indexed

    def ranks(self, queries, advs):
        return [self.rank_of(q, a) for q, a in zip(queries, advs)]

    def gold_rank(self, query, gold_idx):
        """Rank of an existing corpus document (for the clean-utility check)."""
        if query not in self._qcache:
            self._qcache[query] = self.bm25.get_scores(_tok(query))
        sc = self._qcache[query]
        return int(np.sum(sc > sc[gold_idx])) + 1


class DenseRetriever:
    def __init__(self, corpus, model_name="facebook/contriever",
                 query_prefix="", doc_prefix="", device=None):
        assert SentenceTransformer is not None, "pip install sentence-transformers torch"
        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus = list(corpus)
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.emb = self.model.encode([doc_prefix + d for d in self.corpus],
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=True)
        self._qcache = {}

    def _qvecs(self, queries):
        todo = [q for q in dict.fromkeys(queries) if q not in self._qcache]
        if todo:
            v = self.model.encode([self.query_prefix + q for q in todo],
                                  convert_to_numpy=True, normalize_embeddings=True,
                                  batch_size=64, show_progress_bar=False)
            self._qcache.update(zip(todo, v))
        return np.stack([self._qcache[q] for q in queries])

    def ranks(self, queries, advs):
        """Batched: rank of each adversarial passage for its own query."""
        qv = self._qvecs(queries)
        av = self.model.encode([self.doc_prefix + a for a in advs],
                               convert_to_numpy=True, normalize_embeddings=True,
                               batch_size=64, show_progress_bar=False)
        corpus_scores = qv @ self.emb.T                 # (n_q, N)
        adv_scores = np.sum(qv * av, axis=1)            # (n_q,)
        return (np.sum(corpus_scores > adv_scores[:, None], axis=1) + 1).tolist()

    def rank_of(self, query, adv_passage):
        return self.ranks([query], [adv_passage])[0]

    def gold_rank(self, query, gold_idx):
        qv = self._qvecs([query])[0]
        sc = self.emb @ qv
        return int(np.sum(sc > sc[gold_idx])) + 1


def build_retriever(kind, corpus, device=None):
    if kind == "bm25":
        return BM25Retriever(corpus)
    if kind == "contriever":
        return DenseRetriever(corpus, "facebook/contriever", device=device)
    if kind == "e5":
        return DenseRetriever(corpus, "intfloat/e5-base-v2",
                              query_prefix="query: ", doc_prefix="passage: ",
                              device=device)
    raise ValueError(kind)
