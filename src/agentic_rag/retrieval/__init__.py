from .pipeline import Passage, RAGPipeline, RetrievalResult, rrf_fuse

__all__ = ["Passage", "RAGPipeline", "RetrievalResult", "rrf_fuse", "build_pipeline"]


def build_pipeline(settings) -> RAGPipeline:
    from .embeddings import make_embedder
    from .reranker import make_reranker
    from .stores import MemoryStore

    embedder = make_embedder(settings)
    if settings.vector_store == "pgvector":
        from .pg_store import PgVectorStore
        store = PgVectorStore(settings.database_url, embedder.dim)
    else:
        store = MemoryStore(settings.index_dir, embedder.dim)
        store.load()
    return RAGPipeline(store, embedder, make_reranker(settings), settings)
