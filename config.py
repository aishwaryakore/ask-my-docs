CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

RETRIEVER_K = 8                   
ENSEMBLE_WEIGHTS = [0.5, 0.5]   # [BM25, semantic]
RERANKER_TOP_N = 4          
RERANKER_SCORE_THRESHOLD = 0.5    
RERANKER_MODEL = "rerank-english-v3.0"

QA_MODEL = "gpt-4o-mini"      
REWRITER_MODEL = "gpt-3.5-turbo"   
REWRITER_TEMPERATURE = 0  

EMBEDDING_MODEL = "text-embedding-3-small"