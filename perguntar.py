"""
perguntar.py — busca, rerank e montagem de contexto
Versão Codespaces
"""

from google import genai
from google.genai import types
from flashrank import Ranker, RerankRequest

EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K           = 10
TOP_N_RERANK    = 5

SYSTEM_PROMPT = """Você é um assistente de estudo especializado.
Responda sempre em português, de forma clara e didática.
Use apenas as informações dos trechos fornecidos para responder.
Ao final, cite de qual arquivo cada informação veio.
Se a informação não estiver nos trechos, diga isso claramente."""


def gerar_embedding_pergunta(cliente: genai.Client, pergunta: str) -> list[float]:
    resultado = cliente.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=pergunta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return resultado.embeddings[0].values


def buscar_trechos(pergunta: str, colecao, cliente: genai.Client, top_k: int = TOP_K) -> list[dict]:
    embedding  = gerar_embedding_pergunta(cliente, pergunta)
    resultados = colecao.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "texto":    doc,
            "arquivo":  meta["arquivo"],
            "relevancia": round(1 - dist, 3)
        }
        for doc, meta, dist in zip(
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0]
        )
    ]


def rerankar_trechos(pergunta: str, trechos: list[dict], ranker: Ranker, top_n: int = TOP_N_RERANK) -> list[dict]:
    if not trechos:
        return []
    passages       = [{"id": i, "text": t["texto"]} for i, t in enumerate(trechos)]
    rerank_request = RerankRequest(query=pergunta, passages=passages)
    resultados     = ranker.rerank(rerank_request)

    return [
        {
            "texto":             trechos[r["id"]]["texto"],
            "arquivo":           trechos[r["id"]]["arquivo"],
            "relevancia":        trechos[r["id"]]["relevancia"],
            "relevancia_rerank": round(r["score"], 4)
        }
        for r in resultados[:top_n]
    ]


def montar_contexto(trechos: list) -> str:
    contexto = ""
    for i, t in enumerate(trechos, 1):
        contexto += f"\n--- Trecho {i} (arquivo: {t['arquivo']}, relevância rerank: {t['relevancia_rerank']}) ---\n"
        contexto += t["texto"] + "\n"
    return contexto
