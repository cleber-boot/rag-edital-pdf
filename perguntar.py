"""
perguntar.py — busca, rerank e montagem de contexto
Usa gemini-embedding-001 para embedding da pergunta
LLM: Groq llama-3.1-8b-instant (via app.py)
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
Se a informação não estiver nos trechos, diga isso claramente.
Quando um trecho se referir a uma imagem, figura ou página escaneada,
descreva o que foi encontrado com base no contexto disponível."""


def gerar_embedding_pergunta(cliente: genai.Client, pergunta: str) -> list[float]:
    """Gera embedding da pergunta no mesmo espaço vetorial multimodal."""
    resultado = cliente.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=pergunta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return resultado.embeddings[0].values


def buscar_trechos(
    pergunta: str,
    colecao,
    cliente: genai.Client,
    top_k: int = TOP_K
) -> list[dict]:
    """Busca os top_k trechos mais relevantes via embedding multimodal."""
    embedding  = gerar_embedding_pergunta(cliente, pergunta)
    resultados = colecao.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    trechos = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    ):
        tipo = meta.get("tipo", "texto")
        trechos.append({
            "texto":     doc,
            "arquivo":   meta["arquivo"],
            "tipo":      tipo,
            "relevancia": round(1 - dist, 3),
            "meta":      meta
        })
    return trechos


def rerankar_trechos(
    pergunta: str,
    trechos: list[dict],
    ranker: Ranker,
    top_n: int = TOP_N_RERANK
) -> list[dict]:
    """Reordena os trechos usando Flashrank."""
    if not trechos:
        return []

    passages       = [{"id": i, "text": t["texto"]} for i, t in enumerate(trechos)]
    rerank_request = RerankRequest(query=pergunta, passages=passages)
    resultados     = ranker.rerank(rerank_request)

    return [
        {
            "texto":             trechos[r["id"]]["texto"],
            "arquivo":           trechos[r["id"]]["arquivo"],
            "tipo":              trechos[r["id"]]["tipo"],
            "relevancia":        trechos[r["id"]]["relevancia"],
            "relevancia_rerank": round(r["score"], 4),
            "meta":              trechos[r["id"]]["meta"]
        }
        for r in resultados[:top_n]
    ]


def montar_contexto(trechos: list) -> str:
    """Monta o bloco de contexto para o prompt do LLM."""
    contexto = ""
    for i, t in enumerate(trechos, 1):
        tipo  = t.get("tipo", "texto")
        label = {
            "texto":           "Texto",
            "pagina_escaneada": "Página escaneada",
            "figura":          "Figura"
        }.get(tipo, "Conteúdo")

        contexto += (
            f"\n--- Trecho {i} [{label}] "
            f"(arquivo: {t['arquivo']}, "
            f"relevância rerank: {t['relevancia_rerank']}) ---\n"
        )
        contexto += t["texto"] + "\n"
    return contexto