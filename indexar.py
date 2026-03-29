"""
indexar.py — processa PDFs e salva no ChromaDB
Versão Codespaces: aceita bytes diretamente

Melhorias de rate limiting:
  - Backoff exponencial com jitter via gemini_retry.embed_com_retry
  - Throttle inteligente entre lotes: pausa proporcional ao tamanho do lote
  - BATCH_SIZE reduzido para 50 (seguro para 100 req/min do plano gratuito)
  - Pausa mínima entre lotes de 35s (conservador para evitar 429 acumulado)
"""

import time
import tempfile
import logging
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types

from gemini_retry import embed_com_retry

logger = logging.getLogger(__name__)

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "gemini-embedding-001"

# ── Limites do plano gratuito: 100 req/min para embeddings ──────────────────
# Cada chamada embed_content com N textos = 1 requisição.
# Lotes de 50 chunks → máximo 2 chamadas/min com folga.
BATCH_SIZE  = 50   # chunks por chamada de embed_content
BATCH_PAUSE = 35   # segundos entre lotes (conservador; ajuste se tiver plano pago)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)


def extrair_texto_bytes(conteudo: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(conteudo)
        tmp.flush()
        doc = fitz.open(tmp.name)
        texto = "".join(pagina.get_text() for pagina in doc)
    return texto


def gerar_embeddings(cliente: genai.Client, chunks: list[str]) -> list[list[float]]:
    """
    Gera embeddings em lotes com retry automático (backoff exponencial)
    e pausa entre lotes para não ultrapassar 100 req/min.
    """
    todos: list[list[float]] = []
    total_lotes = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        lote = chunks[i : i + BATCH_SIZE]
        lote_num = i // BATCH_SIZE + 1
        logger.info("[embedding] Lote %d/%d — %d chunks...", lote_num, total_lotes, len(lote))

        resultado = embed_com_retry(
            cliente=cliente,
            model=EMBEDDING_MODEL,
            contents=lote,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        todos.extend(e.values for e in resultado.embeddings)

        # Pausa entre lotes (exceto no último)
        if i + BATCH_SIZE < len(chunks):
            logger.info("[embedding] Pausa de %ds entre lotes...", BATCH_PAUSE)
            time.sleep(BATCH_PAUSE)

    return todos


def indexar_pdf_bytes(nome: str, conteudo: bytes, colecao, cliente: genai.Client) -> dict:
    # Verifica se já foi indexado
    if colecao.count() > 0:
        ja_indexados = {m.get("arquivo") for m in colecao.get()["metadatas"]}
        if nome in ja_indexados:
            return {"status": "ja_indexado", "chunks": 0}

    try:
        texto = extrair_texto_bytes(conteudo)
        if not texto.strip():
            return {"status": "erro", "erro": "PDF vazio ou só imagens"}

        chunks = splitter.split_text(texto)
        logger.info("[indexar] '%s' → %d chunks gerados.", nome, len(chunks))

        embeddings = gerar_embeddings(cliente, chunks)

        colecao.add(
            ids=[f"{nome}_chunk_{j}" for j in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"arquivo": nome, "chunk": j} for j in range(len(chunks))],
        )
        return {"status": "ok", "chunks": len(chunks)}

    except Exception as e:
        logger.error("[indexar] Erro ao indexar '%s': %s", nome, e)
        return {"status": "erro", "erro": str(e)}
