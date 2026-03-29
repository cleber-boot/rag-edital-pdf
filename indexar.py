"""
indexar.py — processa PDFs e salva no ChromaDB
Versão Codespaces: aceita bytes diretamente

Correções para indexar vários arquivos:
  - BATCH_SIZE = 20 chunks por chamada (bem conservador)
  - BATCH_PAUSE = 4s entre lotes (100 req/min → 1 req/600ms seguro)
  - Pausa de segurança ao final de cada arquivo (PAUSA_ENTRE_ARQUIVOS)
  - Backoff exponencial com jitter em caso de 429
  - Progresso via callback opcional para exibir no Streamlit
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

# ── Parâmetros de throttle ───────────────────────────────────────────────────
# gemini-embedding-001: 100 req/min no plano gratuito
# Com BATCH_SIZE=20 e BATCH_PAUSE=4s → ~15 req/min → folgado mesmo com vários arquivos
BATCH_SIZE           = 20   # chunks por chamada embed_content
BATCH_PAUSE          = 4    # segundos entre lotes do mesmo arquivo
PAUSA_ENTRE_ARQUIVOS = 10   # segundos de descanso após terminar cada arquivo

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


def gerar_embeddings(
    cliente: genai.Client,
    chunks: list[str],
    callback=None,          # callback(lote_num, total_lotes) para atualizar UI
) -> list[list[float]]:
    """
    Gera embeddings em lotes com:
      - Retry automático (backoff exponencial via embed_com_retry)
      - Pausa fixa entre lotes para respeitar 100 req/min
      - Callback opcional para progresso no Streamlit
    """
    todos: list[list[float]] = []
    total_lotes = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        lote     = chunks[i : i + BATCH_SIZE]
        lote_num = i // BATCH_SIZE + 1

        logger.info("[embedding] Lote %d/%d — %d chunks...", lote_num, total_lotes, len(lote))

        resultado = embed_com_retry(
            cliente=cliente,
            model=EMBEDDING_MODEL,
            contents=lote,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        todos.extend(e.values for e in resultado.embeddings)

        if callback:
            callback(lote_num, total_lotes)

        # Pausa entre lotes — mesmo no último (protege o próximo arquivo)
        time.sleep(BATCH_PAUSE)

    return todos


def indexar_pdf_bytes(
    nome: str,
    conteudo: bytes,
    colecao,
    cliente: genai.Client,
    callback=None,           # callback(lote_num, total_lotes) opcional
    pausar_ao_final: bool = True,  # pausa após terminar para proteger próximo arquivo
) -> dict:
    """
    Indexa um PDF no ChromaDB.

    Parâmetros
    ----------
    nome           : nome do arquivo (usado como ID no banco)
    conteudo       : bytes do PDF
    colecao        : coleção ChromaDB de destino
    cliente        : cliente Gemini para embeddings
    callback       : função chamada a cada lote — útil para barra de progresso
    pausar_ao_final: se True, aguarda PAUSA_ENTRE_ARQUIVOS segundos ao terminar
                     (protege a cota quando há múltiplos arquivos em sequência)
    """
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

        embeddings = gerar_embeddings(cliente, chunks, callback=callback)

        colecao.add(
            ids=[f"{nome}_chunk_{j}" for j in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"arquivo": nome, "chunk": j} for j in range(len(chunks))],
        )

        if pausar_ao_final:
            logger.info("[indexar] Pausa de %ds após '%s'...", PAUSA_ENTRE_ARQUIVOS, nome)
            time.sleep(PAUSA_ENTRE_ARQUIVOS)

        return {"status": "ok", "chunks": len(chunks)}

    except Exception as e:
        logger.error("[indexar] Erro ao indexar '%s': %s", nome, e)
        return {"status": "erro", "erro": str(e)}