"""
indexar.py — processa PDFs e salva no ChromaDB
Versão Codespaces: aceita bytes diretamente
"""

import time
import tempfile

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types

CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE      = 80
BATCH_PAUSE     = 65
MAX_RETRY       = 5
RETRY_WAIT      = 60

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)


def extrair_texto_bytes(conteudo: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(conteudo)
        tmp.flush()
        doc   = fitz.open(tmp.name)
        texto = "".join(pagina.get_text() for pagina in doc)
    return texto


def _embed_lote_com_retry(cliente: genai.Client, lote: list[str]) -> list[list[float]]:
    for tentativa in range(1, MAX_RETRY + 1):
        try:
            resultado = cliente.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=lote,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return [e.values for e in resultado.embeddings]
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[429] Aguardando {RETRY_WAIT}s... (tentativa {tentativa}/{MAX_RETRY})")
                time.sleep(RETRY_WAIT)
            else:
                raise
    raise RuntimeError(f"Falha ao gerar embeddings após {MAX_RETRY} tentativas.")


def gerar_embeddings(cliente: genai.Client, chunks: list[str]) -> list[list[float]]:
    todos       = []
    total_lotes = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        lote     = chunks[i:i + BATCH_SIZE]
        lote_num = i // BATCH_SIZE + 1
        print(f"[embedding] Lote {lote_num}/{total_lotes} — {len(lote)} chunks...")
        todos.extend(_embed_lote_com_retry(cliente, lote))

        if i + BATCH_SIZE < len(chunks):
            print(f"[embedding] Pausa de {BATCH_PAUSE}s...")
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
        print(f"[indexar] '{nome}' → {len(chunks)} chunks gerados.")

        embeddings = gerar_embeddings(cliente, chunks)

        colecao.add(
            ids=[f"{nome}_chunk_{j}" for j in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"arquivo": nome, "chunk": j} for j in range(len(chunks))]
        )
        return {"status": "ok", "chunks": len(chunks)}

    except Exception as e:
        return {"status": "erro", "erro": str(e)}
