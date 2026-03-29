"""
indexar.py — processa PDFs e salva no ChromaDB

Fluxo por página:
  1. Texto nativo, sem imagens embutidas   → extrai texto diretamente
  2. Texto nativo + imagens embutidas      → extrai texto + Gemini Vision descreve cada figura
  3. Sem texto nativo, Tesseract legível   → OCR via Tesseract (por+eng)
  4. Sem texto nativo, OCR ilegível        → Gemini Vision descreve a página inteira

Todo o conteúdo extraído vira texto plano → chunks → embeddings de texto (gemini-embedding-2-preview).

Compatível com app.py: mesma assinatura de indexar_pdf_bytes() incluindo
os parâmetros callback= e pausar_ao_final= introduzidos na versão atual.
"""

import io
import time
import base64
import tempfile

import fitz                    # PyMuPDF
from PIL import Image

try:
    import pytesseract
    _TESSERACT_OK = True
except ImportError:
    _TESSERACT_OK = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types

# ── configuração ──────────────────────────────────────────────────────────────

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "gemini-embedding-001"
VISION_MODEL    = "gemini-1.5-flash"

BATCH_SIZE  = 20    # chunks por lote de embedding
BATCH_PAUSE = 10    # segundos entre lotes
MAX_RETRY   = 7
RETRY_WAIT  = 60    # segundos base ao receber 429

# Mínimo de caracteres para considerar que uma página tem texto nativo
MIN_CHARS_NATIVO = 50

# Mínimo de caracteres retornados pelo Tesseract para aceitar o OCR
MIN_CHARS_OCR = 20

# DPI para rasterizar páginas enviadas ao Tesseract ou ao Vision
RASTER_DPI = 200

# Tamanho mínimo de imagem embutida (pixels) — ignora ícones e decorações
MIN_IMG_PX = 100

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)


# ── helpers de imagem ─────────────────────────────────────────────────────────

def _pagina_para_png_bytes(pagina: fitz.Page, dpi: int = RASTER_DPI) -> bytes:
    """Rasteriza uma página PyMuPDF e retorna PNG em bytes."""
    zoom = dpi / 72
    pix  = pagina.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def _figura_para_png_bytes(doc: fitz.Document, xref: int) -> bytes | None:
    """Extrai uma imagem embutida pelo xref e converte para PNG bytes."""
    try:
        info = doc.extract_image(xref)
        img  = Image.open(io.BytesIO(info["image"])).convert("RGB")
        if img.width < MIN_IMG_PX or img.height < MIN_IMG_PX:
            return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ── Gemini Vision ─────────────────────────────────────────────────────────────

def _vision(cliente: genai.Client, png_bytes: bytes, prompt: str) -> str:
    """
    Envia uma imagem PNG ao Gemini Vision e retorna a descrição textual.
    Retry automático em 429.
    """
    part_img = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
    part_txt = types.Part.from_text(text=prompt)

    for tentativa in range(1, MAX_RETRY + 1):
        try:
            resp = cliente.models.generate_content(
                model=VISION_MODEL,
                contents=[types.Content(role="user", parts=[part_img, part_txt])]
            )
            return resp.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[Vision 429] Aguardando {RETRY_WAIT}s... "
                      f"(tentativa {tentativa}/{MAX_RETRY})")
                time.sleep(RETRY_WAIT)
            else:
                raise
    return "[Vision: falha após múltiplas tentativas]"


# ── extração por página ───────────────────────────────────────────────────────

def _extrair_pagina(
    doc: fitz.Document,
    pagina: fitz.Page,
    num: int,
    cliente: genai.Client,
) -> str:
    """
    Aplica o fluxo de decisão a uma única página e retorna texto extraído.

    Caminho 1 — texto nativo, sem figuras   → get_text()
    Caminho 2 — texto nativo + figuras      → get_text() + Vision por figura
    Caminho 3 — escaneada, OCR legível      → Tesseract
    Caminho 4 — escaneada ilegível/gráfico  → Vision na página inteira
    """
    texto_nativo = pagina.get_text().strip()
    tem_texto    = len(texto_nativo) >= MIN_CHARS_NATIVO
    figuras_xref = [img[0] for img in pagina.get_images(full=True)]
    tem_figuras  = len(figuras_xref) > 0

    # ── Caminho 1 ─────────────────────────────────────────────────────────────
    if tem_texto and not tem_figuras:
        print(f"  [pág {num}] Caminho 1 — texto nativo")
        return texto_nativo

    # ── Caminho 2 ─────────────────────────────────────────────────────────────
    if tem_texto and tem_figuras:
        print(f"  [pág {num}] Caminho 2 — texto + {len(figuras_xref)} figura(s) via Vision")
        partes = [texto_nativo]
        for idx, xref in enumerate(figuras_xref, 1):
            png = _figura_para_png_bytes(doc, xref)
            if png is None:
                continue
            descricao = _vision(
                cliente, png,
                "Descreva detalhadamente o conteúdo desta figura: "
                "gráficos, tabelas, legendas e qualquer texto visível."
            )
            partes.append(f"[Figura {idx} — página {num}]: {descricao}")
        return "\n\n".join(partes)

    # ── Sem texto nativo: tenta Tesseract ─────────────────────────────────────
    png_pagina = _pagina_para_png_bytes(pagina)
    ocr_texto  = ""

    if _TESSERACT_OK:
        try:
            img_pil   = Image.open(io.BytesIO(png_pagina))
            ocr_texto = pytesseract.image_to_string(img_pil, lang="por+eng").strip()
        except Exception as e:
            print(f"  [pág {num}] Tesseract erro: {e}")

    # ── Caminho 3 ─────────────────────────────────────────────────────────────
    if len(ocr_texto) >= MIN_CHARS_OCR:
        print(f"  [pág {num}] Caminho 3 — OCR Tesseract ({len(ocr_texto)} chars)")
        return ocr_texto

    # ── Caminho 4 ─────────────────────────────────────────────────────────────
    print(f"  [pág {num}] Caminho 4 — Gemini Vision (página completa)")
    return _vision(
        cliente, png_pagina,
        "Esta é uma página de documento PDF. Descreva todo o conteúdo visível: "
        "texto, tabelas, gráficos, diagramas e imagens. "
        "Preserve números, datas e termos técnicos."
    )


# ── embeddings ────────────────────────────────────────────────────────────────

def _embed_com_retry(cliente: genai.Client, texto: str, task_type: str) -> list[float]:
    for tentativa in range(1, MAX_RETRY + 1):
        try:
            resultado = cliente.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texto,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return resultado.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[embed 429] Aguardando {RETRY_WAIT}s... "
                      f"(tentativa {tentativa}/{MAX_RETRY})")
                time.sleep(RETRY_WAIT)
            else:
                raise
    raise RuntimeError(f"Falha no embedding após {MAX_RETRY} tentativas.")


def _gerar_embeddings(
    cliente: genai.Client,
    chunks: list[str],
    callback=None,
) -> list[list[float]]:
    todos       = []
    total_lotes = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        lote     = chunks[i : i + BATCH_SIZE]
        lote_num = i // BATCH_SIZE + 1
        print(f"[embedding] Lote {lote_num}/{total_lotes} — {len(lote)} chunks...")

        for chunk in lote:
            todos.append(_embed_com_retry(cliente, chunk, "RETRIEVAL_DOCUMENT"))

        if callback:
            callback(lote_num, total_lotes)

        if i + BATCH_SIZE < len(chunks):
            time.sleep(BATCH_PAUSE)

    return todos


# ── ponto de entrada público ──────────────────────────────────────────────────

def indexar_pdf_bytes(
    nome: str,
    conteudo: bytes,
    colecao,
    cliente: genai.Client,
    callback=None,
    pausar_ao_final: bool = False,
) -> dict:
    """
    Processa um PDF (bytes) aplicando o fluxo de 4 caminhos por página
    e indexa os chunks resultantes no ChromaDB.

    Retorna:
        {"status": "ok",          "chunks": N}
        {"status": "ja_indexado", "chunks": 0}
        {"status": "erro",        "erro":   "mensagem"}
    """
    # ── duplicata ──────────────────────────────────────────────────────────────
    if colecao.count() > 0:
        ja_indexados = {m.get("arquivo") for m in colecao.get()["metadatas"]}
        if nome in ja_indexados:
            return {"status": "ja_indexado", "chunks": 0}

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(conteudo)
            tmp.flush()
            doc = fitz.open(tmp.name)
            n_paginas = len(doc)
            print(f"[indexar] '{nome}' — {n_paginas} página(s)")

            partes = []
            for num, pagina in enumerate(doc, 1):
                texto = _extrair_pagina(doc, pagina, num, cliente)
                if texto.strip():
                    partes.append(f"[Página {num}]\n{texto}")

        texto_completo = "\n\n".join(partes)
        if not texto_completo.strip():
            return {"status": "erro", "erro": "Nenhum conteúdo extraído do PDF"}

        chunks = splitter.split_text(texto_completo)
        print(f"[indexar] '{nome}' → {len(chunks)} chunks gerados")

        embeddings = _gerar_embeddings(cliente, chunks, callback=callback)

        colecao.add(
            ids        = [f"{nome}_chunk_{j}" for j in range(len(chunks))],
            embeddings = embeddings,
            documents  = chunks,
            metadatas  = [{"arquivo": nome, "tipo": "texto", "chunk": j}
                          for j in range(len(chunks))]
        )

        if pausar_ao_final:
            print("[indexar] Pausa de 10s antes do próximo arquivo...")
            time.sleep(10)

        return {"status": "ok", "chunks": len(chunks)}

    except Exception as e:
        return {"status": "erro", "erro": str(e)}