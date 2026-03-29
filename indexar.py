"""
indexar.py — processa PDFs e salva no ChromaDB
Embeddings multimodais via gemini-embedding-2-preview
Suporte a: texto nativo, PDFs escaneados, figuras e gráficos
"""

import io
import time
import tempfile

import fitz  # PyMuPDF
from PIL import Image
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── configuração ──────────────────────────────────────────────
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
EMBEDDING_MODEL = "gemini-embedding-2-preview"

# Máximo de imagens por requisição de embedding (limite da API)
MAX_IMGS_POR_LOTE = 6

# Limite de tokens por minuto (free tier) — pausa entre lotes
BATCH_SIZE  = 20   # chunks de texto por lote
BATCH_PAUSE = 10   # segundos entre lotes
MAX_RETRY   = 5
RETRY_WAIT  = 60

# Mínimo de chars para considerar que a página tem texto nativo
MIN_CHARS_TEXTO = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)


# ── helpers de imagem ─────────────────────────────────────────
def _pagina_para_pil(pagina: fitz.Page, dpi: int = 150) -> Image.Image:
    """Renderiza uma página PDF como imagem PIL."""
    pix = pagina.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _extrair_imagens_pagina(pagina: fitz.Page) -> list[Image.Image]:
    """Extrai imagens embutidas de uma página PDF como PIL Images."""
    imagens = []
    for img_info in pagina.get_images(full=True):
        xref = img_info[0]
        try:
            base_img  = pagina.parent.extract_image(xref)
            img       = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
            # Ignora imagens muito pequenas (ícones, decorações)
            if img.width > 100 and img.height > 100:
                imagens.append(img)
        except Exception:
            continue
    return imagens


# ── extração de conteúdo por página ──────────────────────────
def extrair_conteudo_pdf(conteudo: bytes) -> dict:
    """
    Extrai conteúdo de um PDF separando:
    - chunks de texto (para embedding de texto)
    - imagens de páginas (para embedding de imagem)

    Retorna:
        {
            "chunks_texto": list[str],
            "imagens_pagina": list[{"pagina": int, "imagem": PIL.Image}],
            "imagens_figura": list[{"pagina": int, "figura": int, "imagem": PIL.Image}]
        }
    """
    chunks_texto    = []
    imagens_pagina  = []
    imagens_figura  = []
    texto_acumulado = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(conteudo)
        tmp.flush()
        doc = fitz.open(tmp.name)

        for num_pag, pagina in enumerate(doc, 1):
            texto_nativo = pagina.get_text().strip()

            if len(texto_nativo) >= MIN_CHARS_TEXTO:
                # Página com texto nativo — acumula para chunking
                texto_acumulado.append(f"[Página {num_pag}]\n{texto_nativo}")

                # Extrai figuras embutidas para embedding visual
                figuras = _extrair_imagens_pagina(pagina)
                for i, img in enumerate(figuras, 1):
                    imagens_figura.append({
                        "pagina": num_pag,
                        "figura": i,
                        "imagem": img
                    })
            else:
                # Página escaneada — indexa como imagem
                img_pagina = _pagina_para_pil(pagina)
                imagens_pagina.append({
                    "pagina": num_pag,
                    "imagem": img_pagina
                })

    # Gera chunks do texto acumulado
    if texto_acumulado:
        texto_completo = "\n\n".join(texto_acumulado)
        chunks_texto   = splitter.split_text(texto_completo)

    return {
        "chunks_texto":   chunks_texto,
        "imagens_pagina": imagens_pagina,
        "imagens_figura": imagens_figura,
    }


# ── embeddings com retry ──────────────────────────────────────
def _embed_com_retry(cliente: genai.Client, contents, task_type: str) -> list[float]:
    """Gera embedding de um conteúdo (texto ou imagem) com retry."""
    for tentativa in range(1, MAX_RETRY + 1):
        try:
            resultado = cliente.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return resultado.embeddings[0].values
        except Exception as e:
            erro = str(e)
            if "429" in erro or "RESOURCE_EXHAUSTED" in erro:
                print(f"[429] Aguardando {RETRY_WAIT}s... (tentativa {tentativa}/{MAX_RETRY})")
                time.sleep(RETRY_WAIT)
            else:
                raise
    raise RuntimeError(f"Falha ao gerar embedding após {MAX_RETRY} tentativas.")


def _embed_texto_lote(
    cliente: genai.Client,
    chunks: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """Gera embeddings de uma lista de chunks de texto em lotes."""
    todos = []
    total = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        lote     = chunks[i:i + BATCH_SIZE]
        lote_num = i // BATCH_SIZE + 1
        print(f"[embedding texto] Lote {lote_num}/{total} — {len(lote)} chunks...")

        for chunk in lote:
            emb = _embed_com_retry(cliente, chunk, task_type)
            todos.append(emb)

        if i + BATCH_SIZE < len(chunks):
            time.sleep(BATCH_PAUSE)

    return todos


def _embed_imagem(cliente: genai.Client, imagem: Image.Image) -> list[float]:
    """Gera embedding de uma imagem PIL."""
    return _embed_com_retry(cliente, imagem, "RETRIEVAL_DOCUMENT")


# ── indexação principal ───────────────────────────────────────
def indexar_pdf_bytes(
    nome: str,
    conteudo: bytes,
    colecao,
    cliente: genai.Client,
    callback=None,
    pausar_ao_final: bool = False,
) -> dict:
    """
    Indexa um PDF com embeddings multimodais (gemini-embedding-2-preview).

    - Texto nativo → embedding de texto
    - Páginas escaneadas → embedding de imagem da página inteira
    - Figuras/gráficos embutidos → embedding de imagem da figura

    Retorna dict com:
        status : "ok" | "ja_indexado" | "erro"
        chunks : int  (se ok)
        erro   : str  (se erro)
    """
    # Verifica se já foi indexado
    if colecao.count() > 0:
        ja_indexados = {m.get("arquivo") for m in colecao.get()["metadatas"]}
        if nome in ja_indexados:
            return {"status": "ja_indexado", "chunks": 0}

    try:
        print(f"[indexar] Extraindo conteúdo de '{nome}'...")
        conteudo_pdf = extrair_conteudo_pdf(conteudo)

        chunks_texto   = conteudo_pdf["chunks_texto"]
        imgs_pagina    = conteudo_pdf["imagens_pagina"]
        imgs_figura    = conteudo_pdf["imagens_figura"]

        total_items = len(chunks_texto) + len(imgs_pagina) + len(imgs_figura)
        if total_items == 0:
            return {"status": "erro", "erro": "PDF vazio — nenhum conteúdo extraído"}

        print(f"[indexar] '{nome}' → {len(chunks_texto)} chunks texto, "
              f"{len(imgs_pagina)} páginas escaneadas, "
              f"{len(imgs_figura)} figuras")

        ids        = []
        embeddings = []
        documents  = []
        metadatas  = []
        contador   = 0
        total_lotes = (len(chunks_texto) + BATCH_SIZE - 1) // BATCH_SIZE or 1

        # ── 1. Embeddings de texto ────────────────────────────
        for i, chunk in enumerate(chunks_texto):
            emb = _embed_com_retry(cliente, chunk, "RETRIEVAL_DOCUMENT")
            ids.append(f"{nome}_texto_{i}")
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append({"arquivo": nome, "tipo": "texto", "chunk": i})
            contador += 1

            if callback and i % BATCH_SIZE == 0:
                lote_num = i // BATCH_SIZE + 1
                callback(lote_num, total_lotes)

            if (i + 1) % BATCH_SIZE == 0 and i + 1 < len(chunks_texto):
                time.sleep(BATCH_PAUSE)

        # ── 2. Embeddings de páginas escaneadas ───────────────
        for item in imgs_pagina:
            num_pag = item["pagina"]
            img     = item["imagem"]
            print(f"[indexar] Embedding página escaneada {num_pag}...")

            emb = _embed_imagem(cliente, img)
            ids.append(f"{nome}_pagina_{num_pag}")
            embeddings.append(emb)
            documents.append(f"[Página escaneada {num_pag} de {nome}]")
            metadatas.append({
                "arquivo": nome,
                "tipo":    "pagina_escaneada",
                "pagina":  num_pag
            })
            time.sleep(2)  # pausa entre imagens

        # ── 3. Embeddings de figuras embutidas ────────────────
        for item in imgs_figura:
            num_pag = item["pagina"]
            num_fig = item["figura"]
            img     = item["imagem"]
            print(f"[indexar] Embedding figura {num_fig} da página {num_pag}...")

            emb = _embed_imagem(cliente, img)
            ids.append(f"{nome}_figura_{num_pag}_{num_fig}")
            embeddings.append(emb)
            documents.append(f"[Figura {num_fig} da página {num_pag} de {nome}]")
            metadatas.append({
                "arquivo": nome,
                "tipo":    "figura",
                "pagina":  num_pag,
                "figura":  num_fig
            })
            time.sleep(2)  # pausa entre imagens

        if pausar_ao_final:
            print("[indexar] Pausa de 10s antes do próximo arquivo...")
            time.sleep(10)

        # ── Salva no ChromaDB ─────────────────────────────────
        colecao.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return {"status": "ok", "chunks": len(ids)}

    except Exception as e:
        return {"status": "erro", "erro": str(e)}