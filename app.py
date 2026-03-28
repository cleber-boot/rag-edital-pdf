"""
app.py — Interface Streamlit para RAG com PDFs
Versão Codespaces: ChromaDB local, sem HF Storage
Embeddings: Gemini gemini-embedding-001
LLM: Gemini gemini-2.5-flash-lite
"""

import os
import re
import time
import streamlit as st
import chromadb
from google import genai
from flashrank import Ranker
from dotenv import load_dotenv

from indexar import indexar_pdf_bytes
from perguntar import buscar_trechos, rerankar_trechos, montar_contexto, SYSTEM_PROMPT

load_dotenv()

# ── configuração ──────────────────────────────────────────────
CHROMA_BASE_DIR = "./chroma_bancos"
MODELO_ID       = "gemini-2.5-flash-lite"

# ── inicialização (com cache do Streamlit) ────────────────────
@st.cache_resource
def inicializar():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY não encontrada. Crie um arquivo .env com GEMINI_API_KEY=sua_chave")
        st.stop()
    cliente = genai.Client(api_key=api_key)
    ranker  = Ranker()
    return cliente, ranker

cliente_genai, ranker = inicializar()


# ── helpers de banco ──────────────────────────────────────────
def _slug(texto: str) -> str:
    texto = texto.strip().lower()
    texto = re.sub(r"[^\w\s-]", "", texto)
    texto = re.sub(r"[\s_-]+", "_", texto)
    return texto[:50] or "banco"


@st.cache_resource
def _get_chroma_cliente(nome_banco: str):
    path = os.path.join(CHROMA_BASE_DIR, nome_banco)
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_colecao(nome_banco: str):
    cliente = _get_chroma_cliente(nome_banco)
    return cliente.get_or_create_collection(
        name="pdfs",
        metadata={"hnsw:space": "cosine"}
    )


def listar_bancos() -> list[str]:
    if not os.path.exists(CHROMA_BASE_DIR):
        return []
    return sorted([
        d for d in os.listdir(CHROMA_BASE_DIR)
        if os.path.isdir(os.path.join(CHROMA_BASE_DIR, d))
    ])


def listar_pdfs(nome_banco: str) -> list[str]:
    try:
        col = get_colecao(nome_banco)
        if col.count() == 0:
            return []
        metas = col.get()["metadatas"]
        return sorted(set(m.get("arquivo", "") for m in metas if m.get("arquivo")))
    except Exception:
        return []


# ── geração de resposta com retry ─────────────────────────────
def gerar_resposta(prompt: str, max_tentativas: int = 5) -> str:
    for tentativa in range(max_tentativas):
        try:
            resposta = cliente_genai.models.generate_content(
                model=MODELO_ID,
                contents=prompt
            )
            return resposta.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                espera = 15 * (tentativa + 1)
                st.warning(f"⏳ Limite de API atingido. Aguardando {espera}s... (tentativa {tentativa+1}/{max_tentativas})")
                time.sleep(espera)
            else:
                raise
    raise RuntimeError("Limite de tentativas atingido.")


# ── interface Streamlit ───────────────────────────────────────
st.set_page_config(page_title="📚 RAG com PDFs", layout="wide")
st.title("📚 RAG com PDFs")
st.caption("powered by Gemini 2.5 Flash-Lite + Flashrank + ChromaDB")

aba = st.tabs(["📥 Indexar", "🔍 Perguntar", "📄 Resumir", "📊 Status"])


# ══════════════════════════════════════════════════════════════
# ABA: INDEXAR
# ══════════════════════════════════════════════════════════════
with aba[0]:
    st.header("Indexar PDFs")
    st.info("Digite um nome para o banco e faça upload dos PDFs. Cada banco é um conjunto independente de documentos.")

    nome_banco_input = st.text_input(
        "Nome do banco",
        placeholder="Ex: direito_civil, medicina_2024, tcc_joao",
        help="O nome será convertido para letras minúsculas e sem espaços."
    )
    arquivos = st.file_uploader("Selecione PDFs", type="pdf", accept_multiple_files=True)

    if st.button("📥 Indexar PDFs", type="primary"):
        if not nome_banco_input.strip():
            st.error("⚠️ Digite um nome para o banco.")
        elif not arquivos:
            st.error("⚠️ Selecione ao menos um PDF.")
        else:
            nome_banco = _slug(nome_banco_input)
            colecao = get_colecao(nome_banco)
            st.write(f"📂 Banco: **{nome_banco}**")

            for arquivo in arquivos:
                conteudo = arquivo.read()
                with st.spinner(f"Indexando {arquivo.name}..."):
                    resultado = indexar_pdf_bytes(arquivo.name, conteudo, colecao, cliente_genai)

                if resultado["status"] == "ok":
                    st.success(f"✅ {arquivo.name} — {resultado['chunks']} chunks indexados")
                elif resultado["status"] == "ja_indexado":
                    st.warning(f"↩️ {arquivo.name} — já indexado, pulando")
                else:
                    st.error(f"❌ {arquivo.name} — {resultado['erro']}")

            st.success(f"🗄️ Total de chunks no banco: {colecao.count()}")
            st.cache_resource.clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════
# ABA: PERGUNTAR
# ══════════════════════════════════════════════════════════════
with aba[1]:
    st.header("Perguntar")

    bancos = listar_bancos()
    if not bancos:
        st.warning("Nenhum banco encontrado. Indexe PDFs primeiro.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            banco_sel = st.selectbox("Banco ativo", bancos, key="banco_perguntar")
        with col2:
            top_k = st.slider("Candidatos (top_k)", 5, 20, 10)
            top_n = st.slider("Após rerank (top_n)", 2, 8, 5)

        pergunta = st.text_area("Sua pergunta", placeholder="Ex: Quais são os principais conceitos abordados?", height=80)

        if st.button("🔍 Perguntar", type="primary"):
            if not pergunta.strip():
                st.error("⚠️ Digite uma pergunta.")
            else:
                colecao = get_colecao(banco_sel)
                if colecao.count() == 0:
                    st.error("⚠️ Banco vazio. Indexe PDFs primeiro.")
                else:
                    with st.spinner("Buscando trechos relevantes..."):
                        try:
                            trechos_raw = buscar_trechos(pergunta, colecao, cliente_genai, top_k=top_k)
                            trechos     = rerankar_trechos(pergunta, trechos_raw, ranker, top_n=top_n)
                        except Exception as e:
                            st.error(f"❌ Erro na busca: {e}")
                            st.stop()

                    with st.spinner("Gerando resposta..."):
                        try:
                            contexto = montar_contexto(trechos)
                            prompt   = f"{SYSTEM_PROMPT}\n\nTRECHOS RECUPERADOS:\n{contexto}\n\nPERGUNTA: {pergunta}"
                            resposta = gerar_resposta(prompt)
                        except Exception as e:
                            st.error(f"❌ Erro ao gerar resposta: {e}")
                            st.stop()

                    st.subheader("Resposta")
                    st.write(resposta)

                    with st.expander("📎 Trechos utilizados"):
                        for i, t in enumerate(trechos, 1):
                            st.markdown(f"**#{i} · {t['arquivo']}** | rerank: `{t['relevancia_rerank']}` | embed: `{t['relevancia']}`")
                            st.markdown(f"> {t['texto']}")
                            st.divider()


# ══════════════════════════════════════════════════════════════
# ABA: RESUMIR
# ══════════════════════════════════════════════════════════════
with aba[2]:
    st.header("Resumir PDF Completo")
    st.info("Gera um resumo de **todo** o conteúdo de um PDF indexado, não apenas dos trechos mais similares.")

    bancos = listar_bancos()
    if not bancos:
        st.warning("Nenhum banco encontrado. Indexe PDFs primeiro.")
    else:
        banco_sel_r = st.selectbox("Banco", bancos, key="banco_resumir")
        pdfs        = listar_pdfs(banco_sel_r)

        if not pdfs:
            st.warning("Nenhum PDF encontrado neste banco.")
        else:
            pdf_sel = st.selectbox("PDF para resumir", pdfs)

            if st.button("📄 Gerar Resumo Completo", type="primary"):
                colecao = get_colecao(banco_sel_r)

                with st.spinner("Buscando todos os chunks do PDF..."):
                    resultado = colecao.get(where={"arquivo": pdf_sel})
                    chunks    = resultado["documents"]

                if not chunks:
                    st.error("Nenhum chunk encontrado para este PDF.")
                else:
                    st.write(f"📄 {len(chunks)} chunks encontrados. Gerando resumo...")
                    LOTE = 50
                    resumos_parciais = []
                    total_lotes = (len(chunks) + LOTE - 1) // LOTE

                    barra = st.progress(0, text="Resumindo...")
                    for i in range(0, len(chunks), LOTE):
                        lote       = chunks[i:i + LOTE]
                        lote_num   = i // LOTE + 1
                        texto_lote = "\n\n".join(lote)
                        prompt     = f"""Você é um assistente especializado em resumos.
Abaixo estão trechos do documento '{pdf_sel}'.
Faça um resumo claro, organizado e completo desses trechos em português.
Destaque os pontos principais, conceitos-chave e conclusões importantes.

TRECHOS:
{texto_lote}

RESUMO:"""
                        try:
                            resumo = gerar_resposta(prompt)
                            resumos_parciais.append(resumo)
                        except Exception as e:
                            resumos_parciais.append(f"[Erro no lote {lote_num}: {e}]")

                        barra.progress(lote_num / total_lotes, text=f"Lote {lote_num}/{total_lotes}...")

                    barra.empty()

                    if len(resumos_parciais) == 1:
                        resumo_final = resumos_parciais[0]
                    else:
                        with st.spinner("Consolidando resumos parciais..."):
                            consolidado  = "\n\n---\n\n".join(
                                [f"Parte {i+1}:\n{r}" for i, r in enumerate(resumos_parciais)]
                            )
                            prompt_final = f"""Consolide os resumos parciais abaixo em um único resumo final,
coeso e bem estruturado em português, sobre o documento '{pdf_sel}'.

{consolidado}

RESUMO FINAL:"""
                            try:
                                resumo_final = gerar_resposta(prompt_final)
                            except Exception as e:
                                resumo_final = f"(Erro na consolidação: {e})\n\n" + consolidado

                    st.subheader("Resumo")
                    st.write(resumo_final)


# ══════════════════════════════════════════════════════════════
# ABA: STATUS
# ══════════════════════════════════════════════════════════════
with aba[3]:
    st.header("Status dos Bancos")

    bancos = listar_bancos()
    if not bancos:
        st.warning("Nenhum banco encontrado.")
    else:
        for banco in bancos:
            try:
                col     = get_colecao(banco)
                pdfs    = listar_pdfs(banco)
                chunks  = col.count()
                with st.expander(f"🗄️ **{banco}** — {chunks} chunks | {len(pdfs)} PDF(s)"):
                    for pdf in pdfs:
                        st.write(f"• {pdf}")
            except Exception as e:
                st.error(f"❌ {banco}: {e}")

    if st.button("🔄 Atualizar"):
        st.rerun()
