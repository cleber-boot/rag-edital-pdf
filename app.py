"""
app.py — Interface Streamlit para RAG com PDFs
Versão Codespaces: ChromaDB local, download de PDF

Embeddings : Gemini gemini-embedding-001        (100 req/min — plano gratuito)
LLM        : Groq  moonshotai/kimi-k2-instruct  (60 RPM, 300K TPD, 262K contexto)

Variáveis de ambiente necessárias no .env:
    GEMINI_API_KEY=...
    GROQ_API_KEY=...
"""

import os
import re
import time
import random
import logging
import streamlit as st
import chromadb
from google import genai
from groq import Groq
from flashrank import Ranker
from dotenv import load_dotenv

from indexar import indexar_pdf_bytes
from perguntar import buscar_trechos, rerankar_trechos, montar_contexto, SYSTEM_PROMPT
from gerar_pdf import gerar_pdf_resumo, gerar_pdf_resposta

load_dotenv()

logger = logging.getLogger(__name__)

# ── configuração ──────────────────────────────────────────────
CHROMA_BASE_DIR = "./chroma_bancos"
MODELO_GROQ     = "llama-3.1-8b-instant"  # 60 RPM / 300K TPD / 262K ctx

_ERROS_429 = ("429", "rate_limit_exceeded", "rate limit", "too many requests")
_ERROS_413 = ("413", "request too large", "request_too_large")


# ── inicialização ─────────────────────────────────────────────
@st.cache_resource
def inicializar():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key   = os.environ.get("GROQ_API_KEY")

    erros = []
    if not gemini_key:
        erros.append("GEMINI_API_KEY")
    if not groq_key:
        erros.append("GROQ_API_KEY")
    if erros:
        st.error(f"❌ Chaves não encontradas no .env: {', '.join(erros)}")
        st.stop()

    cliente_gemini = genai.Client(api_key=gemini_key)
    cliente_groq   = Groq(api_key=groq_key)
    ranker         = Ranker()
    return cliente_gemini, cliente_groq, ranker

cliente_gemini, cliente_groq, ranker = inicializar()


# ── helpers de banco ──────────────────────────────────────────
def _slug(texto: str) -> str:
    texto = texto.strip().lower()
    texto = re.sub(r"[^\w\s-]", "", texto)
    texto = re.sub(r"[\s\_-]+", "_", texto)
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


# ── geração de resposta via Groq/Kimi K2 com backoff exponencial ─────
def gerar_resposta(prompt: str, max_tentativas: int = 6) -> str:
    """
    Chama o Kimi K2 via Groq com retry automático (backoff exponencial + jitter).
    Lê o header 'retry-after' quando disponível para respeitar exatamente
    o tempo sugerido pela API.
    """
    espera_base = 5.0
    espera_max  = 90.0

    for tentativa in range(1, max_tentativas + 1):
        try:
            resposta = cliente_groq.chat.completions.create(
                model=MODELO_GROQ,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            return resposta.choices[0].message.content

        except Exception as e:
            erro_str = str(e).lower()
            eh_413   = any(t in erro_str for t in _ERROS_413)
            eh_429   = any(t in erro_str for t in _ERROS_429)

            if eh_413:
                raise RuntimeError(
                    f"❌ Prompt muito grande para o modelo ({len(prompt.split())} palavras aprox). "
                    "Reduza o tamanho do contexto ou use lotes menores."
                ) from e

            if not eh_429:
                raise

            if tentativa == max_tentativas:
                raise RuntimeError(
                    f"Limite de {max_tentativas} tentativas no Groq. Último erro: {e}"
                )

            ESPERA_MAX_RETRY_AFTER = 60.0
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after = e.response.headers.get("retry-after")

            if retry_after:
                espera = min(float(retry_after), ESPERA_MAX_RETRY_AFTER)
                espera += random.uniform(0.5, 2.0)
            else:
                espera = min(espera_base * (2 ** (tentativa - 1)), espera_max)
                espera += random.uniform(-2.0, 2.0)
                espera = max(espera, 2.0)

            logger.warning(
                "[gerar_resposta] 429 Groq/Kimi (tentativa %d/%d). Aguardando %.1fs...",
                tentativa, max_tentativas, espera,
            )
            st.warning(
                f"⏳ Limite da API Groq atingido. Aguardando {espera:.0f}s... "
                f"(tentativa {tentativa}/{max_tentativas})"
            )
            time.sleep(espera)


# ── interface ─────────────────────────────────────────────────
st.set_page_config(page_title="📚 RAG com PDFs", layout="wide")
st.title("📚 RAG com PDFs")
st.caption("powered by Kimi K2 (Groq) + Gemini Embeddings + Flashrank + ChromaDB")

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
            colecao    = get_colecao(nome_banco)
            st.write(f"📂 Banco: **{nome_banco}**")

            total_arquivos = len(arquivos)
            for idx, arquivo in enumerate(arquivos, 1):
                conteudo = arquivo.read()
                st.write(f"📄 **[{idx}/{total_arquivos}]** {arquivo.name}")

                barra     = st.progress(0, text="Aguardando embeddings...")
                status_tx = st.empty()

                def atualizar_progresso(lote_num, total_lotes, _b=barra, _s=status_tx):
                    pct = lote_num / total_lotes
                    _b.progress(pct, text=f"Lote {lote_num}/{total_lotes} de embeddings...")
                    _s.caption(f"⏳ Processando... {int(pct*100)}%")

                eh_ultimo = (idx == total_arquivos)
                resultado = indexar_pdf_bytes(
                    arquivo.name, conteudo, colecao, cliente_gemini,
                    callback=atualizar_progresso,
                    pausar_ao_final=not eh_ultimo,
                )

                barra.empty()
                status_tx.empty()

                if resultado["status"] == "ok":
                    st.success(f"✅ {arquivo.name} — {resultado['chunks']} chunks indexados")
                    if not eh_ultimo:
                        st.caption("⏸️ Aguardando 10s antes do próximo arquivo...")
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

        pergunta = st.text_area(
            "Sua pergunta",
            placeholder="Ex: Quais são os principais conceitos abordados?",
            height=80
        )

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
                            trechos_raw = buscar_trechos(pergunta, colecao, cliente_gemini, top_k=top_k)
                            trechos     = rerankar_trechos(pergunta, trechos_raw, ranker, top_n=top_n)
                        except Exception as e:
                            st.error(f"❌ Erro na busca: {e}")
                            st.stop()

                    with st.spinner("Gerando resposta com Kimi K2..."):
                        try:
                            contexto = montar_contexto(trechos)
                            prompt   = f"TRECHOS RECUPERADOS:\n{contexto}\n\nPERGUNTA: {pergunta}"
                            resposta = gerar_resposta(prompt)
                        except Exception as e:
                            st.error(f"❌ Erro ao gerar resposta: {e}")
                            st.stop()

                    st.session_state["ultima_pergunta"]  = pergunta
                    st.session_state["ultima_resposta"]  = resposta
                    st.session_state["ultimos_trechos"]  = trechos
                    st.session_state["ultimo_banco_p"]   = banco_sel

        if "ultima_resposta" in st.session_state:
            st.subheader("Resposta")
            st.write(st.session_state["ultima_resposta"])

            with st.expander("📎 Trechos utilizados"):
                for i, t in enumerate(st.session_state["ultimos_trechos"], 1):
                    st.markdown(
                        f"**#{i} · {t['arquivo']}** | "
                        f"rerank: `{t['relevancia_rerank']}` | embed: `{t['relevancia']}`"
                    )
                    st.markdown(f"> {t['texto']}")
                    st.divider()

            pdf_bytes = gerar_pdf_resposta(
                pergunta = st.session_state["ultima_pergunta"],
                resposta = st.session_state["ultima_resposta"],
                banco    = st.session_state["ultimo_banco_p"],
                trechos  = st.session_state["ultimos_trechos"]
            )
            st.download_button(
                label     = "📥 Baixar resposta em PDF",
                data      = pdf_bytes,
                file_name = "resposta.pdf",
                mime      = "application/pdf"
            )


# ══════════════════════════════════════════════════════════════
# ABA: RESUMIR
# ══════════════════════════════════════════════════════════════
with aba[2]:
    st.header("Resumir PDF Completo")
    st.info("Gera um resumo de **todo** o conteúdo de um PDF indexado, explicando cada parte encontrada.")

    ESTILOS_RESUMO = {
        "📖 Didático": {
            "descricao": "Explica cada parte do texto em linguagem simples, com exemplos do cotidiano.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um professor explicando um documento para um aluno iniciante.
Abaixo estao trechos do documento '{pdf}'.
Para CADA parte do texto encontrada, explique o que ela significa em linguagem simples e direta.
Use exemplos do cotidiano quando possivel.
Organize a explicacao em topicos, um para cada parte do texto.

TRECHOS:
{texto}

EXPLICACAO DIDATICA DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Voce recebeu explicacoes didaticas de varias partes do documento '{pdf}'.
Consolide tudo em um unico texto explicativo, organizado e coeso, mantendo a linguagem simples.
Preserve as explicacoes de cada parte mas elimine repeticoes.

PARTES EXPLICADAS:
{consolidado}

EXPLICACAO FINAL CONSOLIDADA:"""
        },
        "🔬 Técnico": {
            "descricao": "Analisa cada parte com linguagem técnica, mantendo termos e referências do documento.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um especialista tecnico analisando o documento '{pdf}'.
Para CADA parte do texto encontrada, faca uma analise tecnica detalhada.
Mantenha os termos tecnicos, referencias normativas e dados quantitativos encontrados.
Organize em topicos, um por parte do texto analisada.

TRECHOS:
{texto}

ANALISE TECNICA DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide as analises tecnicas das partes do documento '{pdf}'.
Mantenha o rigor tecnico, os termos especializados e as referencias encontradas.
Elimine repeticoes mas preserve todos os dados tecnicos relevantes.

ANALISES:
{consolidado}

ANALISE TECNICA FINAL CONSOLIDADA:"""
        },
        "⚡ Resumido": {
            "descricao": "Resume cada parte em 2-3 frases objetivas, destacando apenas o essencial.",
            "prompt_lote": lambda pdf, texto: f"""Resuma de forma extremamente objetiva cada parte do documento '{pdf}'.
Para CADA parte do texto encontrada, escreva no maximo 2 a 3 frases destacando apenas o ponto principal.
Seja direto e elimine qualquer informacao secundaria.

TRECHOS:
{texto}

RESUMO OBJETIVO DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide os resumos objetivos das partes do documento '{pdf}'.
Mantenha apenas os pontos mais importantes de cada parte.
O resultado deve ser um resumo executivo curto e direto.

RESUMOS:
{consolidado}

RESUMO EXECUTIVO FINAL:"""
        },
        "🧠 Analítico": {
            "descricao": "Analisa criticamente cada parte, identificando argumentos, evidências e conclusões.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um analista critico examinando o documento '{pdf}'.
Para CADA parte do texto encontrada, identifique e explique:
- Qual e o argumento ou ideia central dessa parte
- Quais evidencias ou dados sao apresentados
- Qual e a conclusao ou implicacao dessa parte
Organize em topicos, um por parte analisada.

TRECHOS:
{texto}

ANALISE CRITICA DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide a analise critica das partes do documento '{pdf}'.
Identifique os padroes, contradicoes e conclusoes gerais que emergem da analise de cada parte.
Apresente uma visao critica integrada do documento.

ANALISES:
{consolidado}

ANALISE CRITICA FINAL INTEGRADA:"""
        },
        "📋 Comparativo": {
            "descricao": "Compara cada parte com conceitos similares, destacando diferenças e inovações.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um especialista comparando o conteudo do documento '{pdf}' com conhecimentos estabelecidos na area.
Para CADA parte do texto encontrada, compare com versoes anteriores, conceitos similares ou praticas comuns.
Destaque o que e novo, diferente ou inovador em cada parte.
Organize em topicos, um por parte comparada.

TRECHOS:
{texto}

COMPARACAO DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide as comparacoes das partes do documento '{pdf}'.
Apresente uma visao geral das diferencas e inovacoes encontradas ao longo de todo o documento.

COMPARACOES:
{consolidado}

COMPARACAO FINAL CONSOLIDADA:"""
        },
    }

    bancos = listar_bancos()
    if not bancos:
        st.warning("Nenhum banco encontrado. Indexe PDFs primeiro.")
    else:
        col_r1, col_r2 = st.columns([2, 2])
        with col_r1:
            banco_sel_r = st.selectbox("Banco", bancos, key="banco_resumir")
            pdfs = listar_pdfs(banco_sel_r)
            if pdfs:
                pdf_sel = st.selectbox("PDF para resumir", pdfs)
            else:
                st.warning("Nenhum PDF neste banco.")
                pdf_sel = None
        with col_r2:
            estilo_sel = st.selectbox(
                "Tipo de resumo",
                list(ESTILOS_RESUMO.keys()),
                key="estilo_resumo"
            )
            if estilo_sel:
                st.caption(ESTILOS_RESUMO[estilo_sel]["descricao"])

        if pdf_sel and st.button("📄 Gerar Resumo", type="primary"):
            estilo  = ESTILOS_RESUMO[estilo_sel]
            colecao = get_colecao(banco_sel_r)

            with st.spinner("Buscando todos os chunks do PDF..."):
                resultado = colecao.get(where={"arquivo": pdf_sel})
                chunks    = resultado["documents"]

            if not chunks:
                st.error("Nenhum chunk encontrado para este PDF.")
            else:
                st.write(f"📄 {len(chunks)} partes encontradas. Aplicando estilo **{estilo_sel}**...")

                # ── CORREÇÃO: lote maior = menos chamadas = menos 429 ──
                LOTE             = 20   # era 5 — reduz chamadas em 4x
                PAUSA_ENTRE_LOTES = 5   # segundos entre lotes para respeitar RPM
                resumos_parciais = []
                total_lotes      = (len(chunks) + LOTE - 1) // LOTE
                barra            = st.progress(0, text="Analisando partes...")

                for i in range(0, len(chunks), LOTE):
                    lote       = chunks[i : i + LOTE]
                    lote_num   = i // LOTE + 1
                    texto_lote = "\n\n---PARTE---\n\n".join(lote)
                    prompt     = estilo["prompt_lote"](pdf_sel, texto_lote)

                    try:
                        resumo = gerar_resposta(prompt)
                        resumos_parciais.append(resumo)
                    except Exception as e:
                        resumos_parciais.append(f"[Erro no lote {lote_num}: {e}]")

                    barra.progress(lote_num / total_lotes, text=f"Parte {lote_num}/{total_lotes}...")

                    # Pausa entre lotes para não estourar RPM
                    if lote_num < total_lotes:
                        time.sleep(PAUSA_ENTRE_LOTES)

                barra.empty()

                if len(resumos_parciais) == 1:
                    resumo_final = resumos_parciais[0]
                else:
                    with st.spinner("Consolidando análise de todas as partes..."):
                        GRUPO = 3
                        grupos = [resumos_parciais[g:g+GRUPO] for g in range(0, len(resumos_parciais), GRUPO)]
                        intermediarios = []
                        for gi, grupo in enumerate(grupos):
                            sub = "\n\n===\n\n".join([f"Secao {i+1}:\n{r}" for i, r in enumerate(grupo)])
                            prompt_sub = estilo["prompt_final"](pdf_sel, sub)
                            try:
                                intermediarios.append(gerar_resposta(prompt_sub))
                                if gi < len(grupos) - 1:
                                    time.sleep(PAUSA_ENTRE_LOTES)
                            except Exception as e:
                                intermediarios.append(f"[Erro grupo {gi+1}: {e}]")

                        if len(intermediarios) == 1:
                            resumo_final = intermediarios[0]
                        else:
                            consolidado_final = "\n\n===\n\n".join(
                                [f"Parte {i+1}:\n{r}" for i, r in enumerate(intermediarios)]
                            )
                            prompt_final = estilo["prompt_final"](pdf_sel, consolidado_final)
                            try:
                                resumo_final = gerar_resposta(prompt_final)
                            except Exception as e:
                                resumo_final = f"(Erro final: {e})\n\n" + consolidado_final

                st.session_state["ultimo_resumo"]     = resumo_final
                st.session_state["ultimo_pdf_resumo"] = pdf_sel
                st.session_state["ultimo_banco_r"]    = banco_sel_r
                st.session_state["ultimo_estilo_r"]   = estilo_sel

        if "ultimo_resumo" in st.session_state:
            estilo_label = st.session_state.get("ultimo_estilo_r", "")
            st.subheader(f"Resultado — {estilo_label}")
            st.write(st.session_state["ultimo_resumo"])

            pdf_bytes = gerar_pdf_resumo(
                nome_pdf = st.session_state["ultimo_pdf_resumo"],
                banco    = st.session_state["ultimo_banco_r"],
                resumo   = st.session_state["ultimo_resumo"],
                estilo   = estilo_label
            )
            nome_arquivo = f"resumo_{st.session_state['ultimo_pdf_resumo']}"
            st.download_button(
                label     = "📥 Baixar resumo em PDF",
                data      = pdf_bytes,
                file_name = nome_arquivo,
                mime      = "application/pdf"
            )


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
                col    = get_colecao(banco)
                pdfs   = listar_pdfs(banco)
                chunks = col.count()
                with st.expander(f"🗄️ **{banco}** — {chunks} chunks | {len(pdfs)} PDF(s)"):
                    for pdf in pdfs:
                        st.write(f"• {pdf}")
            except Exception as e:
                st.error(f"❌ {banco}: {e}")

    if st.button("🔄 Atualizar"):
        st.rerun()