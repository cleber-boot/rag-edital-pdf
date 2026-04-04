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

# Suprime logs verbosos do httpx e groq (429 retry noise)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── configuração ──────────────────────────────────────────────
CHROMA_BASE_DIR = "./chroma_bancos"
MODELO_GROQ     = "moonshotai/kimi-k2-instruct"

# Fallback automático quando o modelo principal esgota o TPD
MODELOS_FALLBACK = [
    "moonshotai/kimi-k2-instruct",   # 1º: kimi — 300k TPD, 10k TPM
    "qwen/qwen3-32b",                # 2º: qwen — 500k TPD, 6k TPM
    "llama-3.3-70b-versatile",       # 3º: 70b  — 100k TPD, 12k TPM
]
_modelo_atual_idx = [0]  # índice mutável do modelo em uso — resetado a cada resumo


def _resetar_modelo():
    """Reseta para o modelo principal no início de cada operação."""
    _modelo_atual_idx[0] = 0


def _modelo_atual() -> str:
    return MODELOS_FALLBACK[_modelo_atual_idx[0]]


def _avancar_modelo() -> str | None:
    """Avança para o próximo modelo da lista. Retorna None se esgotou todos."""
    if _modelo_atual_idx[0] < len(MODELOS_FALLBACK) - 1:
        _modelo_atual_idx[0] += 1
        return _modelo_atual()
    return None

_ERROS_429 = ("429", "rate_limit_exceeded", "rate limit", "too many requests")
_ERROS_413 = ("413", "request too large", "request_too_large", "context_length_exceeded", "context length", "maximum context")


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


# ── geração de resposta com fallback automático de modelo ─────────────
def gerar_resposta(prompt: str, max_tentativas: int = 6) -> str:
    """
    Chama o modelo atual via Groq com retry e backoff exponencial.
    Se o TPD do modelo estiver esgotado, avança automaticamente para o próximo.
    Ordem: kimi-k2 → qwen3-32b → llama-3.3-70b
    """
    espera_base = 10.0
    espera_max  = 60.0

    for tentativa in range(1, max_tentativas + 1):
        try:
            resposta = cliente_groq.chat.completions.create(
                model=_modelo_atual(),
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
            logger.warning("[gerar_resposta] Erro raw: %s", str(e))
            eh_413   = any(t in erro_str for t in _ERROS_413)
            eh_429   = any(t in erro_str for t in _ERROS_429)
            eh_tpd   = "tokens per day" in erro_str or "per day" in erro_str

            if eh_413:
                raise RuntimeError(
                    f"❌ Prompt muito grande para o modelo ({len(prompt.split())} palavras aprox). "
                    "Reduza o tamanho do contexto ou use lotes menores."
                ) from e

            if not eh_429:
                raise

            # TPD esgotado — tenta próximo modelo da lista
            if eh_tpd:
                proximo = _avancar_modelo()
                if proximo:
                    logger.warning("[gerar_resposta] TPD esgotado. Trocando para '%s'", proximo)
                    st.warning(f"⚠️ Limite diário atingido. Trocando para **{proximo}**...")
                    continue  # tenta novamente com o novo modelo sem contar tentativa
                else:
                    raise RuntimeError(
                        "❌ Limite diário esgotado em todos os modelos disponíveis. "
                        "Tente novamente amanhã."
                    )

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
                "[gerar_resposta] 429 Groq/%s (tentativa %d/%d). Aguardando %.1fs...",
                _modelo_atual(), tentativa, max_tentativas, espera,
            )
            st.warning(
                f"⏳ Limite da API Groq atingido. Aguardando {espera:.0f}s... "
                f"(tentativa {tentativa}/{max_tentativas})"
            )
            time.sleep(espera)
            time.sleep(espera)


# ── consolidação via Gemini (contexto maior, sem limite de TPD diário restrito) ──
def _consolidar_gemini(prompt: str, max_tentativas: int = 5) -> str:
    """
    Usa o Gemini 2.0 Flash Lite para consolidar resumos parciais.
    Vantagens: contexto de 1M tokens, sem consumir TPD do Groq.
    """
    from google.genai import types as gtypes
    for tentativa in range(1, max_tentativas + 1):
        try:
            resp = cliente_gemini.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                )
            )
            return resp.text.strip()
        except Exception as e:
            erro = str(e)
            if "429" in erro or "RESOURCE_EXHAUSTED" in erro:
                espera = min(30 * (2 ** (tentativa - 1)), 120)  # 30s, 60s, 120s...
                logger.warning("[consolidar_gemini] 429 — aguardando %ds...", espera)
                time.sleep(espera)
            else:
                raise
    raise RuntimeError("Falha na consolidação via Gemini após múltiplas tentativas.")


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
            "prompt_lote": lambda pdf, texto: f"""Voce e um professor experiente explicando o documento '{pdf}' para um aluno iniciante.
Abaixo estao trechos do documento separados por ---PARTE---.

Para CADA parte do texto:
1. Explique detalhadamente o que ela significa em linguagem simples e acessivel
2. Destaque os conceitos, regras, prazos, valores ou obrigacoes mais importantes
3. Use exemplos praticos do cotidiano para ilustrar quando possivel
4. Nao omita informacoes relevantes — seja completo e extenso

Escreva pelo menos 3 paragrafos por parte. Organize em topicos numerados, um por parte.

TRECHOS:
{texto}

EXPLICACAO DIDATICA DETALHADA DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Voce recebeu explicacoes didaticas detalhadas de varias secoes do documento '{pdf}'.
Consolide tudo em um unico documento explicativo completo, organizado e coeso.

Instrucoes:
- Mantenha TODAS as informacoes importantes de cada secao — nao resuma demais
- Organize por temas ou capitulos logicos
- Use linguagem simples mas seja extenso e detalhado
- Inclua todos os conceitos, regras, prazos e obrigacoes mencionados
- O texto final deve ser longo o suficiente para substituir a leitura do documento original

SECOES EXPLICADAS:
{consolidado}

DOCUMENTO EXPLICATIVO FINAL COMPLETO:"""
        },
        "🔬 Técnico": {
            "descricao": "Analisa cada parte com linguagem técnica, mantendo termos e referências do documento.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um especialista tecnico fazendo analise aprofundada do documento '{pdf}'.
Abaixo estao trechos do documento separados por ---PARTE---.

Para CADA parte do texto:
1. Faca uma analise tecnica completa e detalhada
2. Preserve todos os termos tecnicos, referencias normativas, artigos, incisos e paragrafos citados
3. Destaque dados quantitativos, prazos, percentuais e valores exatos
4. Explique as implicacoes tecnicas e juridicas de cada dispositivo
5. Nao omita nenhum detalhe tecnico relevante

Escreva analises extensas. Organize em topicos numerados, um por parte.

TRECHOS:
{texto}

ANALISE TECNICA DETALHADA DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide as analises tecnicas detalhadas das secoes do documento '{pdf}'.

Instrucoes:
- Mantenha TODOS os termos tecnicos, referencias e dados quantitativos
- Organize por temas, capitulos ou dispositivos legais
- Preserve artigos, incisos, paragrafos e referencias normativas
- O resultado deve ser um documento tecnico completo e extenso
- Nao simplifique — mantenha o rigor tecnico integral

ANALISES:
{consolidado}

ANALISE TECNICA FINAL CONSOLIDADA E COMPLETA:"""
        },
        "⚡ Resumido": {
            "descricao": "Resume cada parte destacando os pontos essenciais com clareza e objetividade.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um especialista fazendo um resumo executivo do documento '{pdf}'.
Abaixo estao trechos do documento separados por ---PARTE---.

Para CADA parte do texto:
1. Identifique e liste TODOS os pontos principais e secundarios relevantes
2. Preserve prazos, valores, percentuais e obrigacoes especificas
3. Seja objetivo mas completo — nao omita informacoes importantes
4. Escreva em bullet points claros e diretos

Organize em topicos numerados, um por parte, com varios bullet points cada.

TRECHOS:
{texto}

PONTOS-CHAVE DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide os pontos-chave das secoes do documento '{pdf}'.

Instrucoes:
- Agrupe os pontos por tema ou categoria logica
- Mantenha TODOS os dados especificos: prazos, valores, percentuais, obrigacoes
- O resultado deve ser um resumo executivo completo, nao um resumo de resumo
- Use bullet points e seja extenso — cubra todos os topicos identificados

PONTOS-CHAVE:
{consolidado}

RESUMO EXECUTIVO COMPLETO:"""
        },
        "🧠 Analítico": {
            "descricao": "Analisa criticamente cada parte, identificando argumentos, evidências e conclusões.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um analista critico especializado examinando o documento '{pdf}'.
Abaixo estao trechos do documento separados por ---PARTE---.

Para CADA parte do texto, desenvolva uma analise critica completa identificando:
1. Argumento ou ideia central — explique em profundidade
2. Evidencias, dados e fundamentos apresentados
3. Implicacoes praticas e consequencias juridicas ou operacionais
4. Pontos fortes, limitacoes ou potenciais problemas
5. Relacao com outras partes do documento quando pertinente

Escreva analises extensas com pelo menos 4 paragrafos por parte.

TRECHOS:
{texto}

ANALISE CRITICA DETALHADA DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide a analise critica detalhada das secoes do documento '{pdf}'.

Instrucoes:
- Integre as analises em um documento critico coeso e extenso
- Identifique padroes, contradicoes e temas recorrentes
- Apresente conclusoes fundamentadas sobre o documento como um todo
- Mantenha todas as observacoes criticas relevantes de cada secao
- O resultado deve ser uma analise critica completa e aprofundada

ANALISES:
{consolidado}

ANALISE CRITICA FINAL INTEGRADA E COMPLETA:"""
        },
        "📋 Comparativo": {
            "descricao": "Compara cada parte com conceitos similares, destacando diferenças e inovações.",
            "prompt_lote": lambda pdf, texto: f"""Voce e um especialista comparando o conteudo do documento '{pdf}' com conhecimentos estabelecidos na area.
Abaixo estao trechos do documento separados por ---PARTE---.

Para CADA parte do texto:
1. Compare detalhadamente com versoes anteriores, legislacoes similares ou praticas comuns da area
2. Destaque o que e novo, diferente, mais restritivo ou mais permissivo
3. Explique o impacto pratico das diferencas identificadas
4. Aponte inovacoes, retrocessos ou mudancas significativas
5. Contextulize no cenario atual da area

Escreva comparacoes extensas com pelo menos 3 paragrafos por parte.

TRECHOS:
{texto}

COMPARACAO DETALHADA DE CADA PARTE:""",
            "prompt_final": lambda pdf, consolidado: f"""Consolide as comparacoes detalhadas das secoes do documento '{pdf}'.

Instrucoes:
- Apresente uma visao comparativa completa e integrada
- Organize por temas ou areas de comparacao
- Destaque as principais mudancas e inovacoes do documento
- Mantenha todos os detalhes comparativos relevantes de cada secao
- O resultado deve ser um documento comparativo extenso e completo

COMPARACOES:
{consolidado}

ANALISE COMPARATIVA FINAL COMPLETA:"""
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

                _resetar_modelo()  # reseta para kimi-k2 a cada novo resumo
                LOTE              = 5    # kimi-k2: 10k TPM — 5 chunks (~3k tokens entrada)
                PAUSA_ENTRE_LOTES = 8    # kimi tem 60 RPM — pausa menor
                MAX_PARALLEL      = 5    # 5 paralelas dentro dos 60 RPM
                total_lotes       = (len(chunks) + LOTE - 1) // LOTE
                resumos_parciais  = [None] * total_lotes
                barra             = st.progress(0, text="Analisando partes...")

                import concurrent.futures

                def processar_lote(args):
                    idx, lote = args
                    texto_lote = "\n\n---PARTE---\n\n".join(lote)
                    prompt     = estilo["prompt_lote"](pdf_sel, texto_lote)
                    try:
                        return idx, gerar_resposta(prompt)
                    except Exception as e:
                        return idx, f"[Erro no lote {idx+1}: {e}]"

                lotes = [(j, chunks[j*LOTE:(j+1)*LOTE]) for j in range(total_lotes)]

                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
                    futuros = {ex.submit(processar_lote, l): l[0] for l in lotes}
                    concluidos = 0
                    for futuro in concurrent.futures.as_completed(futuros):
                        idx, resultado = futuro.result()
                        resumos_parciais[idx] = resultado
                        concluidos += 1
                        barra.progress(concluidos / total_lotes,
                                       text=f"Parte {concluidos}/{total_lotes}...")

                barra.empty()

                if len(resumos_parciais) == 1:
                    resumo_final = resumos_parciais[0]
                else:
                    GRUPO = 5
                    grupos = [resumos_parciais[g:g+GRUPO] for g in range(0, len(resumos_parciais), GRUPO)]
                    intermediarios = [None] * len(grupos)

                    barra_cons  = st.progress(0, text="Consolidando partes...")
                    status_cons = st.empty()

                    def _consolidar_grupo(args):
                        gi, grupo = args
                        sub        = "\n\n===\n\n".join([f"Secao {i+1}:\n{r}" for i, r in enumerate(grupo)])
                        prompt_sub = estilo["prompt_final"](pdf_sel, sub)
                        try:
                            return gi, _consolidar_gemini(prompt_sub)
                        except Exception as e:
                            return gi, f"[Erro grupo {gi+1}: {e}]"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                        futuros_cons = {ex.submit(_consolidar_grupo, (gi, g)): gi for gi, g in enumerate(grupos)}
                        concluidos_cons = 0
                        for futuro in concurrent.futures.as_completed(futuros_cons):
                            gi, resultado = futuro.result()
                            intermediarios[gi] = resultado
                            concluidos_cons += 1
                            barra_cons.progress(concluidos_cons / len(grupos),
                                                text=f"Consolidando grupo {concluidos_cons}/{len(grupos)}...")

                    barra_cons.progress(1.0, text="Gerando consolidação final...")
                    status_cons.caption("⏳ Gerando texto final consolidado...")

                    if len(intermediarios) == 1:
                        resumo_final = intermediarios[0]
                    else:
                        MAX_PALAVRAS_PARTE = 300
                        partes_truncadas = []
                        for i, r in enumerate(intermediarios):
                            palavras = r.split()
                            truncado = " ".join(palavras[:MAX_PALAVRAS_PARTE])
                            if len(palavras) > MAX_PALAVRAS_PARTE:
                                truncado += " [...]"
                            partes_truncadas.append(f"Parte {i+1}:\n{truncado}")

                        consolidado_final = "\n\n===\n\n".join(partes_truncadas)
                        prompt_final = estilo["prompt_final"](pdf_sel, consolidado_final)
                        try:
                            resumo_final = _consolidar_gemini(prompt_final)
                        except Exception as e:
                            resumo_final = f"(Erro final: {e})\n\n" + consolidado_final

                    barra_cons.empty()
                    status_cons.empty()

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