"""
gerar_pdf.py — Gera PDFs formatados de resumos e respostas
Usa fpdf2 para evitar problemas de encoding
"""

from fpdf import FPDF
from datetime import datetime
import io
import re


def _sanitizar(texto: str) -> str:
    """Remove caracteres fora do range latin-1."""
    substituicoes = {
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u2022": "-",
        "\u20ac": "EUR", "\u00e2": "a",
        "\u2192": "->", "\u2190": "<-",
        "\u2605": "*", "\u2713": "OK",
    }
    for orig, sub in substituicoes.items():
        texto = texto.replace(orig, sub)
    # Remove markdown bold/italic simples
    texto = re.sub(r"\*\*(.+?)\*\*", r"\1", texto)
    texto = re.sub(r"\*(.+?)\*",     r"\1", texto)
    texto = re.sub(r"#+\s*",         "",    texto)
    return texto.encode("latin-1", errors="replace").decode("latin-1")


class PDFRelatorio(FPDF):
    def __init__(self, titulo: str):
        super().__init__()
        self.titulo_doc = _sanitizar(titulo)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self._cabecalho()

    def _cabecalho(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 12, self.titulo_doc, ln=True, align="C")
        self.set_font("Arial", "I", 9)
        self.cell(0, 6, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
        self.ln(6)
        self.set_draw_color(100, 100, 200)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def secao(self, titulo: str):
        self.set_font("Arial", "B", 13)
        self.set_fill_color(230, 235, 255)
        self.cell(0, 9, _sanitizar(titulo), ln=True, fill=True)
        self.ln(2)

    def subsecao(self, titulo: str):
        self.set_font("Arial", "B", 11)
        self.cell(0, 7, _sanitizar(titulo), ln=True)
        self.ln(1)

    def corpo(self, texto: str):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, _sanitizar(texto))
        self.ln(3)

    def separador(self):
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def rodape(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def gerar_pdf_resumo(nome_pdf: str, banco: str, resumo: str, estilo: str = "") -> bytes:
    """Gera PDF de um resumo completo."""
    pdf = PDFRelatorio(f"Resumo: {nome_pdf}")

    pdf.secao("Informacoes")
    pdf.corpo(f"Documento: {nome_pdf}")
    pdf.corpo(f"Banco: {banco}")
    if estilo:
        pdf.corpo(f"Tipo de resumo: {estilo}")
    pdf.separador()

    pdf.secao("Resumo Completo")
    pdf.corpo(resumo)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def gerar_pdf_resposta(pergunta: str, resposta: str, banco: str, trechos: list) -> bytes:
    """Gera PDF de uma pergunta e sua resposta com trechos utilizados."""
    pdf = PDFRelatorio("Resposta RAG")

    pdf.secao("Informacoes")
    pdf.corpo(f"Banco: {banco}")
    pdf.separador()

    pdf.secao("Pergunta")
    pdf.corpo(pergunta)
    pdf.separador()

    pdf.secao("Resposta")
    pdf.corpo(resposta)
    pdf.separador()

    pdf.secao("Trechos Utilizados")
    for i, t in enumerate(trechos, 1):
        pdf.subsecao(f"#{i} - {t['arquivo']} | rerank: {t['relevancia_rerank']} | embed: {t['relevancia']}")
        pdf.corpo(t["texto"])
        if i < len(trechos):
            pdf.separador()

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()