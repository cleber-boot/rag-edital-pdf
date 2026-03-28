---
title: RAG com PDFs — Codespaces
---

# 📚 RAG com PDFs

Sistema de perguntas e respostas sobre documentos PDF usando RAG (Retrieval-Augmented Generation).
Versão para rodar localmente no **GitHub Codespaces** ou qualquer ambiente Python.

## Funcionalidades

- 📥 **Indexar PDFs** em bancos vetoriais separados e nomeados
- 🔍 **Perguntar** sobre os documentos com busca semântica + reranking
- 📄 **Resumir** qualquer PDF completo indexado
- 📊 **Status** dos bancos e documentos disponíveis

## Configuração rápida

### 1. Clone e abra no Codespaces
```bash
gh repo clone seu-usuario/seu-repo
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Configure a chave do Gemini
```bash
cp .env.example .env
# Edite o .env e adicione sua GEMINI_API_KEY
# Obtenha em: https://aistudio.google.com
```

### 4. Rode a aplicação
```bash
streamlit run app.py
```

O Codespaces abrirá automaticamente a porta do Streamlit no browser.

## Arquitetura

| Componente | Tecnologia |
|---|---|
| Interface | Streamlit |
| Embeddings | Gemini `gemini-embedding-001` |
| LLM | Gemini `gemini-2.5-flash-lite` |
| Banco vetorial | ChromaDB (local em `./chroma_bancos/`) |
| Reranking | Flashrank |
| Extração de PDF | PyMuPDF |

## Estrutura do projeto

```
├── app.py              # Interface Streamlit principal
├── indexar.py          # Processamento e indexação de PDFs
├── perguntar.py        # Busca, rerank e contexto
├── requirements.txt
├── .env.example        # Modelo do arquivo de configuração
├── .gitignore
└── chroma_bancos/      # Bancos vetoriais (criado automaticamente)
    ├── banco_1/
    ├── banco_2/
    └── ...
```

## Limites do plano gratuito (Gemini AI Studio)

| Modelo | Req/dia | Req/min |
|---|---|---|
| `gemini-2.5-flash-lite` | 500 | 15 |
| `gemini-embedding-001` | 1.500 | 100 |

O sistema tem retry automático com pausa ao receber erro 429.
