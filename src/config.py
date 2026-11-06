"""
Configuracao central: caminhos do projeto e variaveis de ambiente.

Tudo que muda entre maquinas (endereco do Ollama, nome do modelo) vem do .env;
tudo que e estrutura do projeto (onde ficam os dados) fica aqui, num lugar so.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

# --- Documentos internos do hospital (sinteticos, versionados) ---
PASTA_PROTOCOLOS = RAIZ / "data" / "protocolos"
PASTA_PROCEDIMENTOS = RAIZ / "data" / "procedimentos"
PASTA_MODELOS = RAIZ / "data" / "modelos"
PASTAS_DOCUMENTOS = (PASTA_PROTOCOLOS, PASTA_PROCEDIMENTOS, PASTA_MODELOS)

# --- Dados gerados ---
PASTA_BRUTO = RAIZ / "data" / "bruto"            # com PII falsa, fora do Git
PASTA_EXTERNO = RAIZ / "data" / "externo"        # downloads do Hugging Face
PASTA_PROCESSADO = RAIZ / "data" / "processado"  # dataset final, versionado
PASTA_INDICE = RAIZ / "data" / "indice"          # FAISS, regerado
ARQUIVO_BANCO = RAIZ / "data" / "prontuarios.db"
PASTA_OUTPUTS = RAIZ / "outputs"

SEMENTE = int(os.getenv("RANDOM_SEED", "42"))

# --- Assistente ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODELO_AJUSTADO = os.getenv("MODELO_AJUSTADO", "assistente-medico")
TEMPERATURA = float(os.getenv("TEMPERATURA", "0.1"))
MODELO_EMBEDDINGS = os.getenv(
    "MODELO_EMBEDDINGS", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
ARQUIVO_AUDITORIA = RAIZ / os.getenv("ARQUIVO_AUDITORIA", "logs/auditoria.jsonl")
