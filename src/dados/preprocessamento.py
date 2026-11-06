"""
Pre-processamento de texto aplicado a todo exemplo antes da curadoria.

Etapa 1 do requisito "preparar os dados com tecnicas de preprocessing,
anonimizacao e curadoria". Normaliza o que varia entre fontes sem mudar
significado: Unicode, espacos, caracteres de controle e artefatos de formatacao
dos datasets publicos.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ESPACOS = re.compile(r"[ \t ]+")
_QUEBRAS = re.compile(r"\n{3,}")
# MedQuAD escreve "What is (are) Glaucoma ?" — espaco antes da interrogacao
_ESPACO_PONTUACAO = re.compile(r"\s+([?.,;:!])")


def limpar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto or "")
    texto = _CONTROLE.sub("", texto).replace("\r\n", "\n").replace("\r", "\n")
    linhas = [_ESPACOS.sub(" ", l).strip() for l in texto.split("\n")]
    texto = _QUEBRAS.sub("\n\n", "\n".join(linhas))
    return _ESPACO_PONTUACAO.sub(r"\1", texto).strip()


def limpar_exemplo(ex: dict) -> dict:
    """Aplica a limpeza aos tres campos do formato Alpaca."""
    return {**ex, **{k: limpar(ex[k]) for k in ("instruction", "input", "output")}}
