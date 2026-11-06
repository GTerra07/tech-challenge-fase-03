"""
Curadoria do dataset: o que entra no treino e por que o resto saiu.

Etapa 3 do requisito de preparacao. Cada filtro registra quantos exemplos
removeu, e o relatorio vira tabela no relatorio tecnico — "curadoria" sem
numeros seria so uma palavra.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

from src.seguranca.anonimizacao import residuos_pii

# O treino usa max_seq_length=2048 tokens (Aula 2). Estimativa conservadora de
# ~3,5 caracteres por token para portugues/ingles tecnico.
MAX_CARACTERES = int(1900 * 3.5)
MIN_CARACTERES_RESPOSTA = 20
# Tag HTML de verdade ("<br>", "</p>"), nao comparacao estatistica ("p < 0.05 e > 2")
_RE_MARCACAO = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,10}(?:\s[^<>]{0,60})?/?>|&[a-z]+;|\{\{|\}\}")


@dataclass
class Relatorio:
    entrada: Counter = field(default_factory=Counter)
    removidos: dict[str, Counter] = field(default_factory=dict)
    saida: Counter = field(default_factory=Counter)

    def remover(self, motivo: str, ex: dict):
        self.removidos.setdefault(motivo, Counter())[ex["origem"]] += 1

    def como_dict(self) -> dict:
        return {"entrada": dict(self.entrada),
                "removidos": {m: dict(c) for m, c in self.removidos.items()},
                "saida": dict(self.saida)}


def _chave(ex: dict) -> str:
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    return hashlib.sha1(f"{norm(ex['instruction'])}|{norm(ex['input'])}|{norm(ex['output'])}".encode()).hexdigest()


def curar(exemplos: list[dict], pii_por_id: dict[str, list[str]], relatorio: Relatorio) -> list[dict]:
    """
    Filtros, na ordem:
    1. resposta vazia ou curta demais;
    2. marcacao HTML/template residual;
    3. tamanho acima do contexto de treino;
    4. dado pessoal residual (padrao estruturado ou valor do gabarito) — so
       conteudo interno, que e onde ha dado pessoal;
    5. duplicata (apos normalizacao).
    """
    vistos: set[str] = set()
    aprovados = []
    for ex in exemplos:
        relatorio.entrada[ex["origem"]] += 1
        if len(ex["output"]) < MIN_CARACTERES_RESPOSTA:
            relatorio.remover("resposta vazia ou curta", ex)
            continue
        if _RE_MARCACAO.search(ex["input"] + ex["output"]):
            relatorio.remover("marcação residual", ex)
            continue
        if len(ex["instruction"]) + len(ex["input"]) + len(ex["output"]) > MAX_CARACTERES:
            relatorio.remover("acima do contexto de treino", ex)
            continue
        if ex["origem"] == "interno":
            texto = ex["input"] + "\n" + ex["output"]
            if residuos_pii(texto, pii_por_id.get(ex["id_bruto"], [])):
                relatorio.remover("dado pessoal residual", ex)
                continue
        chave = _chave(ex)
        if chave in vistos:
            relatorio.remover("duplicata", ex)
            continue
        vistos.add(chave)
        aprovados.append(ex)
        relatorio.saida[ex["origem"]] += 1
    return aprovados
