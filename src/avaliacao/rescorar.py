"""
Recalcula as verificacoes de comportamento a partir das respostas salvas no Colab.

O notebook grava todas as respostas geradas (base e ajustado) em
`respostas_teste.jsonl`. Se a rubrica for corrigida depois do treino — como
aconteceu com a de perguntas sobre pacientes, que reprovava respostas certas
escritas com outras palavras —, as metricas podem ser recalculadas aqui, sem GPU
e sem gerar nada de novo, sobre as mesmas respostas.

Uso:
    python -m src.avaliacao.rescorar outputs/fine_tuning
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.avaliacao import metricas
from src.config import PASTA_PROCESSADO


def rescorar(pasta: Path, arquivo_teste: Path = PASTA_PROCESSADO / "teste.jsonl") -> dict:
    with open(arquivo_teste, encoding="utf-8") as f:
        teste = {e["id"]: e for e in map(json.loads, f)}
    with open(pasta / "respostas_teste.jsonl", encoding="utf-8") as f:
        respostas = [json.loads(l) for l in f]
    faltando = [r["id"] for r in respostas if r["id"] not in teste]
    if faltando:
        raise ValueError(f"{len(faltando)} respostas sem exemplo no teste (dataset diferente do avaliado?)")
    internos = [r for r in respostas if teste[r["id"]]["origem"] == "interno"]
    m = json.loads((pasta / "metricas_avaliacao.json").read_text(encoding="utf-8"))
    for modelo in ("base", "ajustado"):
        m[modelo]["comportamento"] = metricas.agregar(
            [(r["tarefa"], metricas.verificar(teste[r["id"]], r[modelo])) for r in internos])
    m["rescorado"] = "verificações recalculadas localmente com src/avaliacao/rescorar.py (rubrica atual)"
    (pasta / "metricas_avaliacao.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


if __name__ == "__main__":
    resultado = rescorar(Path(sys.argv[1]))
    for modelo in ("base", "ajustado"):
        print(modelo, json.dumps(resultado[modelo]["comportamento"], ensure_ascii=False))
