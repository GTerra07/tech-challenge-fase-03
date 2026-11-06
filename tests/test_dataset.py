"""
Invariantes do dataset de fine-tuning.

O mais importante: em toda tarefa que trabalha sobre contexto, cada fonte citada
na resposta de referencia esta no contexto de entrada. Se o treino tiver resposta
citando o que o modelo nao recebeu, ele aprende a inventar citacao — o oposto do
requisito de explainability. (Esse defeito existiu numa versao do dataset e foi
pego pelo teste do grafo; este teste existe para ele nunca mais voltar.)
"""

import json

import pytest

from src.avaliacao.metricas import fontes_citadas
from src.config import PASTA_PROCESSADO

TAREFAS_COM_CONTEXTO = {"pergunta", "caso", "recusa", "sem_contexto", "pergunta_paciente"}


def _todos():
    for nome in ("treino", "validacao", "teste"):
        with open(PASTA_PROCESSADO / f"{nome}.jsonl", encoding="utf-8") as f:
            yield from (json.loads(l) for l in f)


@pytest.mark.parametrize("conjunto", ["treino", "validacao", "teste"])
def test_citacoes_estao_no_contexto(conjunto):
    with open(PASTA_PROCESSADO / f"{conjunto}.jsonl", encoding="utf-8") as f:
        for ex in map(json.loads, f):
            if ex["tarefa"] not in TAREFAS_COM_CONTEXTO:
                continue
            no_contexto = set()
            for linha in ex["input"].splitlines():
                if linha.startswith("[") and "]" in linha:
                    no_contexto |= fontes_citadas(linha.split("]", 1)[0])
            fora = fontes_citadas(ex["output"]) - no_contexto
            assert not fora, (ex["id"], sorted(fora))


def test_teste_nao_vaza_para_o_treino():
    ids = {"treino": set(), "teste": set()}
    entradas = {"treino": set(), "teste": set()}
    for nome in ids:
        with open(PASTA_PROCESSADO / f"{nome}.jsonl", encoding="utf-8") as f:
            for ex in map(json.loads, f):
                ids[nome].add(ex["id"])
                entradas[nome].add((ex["instruction"], ex["input"]))
    assert not ids["treino"] & ids["teste"]
    assert not entradas["treino"] & entradas["teste"]


def test_todas_as_tarefas_do_enunciado_estao_no_treino():
    """Protocolos (pergunta/caso), FAQ de medicos, laudos, receitas e limites (recusa)."""
    with open(PASTA_PROCESSADO / "treino.jsonl", encoding="utf-8") as f:
        tarefas = {json.loads(l)["tarefa"] for l in f}
    assert {"pergunta", "pergunta_paciente", "caso", "laudo", "receita", "recusa", "sem_contexto", "pubmedqa",
            "medquad"} <= tarefas
