"""
Estados tipados dos grafos (camada de estado da Aula 2 de LangGraph).

Os nos devolvem so as chaves que alteram; `trilha` e acumulativa
(`Annotated[list, operator.add]`): cada no acrescenta o seu passo, sem mutar o
que ja estava la — e a trilha legivel do que aconteceu, ao lado do log de
auditoria estruturado.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class EstadoFluxoPaciente(TypedDict, total=False):
    execucao: str
    paciente_id: str
    erro: str
    dados_paciente: str          # resumo anonimizado do prontuario
    achados: str                 # valores criticos e pendencias, em texto
    alertas: list[dict]          # achados CRITICO/ALTO calculados pelas regras
    alertas_emitidos: list[dict]
    fontes_contexto: list[str]
    contexto: str
    confianca: float
    sugestao: str
    origem_sugestao: str         # "llm_ajustada" | "regras (fallback)"
    verificacao: dict
    tentativas: int
    decisao_medica: dict
    trilha: Annotated[list[str], operator.add]


class EstadoPergunta(TypedDict, total=False):
    execucao: str
    pergunta: str
    paciente_id: str
    bloqueio: dict
    dados_paciente: str
    fontes_contexto: list[str]
    contexto: str
    confianca: float
    resposta: str
    verificacao: dict
    tentativas: int
    trilha: Annotated[list[str], operator.add]
