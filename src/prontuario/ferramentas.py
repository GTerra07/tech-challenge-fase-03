"""
Ferramentas LangChain de consulta ao prontuario (requisito 2).

Cada ferramenta e um `@tool` com docstring — o contrato que descreve o que ela
faz (Aula 6 de LangChain). Tres regras de seguranca valem para todas:

1. **So leitura**: usam `conectar_leitura`, aberta em modo somente leitura.
2. **SQL parametrizado e fixo**: o modelo nunca escreve SQL; no maximo informa o
   identificador do caso, que e validado por expressao regular antes da consulta.
3. **Minimizacao de dados (LGPD)**: o que sai daqui ja passou pela anonimizacao —
   nome, CPF, prontuario, telefone e endereco nunca chegam ao modelo.

Quem chama as ferramentas e o grafo, de forma deterministica, e nao a LLM: o
modelo ajustado nao decide consultas, so redige a partir do que foi consultado.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from src.prontuario import banco, regras
from src.prontuario.resumo import bloco_achados, contexto_paciente

_RE_ID = re.compile(r"^P-\d{4}$")


class PacienteNaoEncontrado(LookupError):
    pass


def _carregar(paciente_id: str) -> dict:
    paciente_id = paciente_id.strip().upper()
    if not _RE_ID.match(paciente_id):
        raise ValueError(f"identificador de paciente inválido: {paciente_id!r} (formato esperado: P-0000)")
    p = banco.carregar_paciente(paciente_id)
    if p is None:
        raise PacienteNaoEncontrado(paciente_id)
    return p


@tool
def consultar_prontuario(paciente_id: str) -> str:
    """Retorna o resumo atualizado do prontuário de um paciente (idade, sexo, leito, diagnóstico,
    protocolos ativos, alergias, sinais vitais e exames), já anonimizado. Use o identificador
    interno do paciente, no formato P-0000."""
    return contexto_paciente(_carregar(paciente_id))


@tool
def consultar_pendencias(paciente_id: str) -> str:
    """Retorna os valores críticos e os exames pendentes de um paciente, calculados pelas regras
    dos protocolos ativos, com o nível de alerta e a seção do protocolo que define cada um."""
    return bloco_achados(_carregar(paciente_id))


@tool
def listar_pacientes() -> str:
    """Lista os pacientes internados com identificador, leito e diagnóstico (sem dados pessoais)."""
    return "\n".join(f"{p['id']} | leito {p['leito']} | {p['diagnostico']}" for p in banco.listar_pacientes())


FERRAMENTAS = [consultar_prontuario, consultar_pendencias, listar_pacientes]


def carregar_para_regras(paciente_id: str) -> dict:
    """Registro completo para as regras deterministicas (uso interno do grafo, nunca do modelo)."""
    return _carregar(paciente_id)


def alertas_do_paciente(paciente_id: str) -> list[regras.Achado]:
    return regras.alertas(_carregar(paciente_id))
