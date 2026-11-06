"""
Verificacao factual da resposta contra o prontuario (barreira de saida).

Motivo: na avaliacao de ponta a ponta, perguntado sobre a hemocultura de uma
paciente depois que o laboratorio liberou "positiva", o modelo (v1 e v2)
respondeu "pendente" — com o prontuario correto no contexto. O dataset nunca
teve hemocultura positiva, e o modelo seguiu o padrao que conhecia.

Para perguntas sobre exames, pendencias e valores criticos de um paciente, a
resposta certa nao depende do modelo: sai das regras sobre o prontuario
(`perguntas_paciente.resposta`). Aqui comparamos o que a resposta do modelo
AFIRMA (`metricas.situacao`) com o que o prontuario diz. Divergiu: a nova
tentativa recebe o que o prontuario indica; persistindo, o grafo entrega a
resposta montada a partir dos dados. E o principio do projeto aplicado a saida:
as regras decidem, a LLM redige.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from src.avaliacao.metricas import situacao, tipo_da_pergunta
from src.dados.perguntas_paciente import resposta as resposta_do_prontuario


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def exame_citado(pergunta: str, registro: dict) -> str | None:
    """O exame do paciente mencionado na pergunta (o nome mais longo, para 'potassio de controle')."""
    n = _norm(pergunta)
    citados = [e["exame"] for e in registro.get("exames", []) if _norm(e["exame"]) in n]
    return max(citados, key=len) if citados else None


@dataclass
class Checagem:
    aplicavel: bool
    coerente: bool = True
    esperado: str = ""

    @property
    def problema(self) -> str:
        return f"diverge do prontuário — o prontuário indica: {self.esperado}"


def checar(pergunta: str, resposta: str, registro: dict) -> Checagem:
    tipo = tipo_da_pergunta(pergunta)
    exame = exame_citado(pergunta, registro) if tipo == "exame" else None
    if tipo == "exame" and exame is None:
        return Checagem(aplicavel=False)   # pergunta sobre o paciente, mas nao sobre um exame dele
    esperado = resposta_do_prontuario(registro, tipo, exame)
    return Checagem(True, situacao(resposta, tipo) == situacao(esperado, tipo), esperado)
