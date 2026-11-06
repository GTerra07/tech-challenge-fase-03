"""
Perguntas de medicos sobre um paciente especifico, com o prontuario no contexto
(dataset v2).

O modelo v1 so tinha visto o prontuario dentro dos casos do fluxo automatizado;
na demonstracao, perguntado sobre a hemocultura de uma paciente depois que o
resultado chegou, leu o prontuario errado ("pendente ha 0,1 h" para um exame ja
positivo). O v2 treina este formato — o mesmo do grafo de perguntas
(`formato.contexto_com_prontuario`) — com respostas tiradas das regras: a LLM
aprende a narrar o que o prontuario e as regras dizem, sem inventar.

Tres tipos de pergunta, cada um com tres redacoes:
- pendencias: o que esta pendente pelos protocolos ativos;
- criticos: ha valor critico nos resultados;
- exame: a situacao de um exame especifico.
"""

from __future__ import annotations

import random

from src.prontuario import regras

PERGUNTAS = {
    "pendencias": ("Quais exames estão pendentes do {pid}?", "O {pid} tem alguma pendência de protocolo?",
                   "Tem exame atrasado para o paciente {pid}?"),
    "criticos": ("O {pid} tem algum valor crítico?", "Algum resultado do {pid} exige comunicação imediata?",
                 "Há valor crítico no caso {pid}?"),
    "exame": ("Qual a situação do exame {exame} do {pid}?", "Como está {exame} no caso {pid}?",
              "Já saiu o resultado de {exame} do {pid}?"),
}


def _valor(e: dict) -> str:
    v = e.get("valor")
    v = regras._num(v) if isinstance(v, (int, float)) else str(v)
    return f"{v} {e.get('unidade', '')}".strip()


def resposta(paciente: dict, tipo: str, exame: str | None = None) -> str:
    pid = paciente["id"]
    pendentes = regras.exames_pendentes(paciente)
    criticos = regras.valores_criticos(paciente)
    if tipo == "pendencias":
        if not pendentes:
            return f"Pelos protocolos ativos, o caso {pid} não tem exames pendentes."
        linhas = [f"Pendências do caso {pid} pelos protocolos ativos:"] + [f"- {a.linha()}" for a in pendentes]
        if any(a.nivel != regras.INFORMATIVO for a in pendentes):
            linhas.append("As pendências CRÍTICO e ALTO devem ser comunicadas à equipe conforme o PROT-007 §3.")
        return "\n".join(linhas)
    if tipo == "criticos":
        if not criticos:
            return f"Nenhum resultado disponível do caso {pid} atinge critério de valor crítico (Fonte: PROT-007 §2)."
        return "\n".join([f"Valores críticos do caso {pid}:"] + [f"- {a.linha()}" for a in criticos] +
                         ["Devem ser comunicados ao médico responsável em até 30 minutos (Fonte: PROT-007 §3)."])
    # tipo == "exame"
    registro = next(e for e in paciente["exames"] if e["exame"] == exame)
    critico = next((a for a in criticos if a.exame == exame), None)
    pendente = next((a for a in pendentes if a.exame == exame), None)
    if registro.get("status") == "resultado":
        texto = f"{exame.capitalize()} do caso {pid}: resultado disponível, {_valor(registro)}."
        if critico:
            return (f"{texto} É valor crítico ({critico.descricao}) e deve ser comunicado ao médico responsável "
                    f"em até 30 minutos (Fonte: {critico.fonte}; PROT-007 §3).")
        return f"{texto} Não atinge critério de valor crítico (Fonte: PROT-007 §2)."
    if pendente:
        return f"{exame.capitalize()} do caso {pid}: {pendente.descricao} (Fonte: {pendente.fonte})."
    horas = regras._formata_prazo(registro.get("horas_desde_solicitacao", 0))
    return (f"{exame.capitalize()} do caso {pid}: solicitado há {horas} e ainda sem resultado; os protocolos ativos "
            f"não definem prazo para este exame.")


def sortear(paciente: dict, rng: random.Random) -> tuple[str, str]:
    """(pergunta, resposta) de um tipo e uma redacao sorteados para o paciente."""
    tipo = rng.choice(list(PERGUNTAS))
    exame = rng.choice(paciente["exames"])["exame"] if tipo == "exame" else None
    pergunta = rng.choice(PERGUNTAS[tipo]).format(pid=paciente["id"], exame=exame)
    return pergunta, resposta(paciente, tipo, exame)
