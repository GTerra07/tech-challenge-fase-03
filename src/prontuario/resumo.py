"""
Resumo textual do paciente para o prompt.

O resumo e montado a partir do registro completo e SEMPRE passa pela
anonimizacao antes de sair daqui: e o mesmo caminho no dataset de treino e no
assistente. O modelo nunca recebe nome, CPF ou numero de prontuario — so o
identificador interno do caso, idade, sexo, leito e os dados clinicos.
"""

from __future__ import annotations

from src.prontuario import regras
from src.seguranca.anonimizacao import anonimizar


def _valor(e: dict) -> str:
    if e.get("status") != "resultado":
        return f"PENDENTE (solicitado há {regras._formata_prazo(e.get('horas_desde_solicitacao', 0))})"
    v = e.get("valor")
    v = regras._num(v) if isinstance(v, (int, float)) else str(v)
    return f"{v} {e.get('unidade', '')}".strip()


def resumo_bruto(p: dict) -> str:
    """Resumo com os identificadores do prontuario (NAO enviar ao modelo)."""
    sv = p.get("sinais_vitais", {})
    linhas = [
        f"Paciente: {p['nome']} | CPF {p['cpf']} | Prontuário nº {p['prontuario']} | nascimento {p['data_nascimento']}",
        f"Caso {p['id']} | {p['idade']} anos | sexo {p['sexo']} | leito {p['leito']}",
        f"Diagnóstico/hipótese: {p['diagnostico']}",
        f"Protocolos ativos: {', '.join(p.get('protocolos_ativos', [])) or 'nenhum'}",
        f"Alergias: {', '.join(p.get('alergias', [])) or 'nenhuma registrada'}",
    ]
    if sv:
        linhas.append("Sinais vitais: " + ", ".join(f"{k.upper()} {regras._num(v)}" for k, v in sv.items()))
    if p.get("exames"):
        linhas.append("Exames:")
        linhas += [f"- {e['exame']}: {_valor(e)}" for e in p["exames"]]
    return "\n".join(linhas)


def contexto_paciente(p: dict) -> str:
    """Resumo pronto para o prompt: anonimizado no ponto unico do projeto."""
    return anonimizar(resumo_bruto(p))


def bloco_achados(p: dict) -> str:
    """Pendencias e valores criticos calculados pelas regras, em texto."""
    criticos = regras.valores_criticos(p)
    pendentes = regras.exames_pendentes(p)
    partes = ["VALORES CRÍTICOS:"]
    partes += [f"- {a.linha()}" for a in criticos] or ["- nenhum"]
    partes.append("EXAMES PENDENTES:")
    partes += [f"- {a.linha()}" for a in pendentes] or ["- nenhum"]
    return "\n".join(partes)


def sugestao_de_referencia(p: dict) -> str:
    """
    Sugestao montada direto das regras — o alvo dos casos de treino.

    No assistente, quem escreve a sugestao e a LLM ajustada; esta funcao serve
    para gerar o exemplo de treino e como referencia na avaliacao.
    """
    alertas = regras.alertas(p)
    linhas = [f"RESUMO: caso {p['id']}, {p['idade']} anos, sexo {p['sexo']}, leito {p['leito']}, "
              f"{p['diagnostico']}. Protocolos ativos: {', '.join(p.get('protocolos_ativos', [])) or 'nenhum'}."]
    if alertas:
        linhas.append("ALERTAS PARA A EQUIPE:")
        linhas += [f"- {a.linha()}" for a in alertas]
        linhas.append("SUGESTÃO (RASCUNHO PARA VALIDAÇÃO MÉDICA):")
        vistas = set()
        for a in alertas:
            acao = regras.acao_sugerida(a)
            if acao not in vistas:
                vistas.add(acao)
                linhas.append(f"- {acao}")
        if any(a.nivel == regras.CRITICO for a in alertas):
            linhas.append("- Comunicar o médico responsável em até 30 minutos, com registro e leitura de volta (PROT-007 §3 e §4; POP-002).")
    else:
        linhas.append("ALERTAS PARA A EQUIPE: nenhum valor crítico nem pendência atrasada.")
        linhas.append("SUGESTÃO (RASCUNHO PARA VALIDAÇÃO MÉDICA):")
        linhas.append("- Manter o seguimento previsto nos protocolos ativos; nenhuma ação adicional identificada pelas regras.")
    linhas.append("Esta sugestão não é prescrição: a conduta é decisão do médico responsável (PROT-008 §3).")
    return "\n".join(linhas)
