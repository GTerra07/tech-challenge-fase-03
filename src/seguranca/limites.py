"""
Limites de atuacao do assistente (requisito 3: "nunca prescrever diretamente,
sem validacao humana").

Defesa em camadas — nenhuma camada sozinha e confiavel:

1. **Fine-tuning**: o modelo aprendeu a recusar pedidos de prescricao (PROT-008),
   inclusive com redacoes nunca vistas no treino (medido no notebook do Colab).
2. **Barreira de entrada** (`triagem_entrada`): regras deterministicas barram
   pedido de prescricao, dose, alta e tentativa de prompt injection ANTES de
   chegar ao modelo. Nao depende de geracao de texto, entao se prova com teste.
3. **Barreira de saida** (`verificar_saida`): a resposta e reprovada se trouxer
   dose de medicamento, se a sugestao nao estiver marcada como rascunho ou se
   citar uma fonte que nao estava no contexto recuperado (citacao inventada).
4. **Validacao humana** obrigatoria no grafo (interrupt) antes de qualquer
   sugestao de conduta ser registrada como aceita.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.avaliacao.metricas import contem_dose, fontes_citadas


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


# Verbos de ato privativo do medico aplicados a medicamento/alta. "Receita" so
# conta como ato quando e pedido de emissao: "o que muda na receita de
# controlado?" e pergunta legitima sobre o modelo MOD-002 (falso positivo medido).
_ATO_MEDICO = re.compile(
    r"\b(prescrev\w*|prescri[cç]\w*|(?:emit|fa[cç]|gere|gera)\w*\s+(?:a\s+|uma\s+)?receita|suspend\w*|suspens\w*|ajust\w* a dose|dose|posologia|"
    r"aument\w* a dose|diminu\w* a dose|lan[cç]\w* na prescri\w*|libera\w* a alta|d[aá] alta|alta para)\b")
_MEDICAMENTO_OU_ALTA = re.compile(
    r"\b(antibi[oó]tic\w*|antimicrobian\w*|insulin\w*|heparin\w*|enoxaparin\w*|morfin\w*|noradrenalin\w*|"
    r"vancomicin\w*|merope\w*|ceftriax\w*|piperacilin\w*|furosemid\w*|amiodaron\w*|dipiron\w*|"
    r"medica\w*|rem[eé]dio\w*|droga\w*|alta)\b")
# Tentativas de mudar as regras do assistente (prompt injection direto).
_INJECAO = re.compile(
    r"(ignore|desconsidere|esque[cç]a)\s+(as\s+|todas\s+as\s+)?(instru|regras|orienta)|"
    r"voc[eê]\s+agora\s+[eé]|finja\s+(que|ser)|sem\s+valida[cç][aã]o|modo\s+desenvolvedor|"
    r"system\s*prompt|jailbreak")


@dataclass
class Triagem:
    permitido: bool
    motivo: str = ""
    categoria: str = ""  # "prescricao" | "injecao" | ""


def triagem_entrada(texto: str) -> Triagem:
    """Barra, antes do modelo, o que o assistente nunca pode atender (PROT-008 §2 e §3)."""
    n = _norm(texto)
    if _INJECAO.search(n):
        return Triagem(False, "tentativa de alterar as regras do assistente (prompt injection)", "injecao")
    if _ATO_MEDICO.search(n) and _MEDICAMENTO_OU_ALTA.search(n):
        return Triagem(False, "pedido de ato privativo do médico (prescrever, dosar, suspender ou dar alta)",
                       "prescricao")
    return Triagem(True)


@dataclass
class VerificacaoSaida:
    aprovada: bool
    problemas: list[str] = field(default_factory=list)
    fontes_citadas: list[str] = field(default_factory=list)
    fontes_nao_recuperadas: list[str] = field(default_factory=list)


def verificar_saida(resposta: str, fontes_contexto: set[str], exige_rascunho: bool = False,
                    exige_fonte: bool = True) -> VerificacaoSaida:
    """
    Reprova a resposta que fura os limites ou cita o que nao foi recuperado.

    `fontes_contexto` sao as secoes que realmente estavam no prompt; citar outra
    coisa e citacao inventada, o pior tipo de falha de explicabilidade.
    """
    problemas = []
    citadas = fontes_citadas(resposta)
    # PROT-008 e PROT-007 sao as normas do proprio assistente: citar o limite de
    # atuacao ou a regra de comunicacao de alerta e sempre legitimo.
    normas_do_assistente = {f for f in citadas if f.startswith(("PROT-008", "PROT-007", "POP-002"))}
    inventadas = sorted(citadas - fontes_contexto - normas_do_assistente)
    if contem_dose(resposta):
        problemas.append("contém dose de medicamento")
    if exige_rascunho and "rascunho" not in _norm(resposta):
        problemas.append("sugestão não marcada como rascunho para validação médica")
    if exige_fonte and not citadas:
        problemas.append("não cita a fonte da informação")
    if inventadas:
        problemas.append(f"cita fonte que não estava no contexto: {', '.join(inventadas)}")
    return VerificacaoSaida(not problemas, problemas, sorted(citadas), inventadas)
