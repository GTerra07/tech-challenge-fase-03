"""
Perguntas clinicas contextualizadas (LangGraph + LangChain).

Responde duvidas de medicos sobre protocolos internos e, quando a pergunta cita
um paciente (P-0000), contextualiza com o prontuario atualizado.

    START -> triagem_entrada --(pedido proibido / injecao)--> recusar -> END
                 |
           identificar_paciente --(cita paciente)--> consultar_paciente
                 |                                        |
           buscar_protocolos <----------------------------+
                 |--(confianca baixa)--> sem_base -> END
                 |
           responder (LLM ajustada) -> verificar_resposta --(fonte inventada, 1a vez)--> responder
                 |
                 END

A recusa na triagem e deterministica e nao chama o modelo: um pedido de
prescricao nunca chega a LLM. A checagem de saida compara as fontes citadas com
as que estavam no contexto (explainability verificavel, nao declarada).
"""

from __future__ import annotations

import re

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from src.assistente.estado import EstadoPergunta
from src.assistente.llm import Redator
from src.dados.conhecimento_interno import RESPOSTA_RECUSA, RESPOSTA_SEM_CONTEXTO
from src.fine_tuning import formato
from src.prontuario import ferramentas
from src.rag.indice import BuscaProtocolos, contexto
from src.seguranca.anonimizacao import anonimizar
from src.seguranca.auditoria import Auditoria, auditar_no
from src.seguranca.limites import triagem_entrada as aplicar_triagem, verificar_saida

# Abaixo disto a melhor secao recuperada nao trata do assunto (medido: perguntas
# clinicas ficaram >= 0,35; "capital da Franca" ficou em 0,15).
CONFIANCA_MINIMA = 0.25
_RE_PACIENTE = re.compile(r"\bP-\d{4}\b", re.IGNORECASE)
FONTE_PRONTUARIO = "PRONTUÁRIO"


def criar_grafo_pergunta(redator: Redator, busca: BuscaProtocolos, auditoria: Auditoria | None = None,
                         checkpointer=None):
    aud = auditoria or redator.auditoria
    auditado = auditar_no(lambda e: (aud, e["execucao"]))

    @auditado
    def triagem_entrada(e: EstadoPergunta) -> dict:
        t = aplicar_triagem(e["pergunta"])
        if not t.permitido:
            aud.registrar("bloqueio", e["execucao"], categoria=t.categoria, motivo=t.motivo,
                          pergunta=anonimizar(e["pergunta"]))
            return {"bloqueio": {"categoria": t.categoria, "motivo": t.motivo},
                    "trilha": [f"triagem_entrada: BLOQUEADA ({t.motivo})"]}
        return {"tentativas": 0, "trilha": ["triagem_entrada: permitida"]}

    @auditado
    def recusar(e: EstadoPergunta) -> dict:
        return {"resposta": RESPOSTA_RECUSA, "fontes_contexto": ["PROT-008 §2", "PROT-008 §3"],
                "trilha": ["recusar: resposta padrão do PROT-008, sem chamar o modelo"]}

    @auditado
    def identificar_paciente(e: EstadoPergunta) -> dict:
        m = _RE_PACIENTE.search(e["pergunta"])
        pid = e.get("paciente_id") or (m.group(0).upper() if m else "")
        return {"paciente_id": pid,
                "trilha": [f"identificar_paciente: {pid or 'pergunta geral, sem paciente'}"]}

    @auditado
    def consultar_paciente(e: EstadoPergunta) -> dict:
        try:
            dados = ferramentas.consultar_prontuario.invoke({"paciente_id": e["paciente_id"]})
            dados += "\n" + ferramentas.consultar_pendencias.invoke({"paciente_id": e["paciente_id"]})
        except (ferramentas.PacienteNaoEncontrado, ValueError):
            return {"dados_paciente": "", "trilha": [f"consultar_paciente: {e['paciente_id']} não encontrado"]}
        return {"dados_paciente": dados,
                "trilha": [f"consultar_paciente: prontuário atualizado de {e['paciente_id']} (anonimizado)"]}

    @auditado
    def buscar_protocolos(e: EstadoPergunta) -> dict:
        trechos, confianca = busca.buscar(e["pergunta"])
        ctx = contexto(trechos)
        fontes = [t.fonte for t in trechos]
        if e.get("dados_paciente"):
            ctx = f"[{FONTE_PRONTUARIO} {e['paciente_id']}] Prontuário atualizado\n{e['dados_paciente']}\n\n{ctx}"
            fontes = [f"{FONTE_PRONTUARIO} {e['paciente_id']}"] + fontes
        aud.registrar("busca", e["execucao"], fontes=fontes, confianca=confianca,
                      scores={t.fonte: t.score for t in trechos})
        return {"contexto": ctx, "fontes_contexto": fontes, "confianca": confianca,
                "trilha": [f"buscar_protocolos: {', '.join(fontes)} (confiança {confianca:.2f})"]}

    @auditado
    def sem_base(e: EstadoPergunta) -> dict:
        return {"resposta": RESPOSTA_SEM_CONTEXTO,
                "trilha": [f"sem_base: confiança {e['confianca']:.2f} < {CONFIANCA_MINIMA}; modelo não acionado"]}

    @auditado
    def responder(e: EstadoPergunta) -> dict:
        entrada = formato.entrada_pergunta(e["contexto"], e["pergunta"])
        entrada = formato.com_revisao(entrada, e.get("verificacao", {}).get("problemas", []))
        resposta = redator.redigir(formato.INSTRUCAO_PERGUNTA, entrada, e["execucao"], "pergunta_clinica")
        return {"resposta": resposta, "tentativas": e.get("tentativas", 0) + 1,
                "trilha": [f"responder: LLM ajustada (tentativa {e.get('tentativas', 0) + 1})"]}

    @auditado
    def verificar_resposta(e: EstadoPergunta) -> dict:
        fontes_doc = {f for f in e["fontes_contexto"] if not f.startswith(FONTE_PRONTUARIO)}
        v = verificar_saida(e["resposta"], fontes_doc)
        verificacao = {"aprovada": v.aprovada, "problemas": v.problemas, "fontes_citadas": v.fontes_citadas}
        aud.registrar("verificacao_saida", e["execucao"], **verificacao)
        return {"verificacao": verificacao,
                "trilha": [f"verificar_resposta: {'aprovada' if v.aprovada else 'REPROVADA — ' + '; '.join(v.problemas)}"]}

    @auditado
    def anotar_ressalva(e: EstadoPergunta) -> dict:
        """Segunda reprovacao: entrega a resposta com a ressalva explicita, nunca em silencio."""
        aviso = ("\n\n⚠️ Verificação automática: " + "; ".join(e["verificacao"]["problemas"]) +
                 ". Confirme no protocolo antes de usar esta resposta.")
        return {"resposta": e["resposta"] + aviso, "trilha": ["anotar_ressalva: resposta entregue com aviso"]}

    def depois_da_triagem(e):
        return "bloqueada" if e.get("bloqueio") else "permitida"

    def depois_de_identificar(e):
        return "com_paciente" if e.get("paciente_id") else "sem_paciente"

    def depois_da_busca(e):
        return "sem_base" if e["confianca"] < CONFIANCA_MINIMA and not e.get("dados_paciente") else "responder"

    def depois_da_verificacao(e):
        if e["verificacao"]["aprovada"]:
            return "ok"
        return "tentar_de_novo" if e["tentativas"] < 2 else "ressalva"

    g = StateGraph(EstadoPergunta)
    g.add_node("triagem_entrada", triagem_entrada)
    g.add_node("recusar", recusar)
    g.add_node("identificar_paciente", identificar_paciente)
    g.add_node("consultar_paciente", consultar_paciente)
    g.add_node("buscar_protocolos", buscar_protocolos)
    g.add_node("sem_base", sem_base)
    g.add_node("responder", responder, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
    g.add_node("verificar_resposta", verificar_resposta)
    g.add_node("anotar_ressalva", anotar_ressalva)

    g.add_edge(START, "triagem_entrada")
    g.add_conditional_edges("triagem_entrada", depois_da_triagem,
                            {"bloqueada": "recusar", "permitida": "identificar_paciente"})
    g.add_edge("recusar", END)
    g.add_conditional_edges("identificar_paciente", depois_de_identificar,
                            {"com_paciente": "consultar_paciente", "sem_paciente": "buscar_protocolos"})
    g.add_edge("consultar_paciente", "buscar_protocolos")
    g.add_conditional_edges("buscar_protocolos", depois_da_busca, {"sem_base": "sem_base", "responder": "responder"})
    g.add_edge("sem_base", END)
    g.add_edge("responder", "verificar_resposta")
    g.add_conditional_edges("verificar_resposta", depois_da_verificacao,
                            {"ok": END, "tentar_de_novo": "responder", "ressalva": "anotar_ressalva"})
    g.add_edge("anotar_ressalva", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
