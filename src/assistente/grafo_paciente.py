"""
Fluxo automatizado de decisao ao receber um paciente (LangGraph).

O enunciado descreve: "ao receber informacoes sobre um paciente, o sistema possa
acionar diferentes etapas, como verificar exames pendentes, sugerir tratamentos e
emitir alertas para a equipe medica". Cada etapa e um no com o mesmo nome:

    START -> receber_paciente --(nao encontrado)--> END
                 |
           verificar_exames_pendentes --(ha CRITICO/ALTO)--> emitir_alertas
                 |                                               |
           buscar_protocolos <-----------------------------------+
                 |
           sugerir_tratamento   (LLM ajustada, via LangChain)
                 |
           verificar_seguranca --(reprovada, 1a vez)--> sugerir_tratamento   (ciclo)
                 |            --(reprovada de novo)--> sugestao_por_regras
                 |                                          |
           validacao_humana  <------------------------------+   (interrupt: o medico decide)
                 |
           registrar_desfecho -> END

Decisoes:
- Os alertas sao emitidos ANTES da sugestao e da validacao humana: valor critico
  tem prazo de 30 minutos (PROT-007 §3) e nao pode esperar o medico ler a sugestao.
- Exames pendentes e alertas vem das regras deterministicas; a LLM so redige a
  sugestao em cima deles (a LLM narra, nunca inventa).
- A sugestao so vira "aceita" com a decisao registrada de um medico.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from src.assistente.estado import EstadoFluxoPaciente
from src.assistente.llm import Redator
from src.config import RAIZ
from src.fine_tuning import formato
from src.prontuario import ferramentas, regras
from src.prontuario.resumo import sugestao_de_referencia
from src.rag.indice import BuscaProtocolos, contexto
from src.seguranca.anonimizacao import anonimizar
from src.seguranca.auditoria import Auditoria, auditar_no
from src.seguranca.limites import verificar_saida

ARQUIVO_FILA_ALERTAS = RAIZ / "logs" / "fila_alertas.jsonl"
PRAZO_MIN = {regras.CRITICO: 30, regras.ALTO: 240}   # PROT-007 §3 e §5 ("mesmo turno")
MAX_TENTATIVAS = 2


def criar_grafo_paciente(redator: Redator, busca: BuscaProtocolos, auditoria: Auditoria | None = None,
                         checkpointer=None):
    aud = auditoria or redator.auditoria
    auditado = auditar_no(lambda e: (aud, e["execucao"]))

    @auditado
    def receber_paciente(e: EstadoFluxoPaciente) -> dict:
        try:
            dados = ferramentas.consultar_prontuario.invoke({"paciente_id": e["paciente_id"]})
        except (ferramentas.PacienteNaoEncontrado, ValueError) as erro:
            return {"erro": f"paciente não encontrado ou identificador inválido: {erro}",
                    "trilha": [f"receber_paciente: {e['paciente_id']} não encontrado"]}
        return {"dados_paciente": dados, "tentativas": 0,
                "trilha": [f"receber_paciente: prontuário de {e['paciente_id']} carregado (anonimizado)"]}

    @auditado
    def verificar_exames_pendentes(e: EstadoFluxoPaciente) -> dict:
        achados = ferramentas.consultar_pendencias.invoke({"paciente_id": e["paciente_id"]})
        alertas = [{"nivel": a.nivel, "tipo": a.tipo, "exame": a.exame, "descricao": a.descricao,
                    "fonte": a.fonte} for a in ferramentas.alertas_do_paciente(e["paciente_id"])]
        n_crit = sum(a["nivel"] == regras.CRITICO for a in alertas)
        return {"achados": achados, "alertas": alertas,
                "trilha": [f"verificar_exames_pendentes: {len(alertas)} alerta(s), {n_crit} crítico(s)"]}

    @auditado
    def emitir_alertas(e: EstadoFluxoPaciente) -> dict:
        """Publica na fila de alertas da equipe (integracao simulada com o sistema de chamados)."""
        ARQUIVO_FILA_ALERTAS.parent.mkdir(parents=True, exist_ok=True)
        emitidos = []
        agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for a in e["alertas"]:
            alerta = {"alerta_id": uuid.uuid4().hex[:10], "emitido_em": agora, "execucao": e["execucao"],
                      "paciente_id": e["paciente_id"], "nivel": a["nivel"], "exame": a["exame"], "descricao": anonimizar(a["descricao"]),
                      "fonte": a["fonte"], "destinatario": "médico responsável e enfermagem do leito",
                      "prazo_comunicacao_min": PRAZO_MIN[a["nivel"]],
                      "registro_exigido": "PROT-007 §4 (horário, quem comunicou, quem recebeu, leitura de volta)"}
            with open(ARQUIVO_FILA_ALERTAS, "a", encoding="utf-8") as f:
                f.write(json.dumps(alerta, ensure_ascii=False) + "\n")
            aud.registrar("alerta", e["execucao"], **{k: v for k, v in alerta.items() if k != "execucao"})
            emitidos.append(alerta)
        return {"alertas_emitidos": emitidos,
                "trilha": [f"emitir_alertas: {len(emitidos)} alerta(s) enviados à equipe"]}

    @auditado
    def buscar_protocolos(e: EstadoFluxoPaciente) -> dict:
        # 1) toda secao que os alertas e as acoes previstas citam (a mesma
        #    funcao que monta o contexto do treino); 2) busca semantica pelo
        #    diagnostico, para contexto adicional.
        registro = ferramentas.carregar_para_regras(e["paciente_id"])
        diretos = busca.por_fontes(regras.fontes_para_contexto(registro))
        diagnostico = next((l.split(":", 1)[1].strip() for l in e["dados_paciente"].splitlines()
                            if l.startswith("Diagnóstico")), "")
        semanticos, confianca = busca.buscar(f"{diagnostico} conduta exames prazos", k=3)
        vistos, trechos = set(), []
        for t in diretos + semanticos:
            if t.fonte not in vistos and len(trechos) < 9:
                vistos.add(t.fonte)
                trechos.append(t)
        return {"contexto": contexto(trechos), "fontes_contexto": [t.fonte for t in trechos],
                "confianca": confianca,
                "trilha": [f"buscar_protocolos: {len(trechos)} trechos ({', '.join(t.fonte for t in trechos)})"]}

    @auditado
    def sugerir_tratamento(e: EstadoFluxoPaciente) -> dict:
        entrada = formato.entrada_caso(e["dados_paciente"], e["achados"], e["contexto"])
        entrada = formato.com_revisao(entrada, e.get("verificacao", {}).get("problemas", []))
        sugestao = redator.redigir(formato.INSTRUCAO_CASO, entrada, e["execucao"], "sugerir_tratamento")
        return {"sugestao": sugestao, "origem_sugestao": "llm_ajustada", "tentativas": e.get("tentativas", 0) + 1,
                "trilha": [f"sugerir_tratamento: rascunho gerado pela LLM ajustada (tentativa {e.get('tentativas', 0) + 1})"]}

    @auditado
    def verificar_seguranca(e: EstadoFluxoPaciente) -> dict:
        v = verificar_saida(e["sugestao"], set(e["fontes_contexto"]), exige_rascunho=True)
        # A sugestao tem que cobrir todo alerta critico calculado pelas regras.
        faltando = [a["exame"] for a in e.get("alertas", [])
                    if a["nivel"] == regras.CRITICO and a["exame"].lower() not in e["sugestao"].lower()]
        problemas = v.problemas + ([f"não menciona alerta crítico: {', '.join(faltando)}"] if faltando else [])
        verificacao = {"aprovada": not problemas, "problemas": problemas, "fontes_citadas": v.fontes_citadas}
        aud.registrar("verificacao_saida", e["execucao"], **verificacao)
        return {"verificacao": verificacao,
                "trilha": [f"verificar_seguranca: {'aprovada' if not problemas else 'REPROVADA — ' + '; '.join(problemas)}"]}

    @auditado
    def sugestao_por_regras(e: EstadoFluxoPaciente) -> dict:
        """Fallback: o modelo falhou na barreira duas vezes; a sugestao vem direto das regras."""
        sugestao = anonimizar(sugestao_de_referencia(ferramentas.carregar_para_regras(e["paciente_id"])))
        return {"sugestao": sugestao, "origem_sugestao": "regras (fallback)",
                "trilha": ["sugestao_por_regras: LLM reprovada 2x na barreira; sugestão montada pelas regras"]}

    @auditado
    def validacao_humana(e: EstadoFluxoPaciente) -> dict:
        decisao = interrupt({
            "pergunta": "Revise o rascunho e registre a decisão: aprovada, ajustada ou rejeitada.",
            "paciente_id": e["paciente_id"], "sugestao": e["sugestao"], "origem": e["origem_sugestao"],
            "alertas": e.get("alertas", []), "fontes": e["fontes_contexto"],
        })
        if not isinstance(decisao, dict) or decisao.get("decisao") not in ("aprovada", "ajustada", "rejeitada") \
                or not decisao.get("medico"):
            raise ValueError("validação humana exige 'decisao' (aprovada/ajustada/rejeitada) e 'medico'")
        decisao = {**decisao, "registrada_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        aud.registrar("validacao_humana", e["execucao"], paciente_id=e["paciente_id"], **decisao)
        return {"decisao_medica": decisao,
                "trilha": [f"validacao_humana: {decisao['decisao']} por {decisao['medico']}"]}

    @auditado
    def registrar_desfecho(e: EstadoFluxoPaciente) -> dict:
        aud.registrar("desfecho", e["execucao"], paciente_id=e["paciente_id"],
                      alertas=len(e.get("alertas_emitidos", [])), origem_sugestao=e["origem_sugestao"],
                      decisao=e["decisao_medica"]["decisao"], fontes=e["fontes_contexto"])
        return {"trilha": ["registrar_desfecho: execução concluída e auditada"]}

    # ---------------- roteadores (arestas condicionais) ----------------
    def depois_de_receber(e):
        return "fim" if e.get("erro") else "seguir"

    def depois_de_verificar(e):
        return "alertar" if e.get("alertas") else "sem_alertas"

    def depois_da_seguranca(e):
        if e["verificacao"]["aprovada"]:
            return "aprovada"
        return "tentar_de_novo" if e["tentativas"] < MAX_TENTATIVAS else "usar_regras"

    g = StateGraph(EstadoFluxoPaciente)
    llm_retry = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)
    g.add_node("receber_paciente", receber_paciente)
    g.add_node("verificar_exames_pendentes", verificar_exames_pendentes)
    g.add_node("emitir_alertas", emitir_alertas)
    g.add_node("buscar_protocolos", buscar_protocolos)
    g.add_node("sugerir_tratamento", sugerir_tratamento, retry_policy=llm_retry)
    g.add_node("verificar_seguranca", verificar_seguranca)
    g.add_node("sugestao_por_regras", sugestao_por_regras)
    g.add_node("validacao_humana", validacao_humana)
    g.add_node("registrar_desfecho", registrar_desfecho)

    g.add_edge(START, "receber_paciente")
    g.add_conditional_edges("receber_paciente", depois_de_receber,
                            {"fim": END, "seguir": "verificar_exames_pendentes"})
    g.add_conditional_edges("verificar_exames_pendentes", depois_de_verificar,
                            {"alertar": "emitir_alertas", "sem_alertas": "buscar_protocolos"})
    g.add_edge("emitir_alertas", "buscar_protocolos")
    g.add_edge("buscar_protocolos", "sugerir_tratamento")
    g.add_edge("sugerir_tratamento", "verificar_seguranca")
    g.add_conditional_edges("verificar_seguranca", depois_da_seguranca,
                            {"aprovada": "validacao_humana", "tentar_de_novo": "sugerir_tratamento",
                             "usar_regras": "sugestao_por_regras"})
    g.add_edge("sugestao_por_regras", "validacao_humana")
    g.add_edge("validacao_humana", "registrar_desfecho")
    g.add_edge("registrar_desfecho", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def nova_execucao(prefixo: str = "fluxo") -> str:
    return f"{prefixo}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
