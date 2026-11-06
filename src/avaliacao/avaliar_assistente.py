"""
Avaliacao do assistente como sistema (complementa a avaliacao do modelo no Colab).

O notebook do Colab mede o MODELO (base x ajustado). Aqui medimos as pecas em
volta dele, que tambem decidem se a resposta sai certa:

1. Busca nos protocolos — recall@4 e MRR nas perguntas de teste (redacoes nunca
   vistas no treino), comparando so vetorial, so BM25 e hibrida; e a taxa de
   perguntas fora do escopo corretamente desviadas para "sem base".
2. Barreira de entrada — pedidos indevidos barrados (inclusive as redacoes
   separadas para teste) e perguntas legitimas que passam (falso positivo).
3. Ponta a ponta com a LLM ajustada no Ollama (`--com-modelo`): os dois grafos
   rodando de verdade, com taxa de aprovacao na barreira de saida, fonte correta,
   uso do fallback e latencia.

Uso:
    python -m src.avaliacao.avaliar_assistente               # 1 e 2 (sem modelo)
    python -m src.avaliacao.avaliar_assistente --com-modelo  # 1, 2 e 3
"""

from __future__ import annotations

import argparse
import json
import time
from statistics import mean, median

from src.config import PASTA_BRUTO, PASTA_OUTPUTS, PASTA_PROCESSADO
from src.dados import conhecimento_interno as ci

FORA_DO_ESCOPO = [
    "Qual a capital da França?", "Quem ganhou a Copa do Mundo de 2002?", "Me recomenda um filme de ação.",
    "Como faço um bolo de cenoura?", "Qual a cotação do dólar hoje?", "Explique a teoria da relatividade.",
    "Qual o melhor time de futebol do Brasil?", "Como configurar um roteador wi-fi?",
]


def _perguntas_de_teste() -> list[tuple[str, str]]:
    """(pergunta, fonte esperada) das redacoes de FAQ separadas para teste."""
    with open(PASTA_PROCESSADO / "teste.jsonl", encoding="utf-8") as f:
        itens = [json.loads(l) for l in f]
    saida = []
    for ex in itens:
        if ex["tarefa"] == "pergunta":
            pergunta = ex["input"].rsplit("PERGUNTA:", 1)[1].strip()
            pergunta = pergunta.split("escreveu:", 1)[-1].strip()
            saida.append((pergunta, ex["fonte"]))
    return saida


def avaliar_busca(busca) -> dict:
    from src.assistente.grafo_pergunta import CONFIANCA_MINIMA
    from src.rag.indice import _tokens

    perguntas = _perguntas_de_teste()

    def ranking_denso(q):
        return [d.metadata["fonte"] for d, _ in busca.vetorial.similarity_search_with_score(q, k=4)]

    def ranking_bm25(q):
        notas = busca.bm25.get_scores(_tokens(q))
        return [busca.documentos[i].metadata["fonte"] for i in sorted(range(len(notas)), key=lambda i: -notas[i])[:4]]

    def ranking_hibrido(q):
        return [t.fonte for t in busca.buscar(q)[0]]

    resultado = {}
    for nome, fn in (("vetorial", ranking_denso), ("bm25", ranking_bm25), ("hibrida", ranking_hibrido)):
        acertos, rr = 0, []
        for q, alvo in perguntas:
            r = fn(q)
            acertos += alvo in r
            rr.append(1 / (r.index(alvo) + 1) if alvo in r else 0.0)
        resultado[nome] = {"recall@4": round(acertos / len(perguntas), 3), "mrr": round(mean(rr), 3)}
    conf_dentro = [busca.buscar(q)[1] for q, _ in perguntas]
    conf_fora = [busca.buscar(q)[1] for q in FORA_DO_ESCOPO]
    resultado["confianca"] = {
        "limiar": CONFIANCA_MINIMA,
        "perguntas_clinicas_mediana": round(median(conf_dentro), 3),
        "perguntas_clinicas_abaixo_do_limiar": sum(c < CONFIANCA_MINIMA for c in conf_dentro),
        "fora_do_escopo_mediana": round(median(conf_fora), 3),
        "fora_do_escopo_desviadas": f"{sum(c < CONFIANCA_MINIMA for c in conf_fora)}/{len(conf_fora)}",
    }
    resultado["n_perguntas"] = len(perguntas)
    return resultado


def avaliar_barreira() -> dict:
    from src.dados.gerador_interno import POR_TEMPLATE_RECUSA
    from src.dados.preparar_dataset import TEMPLATES_RECUSA_TESTE
    from src.seguranca.limites import triagem_entrada

    with open(PASTA_BRUTO / "pedidos_indevidos.jsonl", encoding="utf-8") as f:
        pedidos = [json.loads(l) for l in f]
    teste = [p for p in pedidos if int(p["id"].split("-")[1]) // POR_TEMPLATE_RECUSA in TEMPLATES_RECUSA_TESTE]
    legitimas = [q for item in ci.FAQ for q in item.perguntas]
    barrados = [p for p in pedidos if not triagem_entrada(p["texto_bruto"]).permitido]
    return {
        "pedidos_indevidos_barrados": f"{len(barrados)}/{len(pedidos)}",
        "redacoes_de_teste_barradas": f"{sum(not triagem_entrada(p['texto_bruto']).permitido for p in teste)}/{len(teste)}",
        "perguntas_legitimas_barradas_falso_positivo": f"{sum(not triagem_entrada(q).permitido for q in legitimas)}/{len(legitimas)}",
        "nao_barrados": [p["texto_bruto"] for p in pedidos if triagem_entrada(p["texto_bruto"]).permitido][:5],
    }


def avaliar_ponta_a_ponta(busca, n_perguntas: int = 20, modelo: str | None = None) -> dict:
    from langgraph.types import Command

    from src.assistente.grafo_paciente import criar_grafo_paciente
    from src.assistente.grafo_pergunta import criar_grafo_pergunta
    from src.assistente.llm import Redator, criar_llm_ajustada
    from src.avaliacao.metricas import cita_fonte, situacao
    from src.dados.perguntas_paciente import PERGUNTAS, resposta
    from src.prontuario import banco
    from src.seguranca.auditoria import Auditoria

    banco.criar_banco()
    arquivo_log = PASTA_OUTPUTS / "avaliacao" / f"auditoria_avaliacao{'_' + modelo if modelo else ''}.jsonl"
    arquivo_log.unlink(missing_ok=True)   # um log por rodada: execucoes repetidas tem os mesmos ids
    aud = Auditoria(arquivo_log)
    redator = Redator(criar_llm_ajustada(modelo) if modelo else None, auditoria=aud)

    perguntas = _perguntas_de_teste()[:n_perguntas]
    grafo_q = criar_grafo_pergunta(redator, busca, aud)
    linhas_q = []
    for i, (q, alvo) in enumerate(perguntas):
        t = time.perf_counter()
        e = grafo_q.invoke({"execucao": f"aval-q{i}", "pergunta": q}, {"configurable": {"thread_id": f"aval-q{i}"}})
        linhas_q.append({"pergunta": q, "fonte_esperada": alvo, "cita_fonte_correta": cita_fonte(e["resposta"], alvo),
                         "aprovada_na_barreira": e.get("verificacao", {}).get("aprovada", False),
                         "tentativas": e.get("tentativas", 0), "segundos": round(time.perf_counter() - t, 1),
                         "resposta": e["resposta"]})

    # Perguntas sobre os pacientes de demonstracao: a resposta leu o prontuario certo?
    # Inclui o cenario em que o v1 falhou: o laboratorio libera um resultado e a
    # pergunta e refeita.
    def perguntar_paciente(pid, tipo, exame, rotulo):
        q = PERGUNTAS[tipo][0].format(pid=pid, exame=exame)
        ex = f"aval-{rotulo}"
        e = grafo_q.invoke({"execucao": ex, "pergunta": q}, {"configurable": {"thread_id": ex}})
        ref = resposta(banco.carregar_paciente(pid), tipo, exame)
        trilha = " ".join(e.get("trilha", []))
        return {"pergunta": q, "referencia": ref, "resposta": e["resposta"],
                "situacao_correta": situacao(e["resposta"], tipo) == situacao(ref, tipo),
                "aprovada_na_barreira": e.get("verificacao", {}).get("aprovada", False),
                # o modelo acertou sozinho, sem a correcao pela verificacao factual?
                "certa_na_1a_tentativa": e.get("tentativas", 0) == 1 and e.get("verificacao", {}).get("aprovada", False),
                "resposta_montada_do_prontuario": "responder_pelo_prontuario" in trilha}

    linhas_pp = []
    for p in banco.listar_pacientes():
        registro = banco.carregar_paciente(p["id"])
        linhas_pp.append(perguntar_paciente(p["id"], "pendencias", None, f"pp-{p['id']}-pend"))
        linhas_pp.append(perguntar_paciente(p["id"], "criticos", None, f"pp-{p['id']}-crit"))
        primeiro = registro["exames"][0]["exame"]
        linhas_pp.append(perguntar_paciente(p["id"], "exame", primeiro, f"pp-{p['id']}-exame"))
    antes = perguntar_paciente("P-0001", "exame", "hemocultura", "pp-atualizacao-antes")
    banco.registrar_resultado("P-0001", "hemocultura", "positiva para bacilo gram-negativo")
    depois = perguntar_paciente("P-0001", "exame", "hemocultura", "pp-atualizacao-depois")
    banco.criar_banco()

    grafo_p = criar_grafo_paciente(redator, busca, aud)
    linhas_p = []
    for p in banco.listar_pacientes():
        cfg = {"configurable": {"thread_id": f"aval-{p['id']}"}}
        t = time.perf_counter()
        grafo_p.invoke({"execucao": f"aval-{p['id']}", "paciente_id": p["id"]}, cfg)
        e = grafo_p.invoke(Command(resume={"decisao": "aprovada", "medico": "avaliacao-automatica"}), cfg)
        linhas_p.append({"paciente": p["id"], "alertas": len(e.get("alertas", [])),
                         "origem_sugestao": e["origem_sugestao"], "tentativas": e["tentativas"],
                         "problemas_1a_tentativa": [ev["problemas"] for ev in aud.eventos(f"aval-{p['id']}")
                                                   if ev["evento"] == "verificacao_saida"][:1],
                         "segundos": round(time.perf_counter() - t, 1), "sugestao": e["sugestao"]})
    return {
        "perguntas": {"n": len(linhas_q),
                      "cita_fonte_correta": round(mean(l["cita_fonte_correta"] for l in linhas_q), 3),
                      "aprovada_na_barreira": round(mean(l["aprovada_na_barreira"] for l in linhas_q), 3),
                      "latencia_mediana_s": median(l["segundos"] for l in linhas_q)},
        "fluxo_paciente": {"n": len(linhas_p),
                           "sugestao_da_llm_aprovada": sum(l["origem_sugestao"] == "llm_ajustada" for l in linhas_p),
                           "fallback_regras": sum(l["origem_sugestao"] != "llm_ajustada" for l in linhas_p),
                           "aprovada_na_1a_tentativa": sum(l["tentativas"] == 1 and l["origem_sugestao"] == "llm_ajustada"
                                                           for l in linhas_p),
                           "latencia_mediana_s": median(l["segundos"] for l in linhas_p)},
        "perguntas_paciente": {"n": len(linhas_pp),
                               "situacao_correta": round(mean(l["situacao_correta"] for l in linhas_pp), 3),
                               "aprovada_na_barreira": round(mean(l["aprovada_na_barreira"] for l in linhas_pp), 3),
                               "certa_na_1a_tentativa": round(mean(l["certa_na_1a_tentativa"] for l in linhas_pp), 3),
                               "montadas_do_prontuario": sum(l["resposta_montada_do_prontuario"] for l in linhas_pp),
                               "resultado_liberado_antes_correto": antes["situacao_correta"],
                               "resultado_liberado_depois_correto": depois["situacao_correta"],
                               "resultado_liberado_depois_na_1a_tentativa": depois["certa_na_1a_tentativa"]},
        "detalhe_perguntas": linhas_q, "detalhe_pacientes": linhas_p,
        "detalhe_perguntas_paciente": linhas_pp + [antes, depois],
    }


def main():
    from src.rag.indice import BuscaProtocolos

    parser = argparse.ArgumentParser()
    parser.add_argument("--com-modelo", action="store_true", help="roda os grafos com a LLM ajustada no Ollama")
    parser.add_argument("--modelo", default=None, help="nome do modelo no Ollama (padrao: MODELO_AJUSTADO do .env)")
    args = parser.parse_args()
    busca = BuscaProtocolos()
    resultado = {"busca": avaliar_busca(busca), "barreira_entrada": avaliar_barreira()}
    if args.com_modelo:
        resultado["ponta_a_ponta"] = avaliar_ponta_a_ponta(busca, modelo=args.modelo)
    destino = PASTA_OUTPUTS / "avaliacao"
    destino.mkdir(parents=True, exist_ok=True)
    nome = (f"assistente_com_modelo{'_' + args.modelo if args.modelo else ''}.json" if args.com_modelo
            else "assistente.json")
    (destino / nome).write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    resumo = {k: v for k, v in resultado.items()}
    if "ponta_a_ponta" in resumo:
        resumo["ponta_a_ponta"] = {k: v for k, v in resultado["ponta_a_ponta"].items() if not k.startswith("detalhe")}
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
