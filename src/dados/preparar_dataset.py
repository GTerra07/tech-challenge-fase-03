"""
Pipeline completo de preparacao do dataset de fine-tuning.

    exportacoes brutas internas (com PII falsa)  ─┐
    PubMedQA (Hugging Face)                       ├─> exemplos Alpaca
    MedQuAD  (Hugging Face)                       ─┘
        -> preprocessing (limpeza)
        -> anonimizacao (so conteudo interno; os publicos ja vem desidentificados)
        -> curadoria (filtros com contagem)
        -> divisao treino / validacao / teste
        -> data/processado/*.jsonl + outputs/dados/relatorio_curadoria.*

Uso:
    python -m src.dados.preparar_dataset
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from src.config import PASTA_BRUTO, PASTA_EXTERNO, PASTA_OUTPUTS, PASTA_PROCESSADO, SEMENTE
from src.dados import conhecimento_interno as ci
from src.dados.curadoria import Relatorio, curar
from src.dados.documentos import carregar_secoes, indice_por_fonte
from src.dados.gerador_interno import POR_TEMPLATE_RECUSA, gerar_exportacoes
from src.dados.preprocessamento import limpar_exemplo
from src.fine_tuning import formato as fmt
from src.prontuario import regras
from src.prontuario.resumo import bloco_achados, contexto_paciente, sugestao_de_referencia
from src.seguranca.anonimizacao import anonimizar

# Divisao dos publicos (PubMedQA tem 1.000 exemplos rotulados por especialistas)
PUBMEDQA_TESTE, PUBMEDQA_VALIDACAO = 200, 50
MEDQUAD_TREINO, MEDQUAD_VALIDACAO, MEDQUAD_TESTE = 400, 50, 100
# Teto do PubMedQA no treino: o foco do ajuste e o conteudo interno do hospital;
# o publico entra para conhecimento clinico geral e para a metrica de acuracia.
PUBMEDQA_TREINO_MAX = 500
# Conteudo interno separado para teste, de forma que meca generalizacao:
TEMPLATES_RECUSA_TESTE = {4, 9}     # redacoes de pedido indevido nunca vistas no treino
PARAFRASE_FAQ_TESTE = 2             # a 3a redacao de cada pergunta frequente
FRACAO_TESTE_INTERNO = 0.15
FRACAO_VALIDACAO = 0.05


def _ex(id_, origem, tarefa, instrucao, entrada, saida, id_bruto=None, **extra):
    return {"id": id_, "origem": origem, "tarefa": tarefa, "instruction": instrucao, "input": entrada,
            "output": saida, "id_bruto": id_bruto or id_, **extra}


def _ler(nome):
    with open(PASTA_BRUTO / nome, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _contexto(secoes):
    return "\n\n".join(s.como_contexto() for s in secoes)


# ---------------------------------------------------------------------------
# Conteudo interno
# ---------------------------------------------------------------------------

def exemplos_internos(rng: random.Random) -> tuple[list[dict], dict[str, list[str]]]:
    secoes = carregar_secoes()
    por_fonte = indice_por_fonte(secoes)
    exemplos, pii = [], {}

    def distratores(excluir, k):
        return rng.sample([s for s in secoes if s.fonte not in excluir], k)

    # Perguntas frequentes: com a secao certa no contexto (2 variacoes de
    # distratores) e, as vezes, SEM ela — o modelo aprende a dizer que nao sabe.
    for r in _ler("faq_medicos.jsonl"):
        pii[r["id"]] = r["pii"]
        pergunta = anonimizar(r["texto_bruto"])
        alvo = por_fonte[r["fonte"]]
        divisao = "teste" if r["id"].endswith(f"-{PARAFRASE_FAQ_TESTE}") else None
        for v in range(3 if divisao is None else 1):
            ctx = distratores({alvo.fonte}, rng.randint(1, 2)) + [alvo]
            rng.shuffle(ctx)
            exemplos.append(_ex(f"{r['id']}-v{v}", "interno", "pergunta", fmt.INSTRUCAO_PERGUNTA,
                                fmt.entrada_pergunta(_contexto(ctx), pergunta),
                                f"{r['resposta']} (Fonte: {r['fonte']})", r["id"], divisao=divisao, fonte=r["fonte"]))
        if rng.random() < 0.5:
            ctx = distratores({alvo.fonte}, 2)
            exemplos.append(_ex(f"{r['id']}-sem", "interno", "sem_contexto", fmt.INSTRUCAO_PERGUNTA,
                                fmt.entrada_pergunta(_contexto(ctx), pergunta), ci.RESPOSTA_SEM_CONTEXTO,
                                r["id"], divisao=divisao, fonte=""))

    # Pedidos fora do limite de atuacao -> recusa citando o PROT-008
    for r in _ler("pedidos_indevidos.jsonl"):
        pii[r["id"]] = r["pii"]
        template = int(r["id"].split("-")[1]) // POR_TEMPLATE_RECUSA
        ctx = distratores(set(), rng.randint(1, 2))
        if rng.random() < 0.5:
            ctx.append(por_fonte["PROT-008 §3"])
        exemplos.append(_ex(r["id"], "interno", "recusa", fmt.INSTRUCAO_PERGUNTA,
                            fmt.entrada_pergunta(_contexto(ctx), anonimizar(r["texto_bruto"])), r["resposta"],
                            divisao="teste" if template in TEMPLATES_RECUSA_TESTE else None, fonte="PROT-008 §2"))

    # Casos de pacientes -> resumo, alertas e sugestao (rascunho) a partir das regras
    for r in _ler("casos_pacientes.jsonl"):
        pii[r["id"]] = r["pii"]
        p = r["paciente"]
        fontes = {a.fonte for a in regras.alertas(p)} | {"PROT-008 §3"}
        fontes |= {f"{prot} §1" for prot in p["protocolos_ativos"] if f"{prot} §1" in por_fonte}
        ctx = [por_fonte[f] for f in sorted(fontes) if f in por_fonte]
        exemplos.append(_ex(r["id"], "interno", "caso", fmt.INSTRUCAO_CASO,
                            fmt.entrada_caso(contexto_paciente(p), anonimizar(bloco_achados(p)), _contexto(ctx)),
                            anonimizar(sugestao_de_referencia(p)), fonte=r["fonte"]))

    # Modelos de laudo e de receita
    for r in _ler("laudos.jsonl"):
        pii[r["id"]] = r["pii"]
        exemplos.append(_ex(r["id"], "interno", "laudo", fmt.INSTRUCAO_LAUDO, anonimizar(r["texto_bruto"]),
                            anonimizar(r["laudo"]), fonte=r["fonte"]))
    for r in _ler("receitas.jsonl"):
        pii[r["id"]] = r["pii"]
        exemplos.append(_ex(r["id"], "interno", "receita", fmt.INSTRUCAO_RECEITA, anonimizar(r["texto_bruto"]),
                            anonimizar(r["receita"]), fonte=r["fonte"]))
    return exemplos, pii


# ---------------------------------------------------------------------------
# Datasets publicos sugeridos no enunciado
# ---------------------------------------------------------------------------

def exemplos_pubmedqa() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir=str(PASTA_EXTERNO / "hf"))
    exemplos = []
    for r in ds:
        ctx = r["context"]
        partes = [f"{rot}: {txt}" for rot, txt in zip(ctx["labels"], ctx["contexts"])]
        entrada = "CONTEXT:\n" + "\n".join(partes) + f"\n\nQUESTION: {r['question']}"
        exemplos.append(_ex(f"pubmedqa-{r['pubid']}", "pubmedqa", "pubmedqa", fmt.INSTRUCAO_PUBMEDQA, entrada,
                            f"{r['final_decision']}. {r['long_answer']}", rotulo=r["final_decision"],
                            fonte=f"PubMed {r['pubid']}"))
    return exemplos


def exemplos_medquad(rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("lavita/MedQuAD", split="train", cache_dir=str(PASTA_EXTERNO / "hf"))
    exemplos, perguntas = [], set()
    for r in ds:
        resposta = (r["answer"] or "").strip()
        chave = (r["question"] or "").strip().lower()
        # Pre-filtro de volume: o MedQuAD tem 47 mil linhas, parte sem resposta
        # (licenca das fontes) e muitas perguntas repetidas. A curadoria abaixo
        # registra os filtros de qualidade; aqui so evitamos carregar o que nunca entraria.
        if not resposta or chave in perguntas or not (150 <= len(resposta) <= 1500):
            continue
        perguntas.add(chave)
        exemplos.append(_ex(f"medquad-{r['question_id']}", "medquad", "medquad", fmt.INSTRUCAO_MEDQUAD,
                            f"QUESTION: {r['question']}",
                            f"{resposta}\n(Source: {r['document_source']} — {r['document_url']})",
                            fonte=r["document_url"]))
    rng.shuffle(exemplos)
    return exemplos[: int((MEDQUAD_TREINO + MEDQUAD_VALIDACAO + MEDQUAD_TESTE) * 1.15)]


# ---------------------------------------------------------------------------
# Divisao e gravacao
# ---------------------------------------------------------------------------

def dividir(exemplos: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    partes = {"treino": [], "validacao": [], "teste": []}
    por_origem = defaultdict(list)
    for ex in exemplos:
        por_origem[ex["origem"]].append(ex)

    # PubMedQA: estratificado pelo rotulo (yes/no/maybe)
    por_rotulo = defaultdict(list)
    for ex in por_origem["pubmedqa"]:
        por_rotulo[ex["rotulo"]].append(ex)
    n = len(por_origem["pubmedqa"])
    for lista in por_rotulo.values():
        rng.shuffle(lista)
        k_teste = round(len(lista) * PUBMEDQA_TESTE / n)
        k_val = round(len(lista) * PUBMEDQA_VALIDACAO / n)
        partes["teste"] += lista[:k_teste]
        partes["validacao"] += lista[k_teste:k_teste + k_val]
        partes["treino"] += lista[k_teste + k_val:][: round(len(lista) * PUBMEDQA_TREINO_MAX / n)]

    mq = por_origem["medquad"]
    partes["teste"] += mq[:MEDQUAD_TESTE]
    partes["validacao"] += mq[MEDQUAD_TESTE:MEDQUAD_TESTE + MEDQUAD_VALIDACAO]
    partes["treino"] += mq[MEDQUAD_TESTE + MEDQUAD_VALIDACAO:MEDQUAD_TESTE + MEDQUAD_VALIDACAO + MEDQUAD_TREINO]

    # Interno: regras fixas (parafrase e redacoes de recusa nunca vistas) + sorteio
    # por registro bruto, para variacoes do mesmo registro nao cairem em lados opostos.
    sorteio = {}
    for ex in por_origem["interno"]:
        destino = ex.get("divisao")
        if destino is None:
            if ex["id_bruto"] not in sorteio:
                r = rng.random()
                sorteio[ex["id_bruto"]] = ("teste" if r < FRACAO_TESTE_INTERNO else
                                           "validacao" if r < FRACAO_TESTE_INTERNO + FRACAO_VALIDACAO else "treino")
            destino = sorteio[ex["id_bruto"]]
        partes[destino].append(ex)

    for lista in partes.values():
        rng.shuffle(lista)
    return partes


def _gravar_jsonl(caminho, exemplos):
    campos = ("id", "origem", "tarefa", "fonte", "rotulo", "instruction", "input", "output")
    with open(caminho, "w", encoding="utf-8") as f:
        for ex in exemplos:
            f.write(json.dumps({c: ex[c] for c in campos if c in ex}, ensure_ascii=False) + "\n")


def _relatorio_md(rel: dict, partes: dict[str, list[dict]]) -> str:
    origens = ["interno", "pubmedqa", "medquad"]
    linhas = ["# Relatório de curadoria do dataset", "",
              "Gerado por `python -m src.dados.preparar_dataset`. Não editar à mão.", "",
              "## Filtros", "", "| Etapa | " + " | ".join(origens) + " |", "|---|" + "---:|" * len(origens)]
    linhas.append("| Entrada | " + " | ".join(str(rel["entrada"].get(o, 0)) for o in origens) + " |")
    for motivo, cont in rel["removidos"].items():
        linhas.append(f"| Removidos: {motivo} | " + " | ".join(str(cont.get(o, 0)) for o in origens) + " |")
    linhas.append("| **Aprovados** | " + " | ".join(f"**{rel['saida'].get(o, 0)}**" for o in origens) + " |")
    linhas += ["", "## Divisão", "", "| Conjunto | Total | " + " | ".join(origens) + " |",
               "|---|---:|" + "---:|" * len(origens)]
    for nome, lista in partes.items():
        c = Counter(ex["origem"] for ex in lista)
        linhas.append(f"| {nome} | {len(lista)} | " + " | ".join(str(c.get(o, 0)) for o in origens) + " |")
    linhas += ["", "## Tarefas no treino", "", "| Tarefa | Exemplos |", "|---|---:|"]
    for tarefa, n in Counter(ex["tarefa"] for ex in partes["treino"]).most_common():
        linhas.append(f"| {tarefa} | {n} |")
    return "\n".join(linhas) + "\n"


def main():
    rng = random.Random(SEMENTE)
    print("1/5 gerando exportações brutas internas (com dado pessoal falso)...")
    for arq, n in gerar_exportacoes().items():
        print(f"     {arq}: {n}")
    print("2/5 montando exemplos (internos anonimizados + PubMedQA + MedQuAD)...")
    internos, pii = exemplos_internos(rng)
    exemplos = internos + exemplos_pubmedqa() + exemplos_medquad(rng)
    print("3/5 pré-processamento...")
    exemplos = [limpar_exemplo(ex) for ex in exemplos]
    print("4/5 curadoria...")
    rel = Relatorio()
    exemplos = curar(exemplos, pii, rel)
    print("5/5 divisão e gravação...")
    partes = dividir(exemplos, rng)
    PASTA_PROCESSADO.mkdir(parents=True, exist_ok=True)
    for nome, lista in partes.items():
        _gravar_jsonl(PASTA_PROCESSADO / f"{nome}.jsonl", lista)
    pasta_rel = PASTA_OUTPUTS / "dados"
    pasta_rel.mkdir(parents=True, exist_ok=True)
    rel_dict = rel.como_dict()
    rel_dict["divisao"] = {n: dict(Counter(ex["origem"] for ex in l)) for n, l in partes.items()}
    (pasta_rel / "relatorio_curadoria.json").write_text(json.dumps(rel_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    md = _relatorio_md(rel_dict, partes)
    (pasta_rel / "relatorio_curadoria.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
