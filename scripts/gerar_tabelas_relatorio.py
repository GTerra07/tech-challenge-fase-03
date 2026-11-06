"""
Injeta no relatorio tecnico e no README as tabelas geradas a partir dos arquivos
de resultado — nenhum numero do relatorio e digitado a mao (licao da Fase 2).

Marcadores: <!-- tabela:NOME --> ... <!-- /tabela:NOME -->

Uso:
    python scripts/gerar_tabelas_relatorio.py
"""

import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
OUT = RAIZ / "outputs"


def _ler(caminho):
    p = OUT / caminho
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def _n(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def tabela_curadoria():
    md = (OUT / "dados" / "relatorio_curadoria.md").read_text(encoding="utf-8")
    return md.split("## Filtros", 1)[1].strip().replace("## Divisão", "**Divisão**").replace(
        "## Tarefas no treino", "**Tarefas no treino**")


def tabela_treino():
    h = _ler("fine_tuning/historico_treino.json")
    if not h:
        return "_(aguardando o treino no Colab)_"
    log = h["log_history"]
    perdas_t = [x["loss"] for x in log if "loss" in x]
    perdas_v = [x["eval_loss"] for x in log if "eval_loss" in x]
    m = h["metricas_treino"]
    return "\n".join([
        "| Item | Valor |", "|---|---|",
        f"| Modelo base | `{h['modelo_base']}` (4 bits) |",
        f"| GPU | {h['gpu']} |",
        f"| Exemplos de treino / validação | {h['n_treino']} / {h['n_validacao']} |",
        f"| Passos (1 época, batch efetivo 8) | {max(x.get('step', 0) for x in log)} |",
        f"| Tempo de treino | {_n(h['minutos_treino'], 1)} min |",
        f"| Perda de treino: primeiro registro → último | {_n(perdas_t[0])} → {_n(perdas_t[-1])} |",
        f"| Perda de validação: primeira → última | {_n(perdas_v[0])} → {_n(perdas_v[-1])} |" if perdas_v else "",
        f"| Perda média de treino da época | {_n(m.get('train_loss', float('nan')))} |",
    ])


ROTULOS = {
    "cita_fonte_correta": "cita a fonte correta", "sem_dose": "não indica dose",
    "admite_sem_contexto": "admite que o contexto não responde", "recusou": "recusa o pedido",
    "cobertura_alertas_criticos": "cobre os alertas críticos", "marca_rascunho": "marca como rascunho",
    "estrutura_mod001": "segue a estrutura MOD-001", "birads_correto": "BI-RADS correto",
    "preserva_itens": "preserva os itens da prescrição",
}
TAREFAS = {"pergunta": "Pergunta sobre protocolo", "sem_contexto": "Pergunta sem resposta no contexto",
           "recusa": "Pedido de prescrição (redações inéditas)", "caso": "Caso de paciente",
           "laudo": "Laudo", "receita": "Receita"}


def tabela_modelo():
    m = _ler("fine_tuning/metricas_avaliacao.json")
    if not m:
        return "_(aguardando a avaliação no Colab)_"
    b, a = m["base"], m["ajustado"]
    linhas = ["| Métrica | LLaMA 3.1 8B base | Ajustado | Variação |", "|---|---:|---:|---:|",
              f"| Perplexidade nas respostas de referência (↓) | {_n(m['perplexidade']['base'], 2)} | "
              f"{_n(m['perplexidade']['ajustado'], 2)} | {_pct(m['perplexidade']['ajustado'] / m['perplexidade']['base'] - 1)} |",
              f"| PubMedQA — acurácia (n={b['pubmedqa']['n']}) | {_n(b['pubmedqa']['acuracia'])} | "
              f"{_n(a['pubmedqa']['acuracia'])} | {_n(a['pubmedqa']['acuracia'] - b['pubmedqa']['acuracia'], 3)} |",
              f"| PubMedQA — F1 macro | {_n(b['pubmedqa']['f1_macro'])} | {_n(a['pubmedqa']['f1_macro'])} | "
              f"{_n(a['pubmedqa']['f1_macro'] - b['pubmedqa']['f1_macro'], 3)} |"]
    for tarefa, sim in a["similaridade"].items():
        sb = b["similaridade"][tarefa]
        linhas.append(f"| {TAREFAS.get(tarefa, tarefa)} — ROUGE-L (n={sim['n']}) | {_n(sb['rouge_l'])} | "
                      f"{_n(sim['rouge_l'])} | {_n(sim['rouge_l'] - sb['rouge_l'], 3)} |")
    for tarefa, sim in a["similaridade"].items():
        sb = b["similaridade"][tarefa]
        linhas.append(f"| {TAREFAS.get(tarefa, tarefa)} — BLEU | {_n(sb['bleu'], 1)} | {_n(sim['bleu'], 1)} | "
                      f"{_n(sim['bleu'] - sb['bleu'], 1)} |")
    linhas += ["", "**Comportamento exigido pelo hospital** (fração dos exemplos de teste que cumprem):", "",
               "| Tarefa | Verificação | Base | Ajustado |", "|---|---|---:|---:|"]
    for tarefa, v in a["comportamento"].items():
        for chave, valor in v.items():
            if chave == "n":
                continue
            linhas.append(f"| {TAREFAS.get(tarefa, tarefa)} (n={v['n']}) | {ROTULOS.get(chave, chave)} | "
                          f"{_pct(b['comportamento'][tarefa][chave])} | **{_pct(valor)}** |")
    return "\n".join(linhas)


def tabela_sistema():
    s = _ler("avaliacao/assistente.json")
    if not s:
        return "_(rodar python -m src.avaliacao.avaliar_assistente)_"
    bu, ba = s["busca"], s["barreira_entrada"]
    c = bu["confianca"]
    return "\n".join([
        f"**Busca nos protocolos** — {bu['n_perguntas']} perguntas de teste (redações nunca vistas no treino):", "",
        "| Estratégia | recall@4 | MRR |", "|---|---:|---:|",
        *[f"| {nome} | {_n(bu[nome]['recall@4'])} | {_n(bu[nome]['mrr'])} |" for nome in ("vetorial", "bm25", "hibrida")],
        "",
        f"Limiar de confiança {_n(c['limiar'], 2)}: mediana {_n(c['perguntas_clinicas_mediana'], 2)} nas perguntas "
        f"clínicas ({c['perguntas_clinicas_abaixo_do_limiar']} abaixo do limiar) contra "
        f"{_n(c['fora_do_escopo_mediana'], 2)} nas fora do escopo ({c['fora_do_escopo_desviadas']} desviadas para "
        "\"sem base\").", "",
        "**Barreira de entrada** (limites de atuação):", "",
        "| Verificação | Resultado |", "|---|---:|",
        f"| Pedidos indevidos barrados | {ba['pedidos_indevidos_barrados']} |",
        f"| … das redações separadas para teste | {ba['redacoes_de_teste_barradas']} |",
        f"| Perguntas legítimas barradas (falso positivo) | {ba['perguntas_legitimas_barradas_falso_positivo']} |",
    ])


def tabela_ponta_a_ponta():
    s = _ler("avaliacao/assistente_com_modelo.json")
    if not s:
        return "_(rodar python -m src.avaliacao.avaliar_assistente --com-modelo)_"
    p, f = s["ponta_a_ponta"]["perguntas"], s["ponta_a_ponta"]["fluxo_paciente"]
    return "\n".join([
        "| Fluxo | Métrica | Resultado |", "|---|---|---:|",
        f"| Perguntas clínicas (n={p['n']}) | cita a fonte correta | {_pct(p['cita_fonte_correta'])} |",
        f"| | aprovada na barreira de saída | {_pct(p['aprovada_na_barreira'])} |",
        f"| | latência mediana | {_n(p['latencia_mediana_s'], 1)} s |",
        f"| Fluxo do paciente (n={f['n']}) | sugestão da LLM aprovada | {f['sugestao_da_llm_aprovada']}/{f['n']} |",
        f"| | aprovada já na 1ª tentativa | {f['aprovada_na_1a_tentativa']}/{f['n']} |",
        f"| | fallback para as regras | {f['fallback_regras']}/{f['n']} |",
        f"| | latência mediana | {_n(f['latencia_mediana_s'], 1)} s |",
    ])


TABELAS = {"curadoria": tabela_curadoria, "treino": tabela_treino, "modelo": tabela_modelo,
           "sistema": tabela_sistema, "ponta_a_ponta": tabela_ponta_a_ponta}


def main():
    for arquivo in (RAIZ / "reports" / "relatorio-tecnico.md", RAIZ / "README.md"):
        if not arquivo.exists():
            continue
        texto = arquivo.read_text(encoding="utf-8")
        for nome, fn in TABELAS.items():
            marcador = rf"<!-- tabela:{nome} -->.*?<!-- /tabela:{nome} -->"
            if re.search(marcador, texto, flags=re.DOTALL):
                bloco = f"<!-- tabela:{nome} -->\n{fn()}\n<!-- /tabela:{nome} -->"
                texto = re.sub(marcador, lambda _: bloco, texto, flags=re.DOTALL)
        arquivo.write_text(texto, encoding="utf-8")
        print("atualizado:", arquivo.relative_to(RAIZ))


if __name__ == "__main__":
    sys.exit(main())
