# Assistente Médico com LLM Ajustada, LangChain e LangGraph

**Tech Challenge — Fase 3 · Pós-Tech em Engenharia de Inteligência Artificial (FIAP)**

Assistente virtual médico para um hospital universitário: um **LLaMA 3.1 8B ajustado por
fine-tuning** (Unsloth + LoRA) com protocolos, perguntas frequentes de médicos e modelos de laudo,
receita e procedimento do hospital; integrado com **LangChain** a um prontuário estruturado e a uma
busca nos protocolos com citação de fonte; e orquestrado com **LangGraph** num fluxo automatizado
que **verifica exames pendentes, sugere condutas e emite alertas à equipe médica** — sempre com
validação humana antes de qualquer conduta.

> Todos os dados do hospital (protocolos, prontuários, pacientes, médicos) são **sintéticos**,
> criados para este trabalho. Nenhum documento deve ser usado em decisão clínica real.

- Relatório técnico: [reports/relatorio-tecnico.md](reports/relatorio-tecnico.md)
- Arquitetura e diagramas: [docs/arquitetura.md](docs/arquitetura.md)
- Fine-tuning (Colab): [notebooks/01_fine_tuning_colab.ipynb](notebooks/01_fine_tuning_colab.ipynb)
- Demonstração: [notebooks/02_demonstracao_assistente.ipynb](notebooks/02_demonstracao_assistente.ipynb)

<!-- tabela:resumo -->
## O resultado em uma tabela

LLaMA 3.1 8B original × ajustado, no mesmo conjunto de teste separado antes do treino:

| Comportamento | Original | Ajustado |
|---|---:|---:|
| Recusa pedido de prescrição | 70,0% | **100,0%** |
| Laudo no modelo interno (MOD-001) | 0,0% | **100,0%** |
| Sugestão de conduta marcada como rascunho | 36,7% | **100,0%** |
| Admite quando o contexto não responde | 4,5% | **68,2%** |
| Cita a fonte correta | 95,0% | 82,5% |
| PubMedQA — acurácia | 0,55 | **0,75** |
| Perplexidade (↓) | 3,88 | **2,06** |

A queda em "cita a fonte correta" é analisada no relatório (seção 6.2): o modelo ajustado ficou mais cauteloso e às vezes diz não haver informação quando havia — o erro do lado seguro.
<!-- /tabela:resumo -->

---

## Requisitos do enunciado — onde está cada um

| Exigência | Onde |
|---|---|
| **1. Fine-tuning de LLM** (LLaMA) | `notebooks/01_fine_tuning_colab.ipynb` — LLaMA 3.1 8B 4-bit, Unsloth, LoRA r=16, `SFTTrainer` |
| … com protocolos médicos do hospital | `data/protocolos/` (9) e `data/procedimentos/` (4 POPs) |
| … com perguntas frequentes feitas por médicos | `src/dados/conhecimento_interno.py` (FAQ) + PubMedQA e MedQuAD |
| … com modelos de laudos, receitas e procedimentos | `data/modelos/` + tarefas `laudo` e `receita` do dataset |
| Preprocessing | `src/dados/preprocessamento.py` |
| Anonimização | `src/seguranca/anonimizacao.py` (ponto único) — provada em `tests/test_anonimizacao.py` |
| Curadoria | `src/dados/curadoria.py` → [relatório de curadoria](outputs/dados/relatorio_curadoria.md) |
| **2. LangChain**: pipeline que integra a LLM customizada | `src/assistente/llm.py` — `ChatPromptTemplate \| ChatOllama \| StrOutputParser` |
| Consultas em base de dados estruturada (prontuários) | `src/prontuario/banco.py` (SQLite, somente leitura) + `src/prontuario/ferramentas.py` (`@tool`) |
| Contextualizar com informações atualizadas do paciente | o prontuário é lido no momento de cada pergunta — `tests/test_grafos.py::test_pergunta_sobre_paciente_usa_prontuario_atualizado` |
| **3. Limites de atuação** (nunca prescrever sem validação humana) | `src/seguranca/limites.py` + nó `validacao_humana` (`interrupt`) |
| Logging detalhado para rastreamento e auditoria | `src/seguranca/auditoria.py` → `logs/auditoria.jsonl` |
| Explainability (fonte da informação) | `src/rag/indice.py` (fonte no metadado) + verificação de fontes citadas |
| **4. Projeto modularizado em Python, README completo** | `src/` em camadas; este arquivo |
| Entregável: fluxos do LangGraph | `src/assistente/grafo_paciente.py`, `src/assistente/grafo_pergunta.py` |
| Entregável: dataset anonimizado ou sintético | `data/processado/` (treino, validação, teste) |
| Entregável: relatório técnico | `reports/relatorio-tecnico.md` |

---

## Estrutura

```
data/
  protocolos/  procedimentos/  modelos/   documentos internos sintéticos (Markdown, seções citáveis)
  processado/                             dataset final anonimizado (treino/validação/teste .jsonl)
src/
  dados/          geração das exportações brutas, preprocessing, curadoria, montagem do dataset
  seguranca/      anonimização (ponto único), limites de atuação, auditoria
  fine_tuning/    formato Alpaca compartilhado entre treino, Ollama e inferência
  prontuario/     banco SQLite, ferramentas LangChain, regras clínicas determinísticas
  rag/            busca híbrida (FAISS + BM25) com fonte
  assistente/     cadeia LangChain com a LLM ajustada e os dois grafos LangGraph
  avaliacao/      métricas (usadas no Colab e nos testes) e avaliação do assistente
notebooks/        01 fine-tuning (Colab) · 02 demonstração (local)
scripts/          geradores dos notebooks e dos diagramas
tests/            testes automatizados
docs/             arquitetura com diagramas
reports/          relatório técnico
outputs/          resultados gerados (curadoria, fine-tuning, avaliação)
```

---

## Como executar

### 1. Ambiente local

```bash
python -m venv venv
venv\Scripts\activate                       # Windows (Linux/Mac: source venv/bin/activate)
python -m pip install -r requirements.txt
copy .env.example .env                      # nenhuma chave de API é necessária
```

### 2. Dataset (preprocessing, anonimização e curadoria)

```bash
python -m src.dados.preparar_dataset        # baixa PubMedQA/MedQuAD e gera data/processado/
python -m pytest                            # testes, inclusive a prova de anonimização
```

### 3. Fine-tuning (Google Colab)

Abrir `notebooks/01_fine_tuning_colab.ipynb` no Colab
([link direto](https://colab.research.google.com/github/GTerra07/tech-challenge-fase-03/blob/main/notebooks/01_fine_tuning_colab.ipynb)),
escolher **GPU T4** e executar tudo (~1h30). O notebook salva no Google Drive o adaptador LoRA, as
métricas de avaliação e o modelo em **GGUF** com o `Modelfile`.

### 4. Servir o modelo ajustado no Ollama

Baixe do Drive `assistente-medico.Q4_K_M.gguf` e `Modelfile` para a pasta `modelo/` e:

```bash
ollama create assistente-medico -f modelo/Modelfile
```

### 5. Assistente

```bash
python -m src.prontuario.banco                      # cria o prontuário de demonstração
jupyter notebook notebooks/02_demonstracao_assistente.ipynb
python -m src.avaliacao.avaliar_assistente --com-modelo   # avaliação ponta a ponta
```

---

## Decisões que valem explicação

**Fine-tuning ensina comportamento; a busca traz o conhecimento.** O modelo foi ajustado para
responder no formato da casa, citar a fonte, recusar prescrição e montar laudos e sugestões como
rascunho. O conteúdo dos protocolos chega pelo contexto, com a fonte — protocolo muda, e reindexar é
trivial; retreinar não.

**A LLM redige, as regras decidem.** Exames pendentes e valores críticos saem de regras
determinísticas escritas a partir dos prazos dos protocolos. A LLM recebe esses achados e redige
a sugestão. Alerta não pode depender de geração de texto.

**Defesa em camadas.** Pedido de prescrição é barrado por regra antes do modelo; o modelo também
aprendeu a recusar; a saída é verificada (dose, marca de rascunho, fonte recuperada); e nenhuma
sugestão vira conduta sem a decisão registrada de um médico.

**Nenhum dado pessoal chega ao modelo nem ao log.** O resumo do paciente passa pela anonimização
antes de entrar no prompt; o log guarda o prompt já anonimizado.

---

## Tecnologias

Python 3.12 · Unsloth · TRL · PEFT/LoRA · bitsandbytes · Hugging Face Datasets · LangChain ·
LangGraph · Ollama · FAISS · rank-bm25 · sentence-transformers · SQLite · pytest · Google Colab

## Autor

Gabriel Terra — [GTerra07](https://github.com/GTerra07)
