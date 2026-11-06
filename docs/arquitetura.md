# Arquitetura da Solução

Assistente virtual médico do Hospital Universitário: LLM ajustada por fine-tuning, integrada com
LangChain ao prontuário e aos protocolos internos, e orquestrada com LangGraph.

Os diagramas dos dois fluxos LangGraph (seções 3 e 4) são **gerados do próprio grafo compilado**
por `python scripts/gerar_diagramas.py` — não são desenhos feitos à mão.

---

## 1. Visão geral

```mermaid
flowchart LR
  subgraph Dados["Preparação de dados (local, testada)"]
    D1[Protocolos, POPs e modelos<br/>de laudo/receita] --> G[Gerador de exportações<br/>com dado pessoal falso]
    G --> A[Anonimização<br/>ponto único]
    P1[(PubMedQA)] --> PP
    P2[(MedQuAD)] --> PP
    A --> PP[Preprocessing → curadoria<br/>→ treino/validação/teste]
  end

  subgraph Colab["Fine-tuning (Google Colab, GPU T4)"]
    PP --> FT[LLaMA 3.1 8B 4-bit<br/>Unsloth + LoRA + SFTTrainer]
    FT --> AV[Avaliação<br/>base × ajustado]
    FT --> GG[GGUF q4_k_m<br/>+ Modelfile]
  end

  subgraph Local["Assistente (máquina do hospital)"]
    GG --> OL[Ollama<br/>assistente-medico]
    OL --> LC["Cadeia LangChain<br/>Prompt | ChatOllama | Parser"]
    DB[(Prontuário SQLite<br/>somente leitura)] --> T[Ferramentas LangChain]
    D1 --> RAG[Busca híbrida<br/>FAISS + BM25]
    LC --> LG[Fluxos LangGraph]
    T --> LG
    RAG --> LG
    LG --> AL[Fila de alertas<br/>da equipe]
    LG --> AU[(Log de auditoria<br/>JSON Lines)]
    LG <--> MED((Médico<br/>validação humana))
  end
```

| Camada | Componente | Arquivo |
|---|---|---|
| Dados | documentos internos, gerador, anonimização, curadoria | `data/`, `src/dados/`, `src/seguranca/anonimizacao.py` |
| Modelo | fine-tuning, avaliação, exportação | `notebooks/01_fine_tuning_colab.ipynb`, `src/fine_tuning/formato.py`, `src/avaliacao/metricas.py` |
| Integração LangChain | cadeia com a LLM ajustada, ferramentas do prontuário | `src/assistente/llm.py`, `src/prontuario/ferramentas.py` |
| Conhecimento | busca híbrida com fonte | `src/rag/indice.py` |
| Regras clínicas | pendências e valores críticos | `src/prontuario/regras.py` |
| Orquestração LangGraph | fluxo do paciente, perguntas clínicas | `src/assistente/grafo_paciente.py`, `src/assistente/grafo_pergunta.py` |
| Segurança | limites de atuação, auditoria | `src/seguranca/limites.py`, `src/seguranca/auditoria.py` |

---

## 2. Princípio que organiza tudo: a LLM redige, as regras decidem

| Decisão | Quem toma | Por quê |
|---|---|---|
| Quais exames estão pendentes, quais valores são críticos | **regras determinísticas** (`regras.py`) | alerta não pode depender de geração de texto; prova-se com teste |
| Que dado do prontuário consultar | **o grafo** (ferramentas chamadas pelo código) | o modelo nunca escreve SQL nem escolhe o que consultar |
| Quais trechos de protocolo entram no contexto | **busca híbrida + regras** | a mesma função define o contexto no treino e na inferência |
| Como redigir a resposta e a sugestão | **LLM ajustada** | é o que o fine-tuning ensinou: formato da casa, citação, recusa |
| Se a sugestão vale | **médico** (interrupt do LangGraph) | nenhuma conduta é registrada sem decisão humana identificada |

---

## 3. Fluxo automatizado do paciente (LangGraph)

Ao receber um paciente: verificar exames pendentes → emitir alertas → sugerir conduta → validação humana.

<!-- grafo:paciente -->
```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	receber_paciente(receber_paciente)
	verificar_exames_pendentes(verificar_exames_pendentes)
	emitir_alertas(emitir_alertas)
	buscar_protocolos(buscar_protocolos)
	sugerir_tratamento(sugerir_tratamento)
	verificar_seguranca(verificar_seguranca)
	sugestao_por_regras(sugestao_por_regras)
	validacao_humana(validacao_humana)
	registrar_desfecho(registrar_desfecho)
	__end__([<p>__end__</p>]):::last
	__start__ --> receber_paciente;
	buscar_protocolos --> sugerir_tratamento;
	emitir_alertas --> buscar_protocolos;
	receber_paciente -. &nbsp;fim&nbsp; .-> __end__;
	receber_paciente -. &nbsp;seguir&nbsp; .-> verificar_exames_pendentes;
	sugerir_tratamento --> verificar_seguranca;
	sugestao_por_regras --> validacao_humana;
	validacao_humana --> registrar_desfecho;
	verificar_exames_pendentes -. &nbsp;sem_alertas&nbsp; .-> buscar_protocolos;
	verificar_exames_pendentes -. &nbsp;alertar&nbsp; .-> emitir_alertas;
	verificar_seguranca -. &nbsp;tentar_de_novo&nbsp; .-> sugerir_tratamento;
	verificar_seguranca -. &nbsp;usar_regras&nbsp; .-> sugestao_por_regras;
	verificar_seguranca -. &nbsp;aprovada&nbsp; .-> validacao_humana;
	registrar_desfecho --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
<!-- /grafo:paciente -->

- **Alertas antes da sugestão.** Valor crítico tem prazo de 30 minutos (PROT-007 §3); não pode esperar o médico ler a sugestão.
- **Ciclo de correção.** Se a barreira de saída reprova a sugestão (dose, sem marca de rascunho, fonte inventada ou alerta crítico omitido), o grafo volta a `sugerir_tratamento` uma vez; na segunda reprovação usa a sugestão das regras (`sugestao_por_regras`), marcada como fallback.
- **Pausa real.** `validacao_humana` usa `interrupt()` com checkpointer: o estado fica salvo e o fluxo só continua com `Command(resume={"decisao": ..., "medico": ...})`.
- **Retry com backoff** no nó que chama o modelo (`RetryPolicy`, 3 tentativas).

## 4. Perguntas clínicas contextualizadas (LangGraph + LangChain)

<!-- grafo:pergunta -->
```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	triagem_entrada(triagem_entrada)
	recusar(recusar)
	identificar_paciente(identificar_paciente)
	consultar_paciente(consultar_paciente)
	buscar_protocolos(buscar_protocolos)
	sem_base(sem_base)
	responder(responder)
	verificar_resposta(verificar_resposta)
	anotar_ressalva(anotar_ressalva)
	__end__([<p>__end__</p>]):::last
	__start__ --> triagem_entrada;
	buscar_protocolos -.-> responder;
	buscar_protocolos -.-> sem_base;
	consultar_paciente --> buscar_protocolos;
	identificar_paciente -. &nbsp;sem_paciente&nbsp; .-> buscar_protocolos;
	identificar_paciente -. &nbsp;com_paciente&nbsp; .-> consultar_paciente;
	responder --> verificar_resposta;
	triagem_entrada -. &nbsp;permitida&nbsp; .-> identificar_paciente;
	triagem_entrada -. &nbsp;bloqueada&nbsp; .-> recusar;
	verificar_resposta -. &nbsp;ok&nbsp; .-> __end__;
	verificar_resposta -. &nbsp;ressalva&nbsp; .-> anotar_ressalva;
	verificar_resposta -. &nbsp;tentar_de_novo&nbsp; .-> responder;
	anotar_ressalva --> __end__;
	recusar --> __end__;
	sem_base --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
<!-- /grafo:pergunta -->

- **Triagem antes do modelo.** Pedido de prescrição, dose, alta ou tentativa de *prompt injection* é recusado por regra, sem chamar a LLM.
- **Prontuário atualizado.** Se a pergunta cita um paciente (`P-0000`), as ferramentas leem o banco no momento da pergunta.
- **Sem base, sem resposta.** Se a melhor seção recuperada tem confiança abaixo de 0,25, o modelo não é acionado.
- **Fonte verificada.** Toda fonte citada na resposta precisa estar entre as recuperadas; senão, nova tentativa e, persistindo, a resposta sai com ressalva explícita.

---

## 5. Defesa em camadas (limites de atuação)

```mermaid
flowchart TB
  Q[Pedido do médico] --> L1{1. Triagem de entrada<br/>regras}
  L1 -- prescrição / dose / alta / injeção --> R[Recusa padrão PROT-008<br/>sem chamar o modelo]
  L1 -- permitido --> L2[2. LLM ajustada<br/>aprendeu a recusar e a citar]
  L2 --> L3{3. Barreira de saída<br/>dose? rascunho? fonte recuperada?}
  L3 -- reprovada --> L2
  L3 -- aprovada --> L4{4. Validação humana<br/>interrupt}
  L4 --> REG[(Auditoria: decisão,<br/>médico, horário)]
```

---

## 6. Do treino à inferência: o mesmo formato

```mermaid
flowchart LR
  F[src/fine_tuning/formato.py<br/>template Alpaca + instruções] --> T[Dataset de treino]
  F --> C[Notebook do Colab<br/>treino e avaliação]
  F --> M[Modelfile do Ollama<br/>TEMPLATE]
  F --> A[Assistente<br/>prompt de inferência]
```

O template Alpaca (Aula 2 de Fine-Tuning) é definido uma vez e usado nos quatro lugares. Se o formato
da inferência diferisse do treino, o modelo ajustado se comportaria como se não tivesse sido treinado.

---

## 7. Como isso rodaria em produção

```mermaid
flowchart LR
  PEP[Prontuário eletrônico<br/>HL7 FHIR] -->|réplica somente leitura| API
  LAB[Sistema do laboratório] -->|evento: resultado liberado| FILA[(Fila de eventos)]
  FILA --> WK[Worker LangGraph<br/>fluxo do paciente]
  WK --> API[Serviço do assistente<br/>API interna + OAuth2]
  API --> INF[Servidor de inferência<br/>vLLM/Ollama em GPU on-premise]
  API --> VEC[(Índice vetorial<br/>versionado por protocolo)]
  WK --> CK[(Checkpointer<br/>Postgres)]
  WK --> PAG[Sistema de chamados<br/>da enfermagem]
  API --> OBS[Logs, métricas e<br/>alertas de operação]
```

Detalhamento (implantação, monitoramento, versionamento, rollback e escalabilidade) no relatório técnico, seção *Implementação em Produção / MLOps*.
