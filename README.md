# Assistente Médico com LLM Ajustada, LangChain e LangGraph

**Tech Challenge — Fase 3 · Pós-Tech em Engenharia de Inteligência Artificial (FIAP)**

Assistente virtual médico para um hospital universitário: um **LLaMA 3.1 8B ajustado por
fine-tuning** (Unsloth + LoRA) com protocolos, perguntas frequentes de médicos e modelos de laudo,
receita e procedimento do hospital; integrado com **LangChain** a um prontuário estruturado e a
uma busca nos protocolos com citação de fonte; e orquestrado com **LangGraph** num fluxo
automatizado que verifica exames pendentes, sugere condutas para validação humana e emite
alertas à equipe.

> 🚧 Em construção — README completo na entrega.

## Pipeline de dados (pronto)

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt

python -m src.dados.preparar_dataset      # gera data/processado/ e outputs/dados/
python -m pytest
```

Todos os dados internos do hospital são **sintéticos** e estão declarados como tal.

## Fine-tuning (Google Colab)

`notebooks/01_fine_tuning_colab.ipynb` — abrir no Colab com GPU T4 e executar tudo.
