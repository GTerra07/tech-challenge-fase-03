# Relatório de curadoria do dataset

Gerado por `python -m src.dados.preparar_dataset`. Não editar à mão.

## Filtros

| Etapa | interno | pubmedqa | medquad |
|---|---:|---:|---:|
| Entrada | 871 | 1000 | 632 |
| Removidos: duplicata | 44 | 0 | 0 |
| Removidos: marcação residual | 0 | 3 | 0 |
| **Aprovados** | **827** | **997** | **632** |

## Divisão

| Conjunto | Total | interno | pubmedqa | medquad |
|---|---:|---:|---:|---:|
| treino | 1492 | 591 | 501 | 400 |
| validacao | 126 | 26 | 50 | 50 |
| teste | 510 | 210 | 200 | 100 |

## Tarefas no treino

| Tarefa | Exemplos |
|---|---:|
| pubmedqa | 501 |
| medquad | 400 |
| caso | 213 |
| pergunta | 174 |
| recusa | 95 |
| laudo | 65 |
| sem_contexto | 26 |
| receita | 18 |
