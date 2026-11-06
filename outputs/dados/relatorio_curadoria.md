# Relatório de curadoria do dataset

Gerado por `python -m src.dados.preparar_dataset`. Não editar à mão.

## Filtros

| Etapa | interno | pubmedqa | medquad |
|---|---:|---:|---:|
| Entrada | 872 | 1000 | 632 |
| Removidos: duplicata (mesma entrada) | 46 | 0 | 0 |
| Removidos: acima do contexto de treino | 2 | 0 | 0 |
| Removidos: marcação residual | 0 | 3 | 0 |
| **Aprovados** | **824** | **997** | **632** |

## Divisão

| Conjunto | Total | interno | pubmedqa | medquad |
|---|---:|---:|---:|---:|
| treino | 1488 | 587 | 501 | 400 |
| validacao | 126 | 26 | 50 | 50 |
| teste | 511 | 211 | 200 | 100 |

## Tarefas no treino

| Tarefa | Exemplos |
|---|---:|
| pubmedqa | 501 |
| medquad | 400 |
| caso | 212 |
| pergunta | 174 |
| recusa | 93 |
| laudo | 64 |
| sem_contexto | 25 |
| receita | 19 |
