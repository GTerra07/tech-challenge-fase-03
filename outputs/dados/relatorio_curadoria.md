# Relatório de curadoria do dataset

Gerado por `python -m src.dados.preparar_dataset`. Não editar à mão.

## Filtros

| Etapa | interno | pubmedqa | medquad |
|---|---:|---:|---:|
| Entrada | 1141 | 1000 | 632 |
| Removidos: acima do contexto de treino | 2 | 0 | 0 |
| Removidos: duplicata (mesma entrada) | 44 | 0 | 0 |
| Removidos: marcação residual | 0 | 3 | 0 |
| **Aprovados** | **1095** | **997** | **632** |

## Divisão

| Conjunto | Total | interno | pubmedqa | medquad |
|---|---:|---:|---:|---:|
| treino | 1732 | 831 | 501 | 400 |
| validacao | 150 | 50 | 50 | 50 |
| teste | 514 | 214 | 200 | 100 |

## Tarefas no treino

| Tarefa | Exemplos |
|---|---:|
| pubmedqa | 501 |
| medquad | 400 |
| pergunta_paciente | 230 |
| caso | 228 |
| pergunta | 177 |
| recusa | 89 |
| laudo | 69 |
| receita | 22 |
| sem_contexto | 16 |
