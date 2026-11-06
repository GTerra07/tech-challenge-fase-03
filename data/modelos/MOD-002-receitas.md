---
codigo: MOD-002
titulo: Modelos internos de receita
versao: "1.0"
vigencia: 2026-01-01
responsavel: Farmácia Clínica e Diretoria Clínica
---

> Documento SINTÉTICO criado para fins acadêmicos (Tech Challenge FIAP, Fase 3). Não usar em prática real.

# MOD-002 — Modelos internos de receita

## 1. Regra geral

A receita é emitida e assinada exclusivamente pelo médico (PROT-008). O assistente virtual pode apenas formatar, no modelo da casa, uma prescrição JÁ DEFINIDA pelo médico, como rascunho para conferência; ele não escolhe medicamento, dose, via ou duração.

## 2. Receita simples

Campos obrigatórios: identificação do paciente; para cada item, nome do medicamento pela denominação genérica, concentração, forma farmacêutica, quantidade, via de administração, posologia e duração; data; nome do médico, registro no conselho e assinatura.

## 3. Receita de controle especial

Utilizada para medicamentos sujeitos a controle especial. Emitida em duas vias, com identificação completa do emitente (nome, registro no conselho, endereço do serviço), do paciente e, na dispensação, do comprador. Cada receita deve conter quantidade por extenso.

## 4. Rascunho gerado pelo assistente

O rascunho começa com a linha "RASCUNHO PARA CONFERÊNCIA — NÃO É RECEITA VÁLIDA" e termina com "Prescrição definida por [MEDICO]. Requer revisão, assinatura e carimbo do médico responsável."
