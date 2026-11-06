"""
Gera notebooks/01_fine_tuning_colab.ipynb.

O notebook e gerado por script (e nao editado a mao) para que cada mudanca
fique revisavel no diff e para garantir que ele seja versionado sem saidas.

Uso:
    python scripts/montar_notebook_colab.py
"""

from pathlib import Path

import nbformat as nbf

DESTINO = Path(__file__).resolve().parent.parent / "notebooks" / "01_fine_tuning_colab.ipynb"
REPO = "https://github.com/GTerra07/tech-challenge-fase-03.git"

celulas = []
md = lambda s: celulas.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: celulas.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Fine-tuning do assistente médico — LLaMA 3.1 8B com Unsloth + LoRA

**Tech Challenge Fase 3 · requisito 1: fine-tuning de LLM com dados médicos internos.**

Este notebook roda no **Google Colab com GPU** e segue o caminho da Aula 2 da disciplina
*Fine-Tuning e RAG para Documentos*: modelo LLaMA quantizado em 4 bits, adaptadores **LoRA**
(PEFT), biblioteca **Unsloth**, `SFTTrainer` do TRL e template **Alpaca** com token EOS.

O que ele faz, em ordem:

1. Clona o repositório e carrega o dataset já preparado (preprocessing, anonimização e curadoria
   rodam localmente e estão testados — ver `src/dados/` e `outputs/dados/relatorio_curadoria.md`).
2. Treina uma época sobre o conjunto de treino, acompanhando a perda de validação.
3. Salva o adaptador LoRA no Google Drive **antes** de qualquer outra etapa.
4. Avalia o **mesmo modelo com o adaptador desligado (base) e ligado (ajustado)** no mesmo
   conjunto de teste separado: perplexidade, acurácia no PubMedQA, ROUGE-L, BLEU e verificações
   de comportamento (cita a fonte? recusa prescrever? cobre os alertas críticos?).
5. Exporta o modelo ajustado em **GGUF** com o `Modelfile` do Ollama, para servir localmente
   no assistente LangChain/LangGraph.

**Como rodar:** `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4` e depois
`Ambiente de execução → Executar tudo`. Tempo total esperado: cerca de 1h30 na T4.
""")

md("## 1. Google Drive e repositório")
code(f"""
from google.colab import drive
drive.mount('/content/drive')

import os, sys, subprocess
PASTA_DRIVE = '/content/drive/MyDrive/tech-challenge-fase-03'
os.makedirs(PASTA_DRIVE, exist_ok=True)

REPO = '/content/tech-challenge-fase-03'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '--depth', '1', '{REPO}', REPO], check=True)
sys.path.insert(0, REPO)
print(subprocess.run(['git', '-C', REPO, 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)
""")

md("""
## 2. Dependências

Mesma instalação do notebook oficial do Unsloth para Colab (versões fixadas de `transformers` e
`trl`, que mudam de API com frequência — o próprio material da Aula 2 alerta para isso), mais
`rouge_score` e `sacrebleu` para as métricas.
""")
code("""
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth
    !unsloth install-kernels
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.57.6
!pip install --no-deps trl==0.22.2
!pip install rouge_score sacrebleu
""")

md("## 3. Configuração")
code("""
from unsloth import FastLanguageModel, is_bfloat16_supported
import json, math, time, random
import torch
from datasets import load_dataset

from src.fine_tuning import formato as fmt
from src.avaliacao import metricas

MODELO_BASE = 'unsloth/Llama-3.1-8B-Instruct'   # LLaMA 3.1 8B, carregado em 4 bits
max_seq_length = 2048
load_in_4bit = True
SEMENTE = 3407
random.seed(SEMENTE)
torch.manual_seed(SEMENTE)

print(torch.cuda.get_device_name(0), '| bf16:', is_bfloat16_supported())
""")

md("""
## 4. Dataset

Gerado por `python -m src.dados.preparar_dataset` (ver o relatório de curadoria no repositório).
Cada exemplo já está no formato instrução / entrada / saída da Aula 2.
""")
code("""
arquivos = {n: f'{REPO}/data/processado/{n}.jsonl' for n in ('treino', 'validacao', 'teste')}
dados = load_dataset('json', data_files=arquivos)
print(dados)
from collections import Counter
print('tarefas no treino:', Counter(dados['treino']['tarefa']))
""")

md("""
## 5. Modelo base 4-bit e adaptadores LoRA

Hiperparâmetros da Aula 2: posto `r=16`, `lora_alpha=16`, LoRA nas projeções de atenção
(`q,k,v,o`) e do MLP (`gate,up,down`), gradient checkpointing do Unsloth.
""")
code("""
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODELO_BASE,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state = SEMENTE,
    use_rslora = False,
    loftq_config = None,
)
model.print_trainable_parameters()
""")

md("""
## 6. Template Alpaca + EOS

O template vem de `src/fine_tuning/formato.py` — o **mesmo** usado pelo assistente na inferência
e pelo `Modelfile` do Ollama. Sem o EOS no fim, o modelo não aprende a parar.
""")
code("""
EOS_TOKEN = tokenizer.eos_token
print('EOS:', EOS_TOKEN)

def formatar(lote):
    return {'text': [fmt.prompt(i, e, s) + EOS_TOKEN
                     for i, e, s in zip(lote['instruction'], lote['input'], lote['output'])]}

treino = dados['treino'].map(formatar, batched=True)
validacao = dados['validacao'].map(formatar, batched=True)
print(treino[0]['text'][:1500])
""")

md("""
## 7. Treino com `SFTTrainer`

Mesma configuração da aula (batch 2 × acumulação 4 = 8, lr 2e-4, `adamw_8bit`, scheduler linear),
com uma diferença deliberada: a aula treina só **60 passos** (demonstração, ~5% de uma época); aqui
treinamos **uma época completa** e medimos a perda de validação durante o treino.
""")
code("""
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = treino,
    eval_dataset = validacao,
    dataset_text_field = 'text',
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        eval_strategy = 'steps',
        eval_steps = 40,
        per_device_eval_batch_size = 2,
        optim = 'adamw_8bit',
        weight_decay = 0.01,
        lr_scheduler_type = 'linear',
        seed = SEMENTE,
        output_dir = 'outputs',
        report_to = 'none',
    ),
)
inicio = time.time()
estatisticas = trainer.train()
minutos_treino = (time.time() - inicio) / 60
print(f'treino: {minutos_treino:.1f} min', estatisticas.metrics)
""")

md("## 8. Salvar o adaptador LoRA e a curva de treino no Drive (antes de tudo)")
code("""
PASTA_SAIDA = f'{PASTA_DRIVE}/resultados'
os.makedirs(PASTA_SAIDA, exist_ok=True)
model.save_pretrained(f'{PASTA_DRIVE}/lora_model')
tokenizer.save_pretrained(f'{PASTA_DRIVE}/lora_model')

with open(f'{PASTA_SAIDA}/historico_treino.json', 'w') as f:
    json.dump({'log_history': trainer.state.log_history, 'minutos_treino': minutos_treino,
               'metricas_treino': estatisticas.metrics, 'gpu': torch.cuda.get_device_name(0),
               'modelo_base': MODELO_BASE, 'n_treino': len(treino), 'n_validacao': len(validacao)}, f, indent=2)

import matplotlib.pyplot as plt
log = trainer.state.log_history
passos_t = [h['step'] for h in log if 'loss' in h]; perda_t = [h['loss'] for h in log if 'loss' in h]
passos_v = [h['step'] for h in log if 'eval_loss' in h]; perda_v = [h['eval_loss'] for h in log if 'eval_loss' in h]
plt.figure(figsize=(8, 4))
plt.plot(passos_t, perda_t, label='treino'); plt.plot(passos_v, perda_v, 'o-', label='validação')
plt.xlabel('passo'); plt.ylabel('perda'); plt.title('Curva de perda do fine-tuning'); plt.legend(); plt.grid(alpha=.3)
plt.savefig(f'{PASTA_SAIDA}/curva_perda.png', dpi=150, bbox_inches='tight'); plt.show()
""")

md("""
## 9. Avaliação: modelo base × modelo ajustado

O **mesmo** modelo é avaliado com o adaptador LoRA **desligado** (= LLaMA 3.1 8B Instruct original)
e **ligado** (= ajustado), sobre o **mesmo** conjunto de teste, que não foi visto no treino.
Para o conteúdo interno, o teste inclui redações de perguntas e de pedidos de prescrição que
**nunca apareceram** no treino.

### 9.1 Perplexidade (só sobre os tokens da resposta)
""")
code("""
teste = dados['teste']
rng = random.Random(SEMENTE)

LIMITE_POR_TAREFA = {'pubmedqa': 200, 'medquad': 50, 'pergunta': 40, 'sem_contexto': 30,
                     'recusa': 40, 'caso': 30, 'laudo': 20, 'receita': 20}
amostra = []
for tarefa, limite in LIMITE_POR_TAREFA.items():
    linhas = [r for r in teste if r['tarefa'] == tarefa]
    rng.shuffle(linhas)
    amostra += linhas[:limite]
print(Counter(r['tarefa'] for r in amostra))

@torch.no_grad()
def perplexidade(exemplos):
    model.eval()
    soma, n_tokens = 0.0, 0
    for ex in exemplos:
        p = fmt.prompt(ex['instruction'], ex['input'])
        ids_p = tokenizer(p, return_tensors='pt').input_ids
        ids = tokenizer(p + ex['output'] + EOS_TOKEN, return_tensors='pt',
                        truncation=True, max_length=max_seq_length).input_ids.to('cuda')
        rotulos = ids.clone(); rotulos[:, :ids_p.shape[1]] = -100
        n = int((rotulos != -100).sum())
        if n == 0:
            continue
        soma += model(input_ids=ids, labels=rotulos).loss.item() * n
        n_tokens += n
    return math.exp(soma / n_tokens)

amostra_ppl = [r for r in amostra if r['tarefa'] != 'pubmedqa'][:120]
ppl = {'ajustado': perplexidade(amostra_ppl)}
with model.disable_adapter():
    ppl['base'] = perplexidade(amostra_ppl)
print(ppl)
""")

md("### 9.2 Geração das respostas (em lote) e métricas")
code("""
FastLanguageModel.for_inference(model)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_NOVOS = {'pubmedqa': 16, 'medquad': 300, 'pergunta': 220, 'sem_contexto': 120, 'recusa': 160,
             'caso': 450, 'laudo': 320, 'receita': 320}

@torch.no_grad()
def gerar(exemplos, lote=8):
    saidas = {}
    for tarefa in MAX_NOVOS:
        grupo = [e for e in exemplos if e['tarefa'] == tarefa]
        for i in range(0, len(grupo), lote):
            parte = grupo[i:i + lote]
            prompts = [fmt.prompt(e['instruction'], e['input']) for e in parte]
            enc = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
            out = model.generate(**enc, max_new_tokens=MAX_NOVOS[tarefa], use_cache=True, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            textos = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for e, t in zip(parte, textos):
                saidas[e['id']] = t.split('### Instruction:')[0].strip()
    return saidas

inicio = time.time()
respostas = {'ajustado': gerar(amostra)}
with model.disable_adapter():
    respostas['base'] = gerar(amostra)
print(f'geração: {(time.time() - inicio) / 60:.1f} min')
""")
code("""
def avaliar(resp):
    r = {}
    pm = [e for e in amostra if e['tarefa'] == 'pubmedqa']
    r['pubmedqa'] = metricas.acuracia_pubmedqa([metricas.rotulo_previsto(resp[e['id']]) for e in pm],
                                                [e['rotulo'] for e in pm])
    texto = [e for e in amostra if e['tarefa'] != 'pubmedqa']
    por_tarefa = {}
    for e in texto:
        por_tarefa.setdefault(e['tarefa'], []).append(e)
    r['similaridade'] = {t: {'rouge_l': sum(metricas.rouge_l(resp[e['id']], e['output']) for e in es) / len(es),
                             'bleu': metricas.bleu([resp[e['id']] for e in es], [e['output'] for e in es]),
                             'n': len(es)}
                         for t, es in por_tarefa.items()}
    internos = [e for e in texto if e['origem'] == 'interno']
    r['comportamento'] = metricas.agregar([(e['tarefa'], metricas.verificar(e, resp[e['id']])) for e in internos])
    return r

resultado = {m: avaliar(respostas[m]) for m in ('base', 'ajustado')}
resultado['perplexidade'] = ppl
resultado['modelo_base'] = MODELO_BASE
resultado['amostra'] = dict(Counter(r['tarefa'] for r in amostra))

with open(f'{PASTA_SAIDA}/metricas_avaliacao.json', 'w') as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
with open(f'{PASTA_SAIDA}/respostas_teste.jsonl', 'w') as f:
    for e in amostra:
        f.write(json.dumps({'id': e['id'], 'tarefa': e['tarefa'], 'referencia': e['output'],
                            'base': respostas['base'][e['id']], 'ajustado': respostas['ajustado'][e['id']]},
                           ensure_ascii=False) + '\\n')

for m in ('base', 'ajustado'):
    print(f"== {m}: PubMedQA acurácia {resultado[m]['pubmedqa']['acuracia']:.3f}")
    for t, v in resultado[m]['comportamento'].items():
        print('  ', t, {k: round(x, 3) for k, x in v.items()})
print('perplexidade', ppl)
""")

md("### 9.3 Exemplo lado a lado")
code("""
for tarefa in ('recusa', 'caso', 'pergunta'):
    e = next(x for x in amostra if x['tarefa'] == tarefa)
    print('=' * 100, '\\n', tarefa.upper(), '| entrada:\\n', e['input'][-600:])
    print('\\n--- BASE ---\\n', respostas['base'][e['id']][:900])
    print('\\n--- AJUSTADO ---\\n', respostas['ajustado'][e['id']][:900])
""")

md("""
## 10. Exportar para o Ollama (GGUF q4_k_m + Modelfile)

O Ollama é o caminho de integração da Aula 5 de LangChain. O `Modelfile` usa o **mesmo template
Alpaca** do treino: a instrução vai como mensagem de sistema e a entrada como mensagem do usuário.
""")
code("""
model.save_pretrained_gguf('assistente_gguf', tokenizer, quantization_method='q4_k_m')

import glob, shutil
gguf = sorted(glob.glob('**/*.gguf', recursive=True), key=os.path.getmtime)[-1]
print('gerado:', gguf, f'{os.path.getsize(gguf) / 1e9:.2f} GB')
shutil.copy(gguf, f'{PASTA_DRIVE}/assistente-medico.Q4_K_M.gguf')

MODELFILE = '''FROM ./assistente-medico.Q4_K_M.gguf
TEMPLATE \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{ .System }}

### Input:
{{ .Prompt }}

### Response:
\"\"\"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "### Instruction:"
PARAMETER temperature 0.1
PARAMETER num_ctx 4096
'''
with open(f'{PASTA_DRIVE}/Modelfile', 'w') as f:
    f.write(MODELFILE)
print(os.listdir(PASTA_DRIVE))
""")

md("""
## 11. Próximo passo (máquina local)

Baixe do Drive, pasta `tech-challenge-fase-03`, para a pasta `modelo/` do repositório:
`assistente-medico.Q4_K_M.gguf`, `Modelfile` e a pasta `resultados/` (vai para `outputs/fine_tuning/`).
Depois: `ollama create assistente-medico -f modelo/Modelfile`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celulas
nb["metadata"] = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
DESTINO.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, DESTINO)
print(f"gerado: {DESTINO} ({len(celulas)} células)")
