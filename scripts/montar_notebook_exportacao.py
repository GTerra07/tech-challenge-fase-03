"""
Gera notebooks/01b_exportar_gguf_colab.ipynb.

Plano B do fine-tuning: a sessao gratuita do Colab dura no maximo 3h30. Se ela
acabar depois de o adaptador LoRA ter sido salvo no Drive (passo 8 do notebook
principal) mas antes da exportacao, este notebook carrega o adaptador do Drive e
faz so a exportacao GGUF + Modelfile, sem retreinar.

Uso:
    python scripts/montar_notebook_exportacao.py
"""

from pathlib import Path

import nbformat as nbf

DESTINO = Path(__file__).resolve().parent.parent / "notebooks" / "01b_exportar_gguf_colab.ipynb"
REPO = "https://github.com/GTerra07/tech-challenge-fase-03.git"

celulas = []
md = lambda s: celulas.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: celulas.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Exportação GGUF a partir do adaptador salvo no Drive

Plano B do `01_fine_tuning_colab.ipynb`: carrega o adaptador LoRA de
`MyDrive/tech-challenge-fase-03/lora_model` e exporta o modelo ajustado em GGUF q4_k_m com o
`Modelfile` do Ollama. Não retreina. Ambiente: GPU T4.
""")
code(f"""
from google.colab import drive
drive.mount('/content/drive')
import os, sys, subprocess
VERSAO = 'v2'   # pasta do adaptador no Drive ('' para o v1, que ficou na raiz)
PASTA_DRIVE = '/content/drive/MyDrive/tech-challenge-fase-03' + (f'/{{VERSAO}}' if VERSAO else '')
REPO = '/content/tech-challenge-fase-03'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '--depth', '1', '{REPO}', REPO], check=True)
sys.path.insert(0, REPO)
assert os.path.exists(f'{{PASTA_DRIVE}}/lora_model'), 'adaptador não encontrado no Drive'
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
""")
code("""
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f'{PASTA_DRIVE}/lora_model',   # base + adaptador LoRA treinado
    max_seq_length = 2048, dtype = None, load_in_4bit = True)
model.save_pretrained_gguf('assistente_gguf', tokenizer, quantization_method='q4_k_m')
""")
code("""
import glob, shutil
gguf = sorted(glob.glob('**/*.gguf', recursive=True), key=os.path.getmtime)[-1]
print('gerado:', gguf, f'{os.path.getsize(gguf) / 1e9:.2f} GB')
shutil.copy(gguf, f'{PASTA_DRIVE}/assistente-medico.Q4_K_M.gguf')

from src.fine_tuning.formato import ALPACA
cabecalho = ALPACA.split('### Instruction:')[0]
MODELFILE = ('FROM ./assistente-medico.Q4_K_M.gguf\\n'
             'TEMPLATE \"\"\"' + cabecalho + '### Instruction:\\n{{ .System }}\\n\\n### Input:\\n{{ .Prompt }}\\n\\n### Response:\\n\"\"\"\\n'
             'PARAMETER stop \"<|eot_id|>\"\\nPARAMETER stop \"<|end_of_text|>\"\\nPARAMETER stop \"### Instruction:\"\\n'
             'PARAMETER temperature 0.1\\nPARAMETER num_ctx 4096\\n')
with open(f'{PASTA_DRIVE}/Modelfile', 'w') as f:
    f.write(MODELFILE)
print(MODELFILE)
print(os.listdir(PASTA_DRIVE))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celulas
nb["metadata"] = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, DESTINO)
print(f"gerado: {DESTINO} ({len(celulas)} células)")
