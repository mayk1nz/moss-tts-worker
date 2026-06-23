# Worker GPU do RunPod — MOSS-TTS-v1.5 (OpenMOSS, Apache 2.0).
# 31 idiomas incl PT/ES/EN, zero-shot voice cloning.
# WER ~0.039 (rival ElevenLabs V3 ~0.036). SOTA Seed-TTS-eval vs F5/VoxCPM/CosyVoice.
#
# IMPORTANTE: MOSS-TTS-v1.5 requer torch==2.9.1+cu128 + transformers==5.0.0
# (especificado no pyproject.toml). Imagem base eh torch 2.7.1, fazemos
# force-reinstall do trio + transformers 5.0 no final.

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV HF_HOME=/app/hf \
    MP3_BITRATE=64k \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # MOSS-TTS-v1.5 = 8B BF16 ~16GB modelo, sobra ~6GB pra ativacoes em 24GB GPU
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System libs (soundfile/ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: base deps que NAO sao do torch
RUN pip install --no-cache-dir \
      runpod soundfile numpy huggingface_hub einops accelerate sentencepiece

# Layer 2: transformers 5.0.0 exato (requirement do MOSS-TTS-v1.5)
RUN pip install --no-cache-dir transformers==5.0.0

# Layer 3: force-reinstall trio torch matched cu128 (mesmo padrao chatterbox/voxcpm)
# Pinado em 2.9.1+cu128 conforme pyproject.toml do MOSS-TTS.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# Pre-baixa pesos (cold start nao precisa baixar). 16GB+ download — pode falhar
# no build host se nao tiver disk. Se cair, runtime baixa.
RUN python -c "\
import os; os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='1200';\
try:\
    from huggingface_hub import snapshot_download;\
    snapshot_download(repo_id='OpenMOSS-Team/MOSS-TTS-v1.5', cache_dir='/app/hf', allow_patterns=['*.json','*.py','*.txt','*.safetensors']);\
    print('PRE-BAKE OK');\
except Exception as e:\
    print(f'PRE-BAKE SKIP: {type(e).__name__}: {str(e)[:300]}')\
" || true

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
