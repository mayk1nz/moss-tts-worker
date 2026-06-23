# moss-tts-worker

RunPod serverless worker rodando **MOSS-TTS-v1.5** (OpenMOSS, Apache 2.0) — modelo TTS state-of-the-art com paridade técnica vs ElevenLabs V3.

## Specs

- **Modelo:** [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
- **License:** Apache 2.0 (livre comercial)
- **Tamanho:** 8B params (BF16 → ~16GB no disco/VRAM)
- **Idiomas:** 31 (incl PT/ES/EN)
- **WER vs ElevenLabs V3:** 0.0391 vs 0.0363 (gap quase zero — SOTA aberto)
- **Benchmark Seed-TTS-eval:** supera F5-TTS, VoxCPM, CosyVoice (declarado pelos autores)
- **VRAM minima:** 24GB recomendado (16GB modelo + ativacoes BF16)
- **Container disk:** **30GB+** (modelo + deps)
- **Sample rate nativo:** 48kHz

## Requisitos torch/transformers (estritos)

MOSS-TTS-v1.5 fixa no pyproject.toml:
- `torch==2.9.1+cu128`
- `torchaudio==2.9.1+cu128`
- `transformers==5.0.0`

Dockerfile faz force-reinstall do trio cu128 + transformers exato pra evitar conflict ABI.

## Input do job

```json
{
  "input": {
    "texts": ["frase 1", "frase 2"],
    "ref_audio_b64": "<base64 do wav/mp3 de referencia>",
    "ref_text": "(opcional, ignorado pelo MOSS — voice cloning eh apenas por ref audio)",
    "language": "(opcional, ignorado — MOSS auto-detecta)"
  }
}
```

## Output

```json
{
  "audio_b64": "...",
  "audio_format": "mp3",
  "sample_rate": 48000,
  "n_chunks": 2,
  "chunk_durations": [3.4, 2.1],
  "gen_seconds": 2.8,
  "audio_seconds": 5.5,
  "rtf": 0.509,
  "model": "OpenMOSS-Team/MOSS-TTS-v1.5"
}
```

Em caso de erro retorna `{"error": true, "exception_type": ..., "traceback": ...}` em vez de crashar.

## RunPod deploy

- Queue mode
- **GPU 24GB+ (recomendado 32GB pra margem)**
- max_workers 5, idle_timeout 5s, FlashBoot ON
- **Container disk 30GB** (modelo 8B BF16 ocupa ~16GB)
