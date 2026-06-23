"""RunPod serverless worker — MOSS-TTS-v1.5 (OpenMOSS, Apache 2.0).

31 idiomas incl PT/ES/EN. Zero-shot voice cloning via processor.build_user_message
com argumento `reference=[ref_audio_path]`. WER ~0.039 (paridade ElevenLabs V3).

Output: mp3 64k (compativel com gateway que ja espera esse formato).
Sample rate nativo: 48kHz (modelo retorna numpy/tensor).
"""
import base64
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf
import torch
import runpod

DEVICE = os.environ.get("DEVICE", "cuda")
MP3_BITRATE = os.environ.get("MP3_BITRATE", "64k")
MODEL_ID = os.environ.get("MODEL_ID", "OpenMOSS-Team/MOSS-TTS-v1.5")
# bf16 segue o tensor_type do modelo no HF (recomendado pra 8B em 24GB GPU)
DTYPE = os.environ.get("DTYPE", "bfloat16")

print(f"[boot] python={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[boot] cuda_device={torch.cuda.get_device_name(0)} "
          f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

print(f"[boot] importando transformers (AutoModel + AutoProcessor) ...", flush=True)
from transformers import AutoModel, AutoProcessor

print(f"[boot] carregando MOSS-TTS ({MODEL_ID}, device={DEVICE}, dtype={DTYPE}) ...", flush=True)
_t0 = time.time()
try:
    torch_dtype = getattr(torch, DTYPE)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch_dtype
    )
    if DEVICE == "cuda":
        model = model.to(DEVICE)
    model.eval()

    # Sample rate — tenta pegar do model ou processor; fallback 48kHz (padrao MOSS).
    SR = (getattr(getattr(model, "config", None), "sample_rate", None)
          or getattr(processor, "sample_rate", None)
          or 48000)
    SR = int(SR)
    print(f"[boot] modelo carregado em {time.time() - _t0:.1f}s | sr={SR}Hz", flush=True)
except Exception as e:
    print(f"[boot] ERRO no model load: {type(e).__name__}: {e}",
          flush=True, file=sys.stderr)
    traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()
    raise

_ref_cache = {}  # sha256 -> path da ref no disco


def _get_ref_path(ref_audio_b64, ref_text):
    key = hashlib.sha256((ref_audio_b64[:1024] + "|" + ref_text).encode()).hexdigest()
    cached = _ref_cache.get(key)
    if cached and os.path.exists(cached):
        return cached
    path = os.path.join(tempfile.gettempdir(), f"moss_ref_{key[:16]}.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(ref_audio_b64))
    _ref_cache[key] = path
    return path


def _extract_audio(outputs):
    """MOSS-TTS pode retornar lista de tensors/arrays ou dict com 'audio' key.
    Normaliza pra numpy float32 1-D."""
    # Caso 1: lista de tensors/arrays
    if isinstance(outputs, list):
        outputs = outputs[0] if outputs else None
    # Caso 2: dict
    if isinstance(outputs, dict):
        outputs = outputs.get("audio") or outputs.get("waveform") or outputs.get("output")
    # Tensor -> numpy
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.squeeze().detach().cpu().numpy()
    return np.asarray(outputs, dtype=np.float32).reshape(-1)


def _handler_impl(job):
    inp = job["input"]
    texts = inp["texts"]
    ref_audio_b64 = inp["ref_audio_b64"]
    ref_text = inp.get("ref_text", "")
    # MOSS-TTS auto-detecta idioma do texto (como OmniVoice/VoxCPM).
    _language = inp.get("language") or inp.get("lang") or "en"

    ref_path = _get_ref_path(ref_audio_b64, ref_text)

    pieces = []
    durations = []
    t0 = time.time()
    for txt in texts:
        # API MOSS-TTS: processor.build_user_message(text, reference=[paths])
        conversations = [
            [processor.build_user_message(text=txt, reference=[ref_path])]
        ]
        batch = processor(conversations, mode="generation")
        # Move tensores pro device
        batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        with torch.inference_mode():
            outputs = model.generate(**batch)

        arr = _extract_audio(outputs)
        pieces.append(arr)
        durations.append(round(len(arr) / SR, 3))
    gen_seconds = round(time.time() - t0, 3)

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    audio_seconds = round(len(audio) / SR, 3)

    # WAV -> MP3 via ffmpeg pipe (mesmo padrao dos outros workers)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, SR, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", "pipe:0",
         "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
         "-f", "mp3", "pipe:1"],
        input=wav_buf.read(), capture_output=True, check=True,
    )
    mp3_bytes = proc.stdout

    return {
        "audio_b64": base64.b64encode(mp3_bytes).decode(),
        "audio_format": "mp3",
        "sample_rate": SR,
        "n_chunks": len(texts),
        "chunk_durations": durations,
        "gen_seconds": gen_seconds,
        "audio_seconds": audio_seconds,
        "rtf": round(gen_seconds / audio_seconds, 4) if audio_seconds else None,
        "model": MODEL_ID,
    }


def handler(job):
    """Wrapper que captura QUALQUER exception e retorna no payload em vez de crashar."""
    try:
        return _handler_impl(job)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[handler ERROR] {type(e).__name__}: {e}\n{tb}",
              flush=True, file=sys.stderr)
        return {
            "error": True,
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "traceback": tb,
        }


runpod.serverless.start({"handler": handler})
