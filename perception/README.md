# Perception Stack

Modular ASR/TTS perception plugins running as an MCP HTTP server. Connects to Agent Core via MCP tool calls and exchanges audio/text over ROS2 DDS topics.

## Audio Requirements for ASR

The ASR plugin (VAD + speech recognition) has strict requirements on the audio stream it receives. Any mic driver that does not meet these requirements will produce no output.

### ROS2 Message Type

```
audio_msgs/AudioChunk
  std_msgs/Header header
  string format          # must be "audio/pcm-16k"
  uint8[] data           # raw PCM bytes (little-endian signed 16-bit)
```

### PCM Format

| Parameter | Required value |
|-----------|---------------|
| Encoding | 16-bit signed integer, little-endian (PCM_S16_LE) |
| Sample rate | **16 000 Hz** |
| Channels | **Mono (1 channel)** |
| `format` field | `"audio/pcm-16k"` |

### Chunk Size

| Parameter | Constraint |
|-----------|-----------|
| Minimum | **1 024 bytes** (512 samples, ~32 ms) |
| Recommended | 1 024 – 4 096 bytes (32 – 128 ms per chunk) |
| Maximum | No hard limit, but very large chunks increase latency |

Chunks smaller than 1 024 bytes are **silently discarded** by the VAD. This is the most common cause of "ASR receives audio but never outputs anything."

> **Why 512 samples?** The Silero VAD model requires at least one 512-sample window to compute a speech probability. WebRTC VAD requires 480-sample (30 ms) frames. Both backends use 512 samples as the minimum chunk size.

### Common Pitfalls

#### External USB mic (ALSA, 48 kHz native rate)

Most USB audio interfaces run at 48 000 Hz. After downsampling to 16 000 Hz, a 512-frame ALSA period becomes only **170 samples (340 bytes)** — below the VAD minimum.

**Fix (already applied in `phanthymotus-driver`):** Buffer resampled output until 512 samples are accumulated before publishing each `AudioChunk`.

If you are writing a custom mic driver, apply the same buffering pattern:

```python
TARGET = 1024  # bytes (512 int16 samples)
_buf = bytearray()

# Inside your capture loop, after resampling:
_buf += resampled_bytes
while len(_buf) >= TARGET:
    chunk, _buf = bytes(_buf[:TARGET]), _buf[TARGET:]
    publish(chunk)
```

#### Native G1 robot mic (UDP multicast)

Publishes raw 16 kHz PCM at 1 024 bytes per chunk. No resampling or buffering needed.

---

## VAD Tuning

The VAD parameters can be adjusted per ASR canvas card via the instance config (⚙ button):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `vad_threshold` | `0.5` | Speech probability threshold (0–1). Raise to `0.7`–`0.85` in noisy environments (e.g. robot motor noise). |
| `vad_silence_ms` | `400` | Silence duration (ms) required before an utterance is considered complete. |

---

## Topic Naming

| Direction | Topic pattern | Format |
|-----------|--------------|--------|
| Input (mic) | `/{namespace}/mic/audio` or `/{namespace}/ext_mic/{id}/audio` | `audio/pcm-16k` |
| Output (ASR result) | `{input_topic}/asr` | `data/json` |

ASR result JSON:
```json
{
  "text": "recognized speech text",
  "audio_start_ts": 1234567890.123,
  "audio_end_ts":   1234567891.456,
  "asr_complete_ts": 1234567891.789
}
```

## Nearest Obstacle Distance

The `obstacle` processor estimates the nearest forward obstacle distance from
a compressed camera image. PNG inputs default to the indoor evaluation rule;
JPG/JPEG inputs default to the outdoor rule. A canvas instance can override
the scene with `indoor` or `outdoor`.

The processor publishes JSON to `<input_topic>/obstacle_distance`:

```json
{
  "timestamp": 1234567890.123,
  "distance_m": 2.43,
  "near_obstacle": false,
  "confidence": 0.9,
  "degraded": false,
  "reason": ""
}
```

For leaderboard-style single-image inference, run from the repository root:

```bash
PYTHONPATH=perception python3 perception/judge_obstacle_distance.py image.png
```

Successful stdout contains only one floating-point value. Runtime diagnostics
and model library messages are redirected to stderr.

### Models

Model files must live outside Git. The default cache root is
`/models/obstacle_distance`, mounted from `/opt/embodied/models` by the
deployment service. Supported environment variables:

| Variable | Meaning |
|---|---|
| `OBSTACLE_MODEL_DIR` | Root directory for obstacle models |
| `OBSTACLE_DEPTH_INDOOR_MODEL` | Hugging Face model ID or local model directory |
| `OBSTACLE_DEPTH_OUTDOOR_MODEL` | Hugging Face model ID or local model directory |
| `OBSTACLE_DETECTOR_MODEL` | Local YOLOv8n weight path |
| `OBSTACLE_DETECTOR_MODEL_URL` | URL used when the detector file is absent |
| `OBSTACLE_DETECTOR_MODEL_SHA256` | Optional detector checksum |
| `OBSTACLE_OUTDOOR_COMPENSATION_M` | Camera-to-front-bumper compensation; default `1.7` |
| `OBSTACLE_FALLBACK_DISTANCE_M` | Conservative finite result after inference failure |

For an offline judge, download the two Hugging Face model snapshots into the
external model volume, mirror them to the required JuiceFS HTTP service, and
set `OBSTACLE_DEPTH_INDOOR_MODEL` and `OBSTACLE_DEPTH_OUTDOOR_MODEL` to the
extracted local directories. See
`plugins/obstacle_distance/MODELS.md` for exact model IDs and licenses.

This is a zero-shot baseline because no competition training set is available.
The `1.7 m` outdoor compensation must be calibrated when official camera
extrinsics or labeled validation data become available.
