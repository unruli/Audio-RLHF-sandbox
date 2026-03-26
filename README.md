# Audio-RLHF-sandbox

A proof-of-concept sandbox for solving the **taste gap** in AI audio mixing. This repository provides the data-engineering foundation needed to capture subjective mixing decisions from professional audio engineers and map them into structured data suitable for training a Multi-Modal Audio AI using **Reinforcement Learning from Human Feedback (RLHF)**.

---

## Goal

Map human artistic intent into structured data for AI training by recording the causal chain: *raw audio state → engineer action → mixed result → subjective rating*.

---

## Features

### Component 1 – Action-State Data Schema (`schema/`)

A JSON Schema (`schema/action_state_log.schema.json`) that logs the causal relationship between:

- **Pre-action state** – objective audio features measured from the raw/unprocessed track.
- **DAW action** – the specific parameter change made by the engineer (e.g. EQ, compression, reverb), including the plugin, parameter name, before/after values, and an intent label.
- **Post-action state** – objective audio features measured after the action is applied.
- **Subjective feedback** *(optional)* – a structured rating capturing overall quality and perceptual attribute changes (warmth, brightness, clarity, punch, spaciousness, loudness balance).

See `examples/example_action_state_log.json` for a complete, validated example.

### Component 2 – Baseline Feature Extractor (`scripts/extract_features.py`)

A Python script that measures **objective audio metrics** from an audio file and outputs a JSON object conforming to the `AudioState` definition in the schema.

Metrics extracted:

| Feature | Description |
|---|---|
| `loudness_lufs` | Integrated loudness per ITU-R BS.1770-4 |
| `true_peak_dbtp` | True peak level in dBTP |
| `spectral_centroid_hz` | Mean spectral centroid (perceived brightness) |
| `spectral_rolloff_hz` | 85 % spectral roll-off frequency |
| `spectral_flatness` | Wiener entropy (0 = pure tone, 1 = white noise) |
| `rms_db` | RMS energy level in dB |
| `zero_crossing_rate` | Mean zero-crossing rate |
| `mfcc` | 13 mean MFCCs (timbral texture) |

---

## Project Structure

```
Audio-RLHF-sandbox/
├── schema/
│   └── action_state_log.schema.json   # JSON Schema (Component 1)
├── scripts/
│   └── extract_features.py            # Baseline feature extractor (Component 2)
├── examples/
│   └── example_action_state_log.json  # Validated example log entry
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract features from an audio file

```bash
python scripts/extract_features.py path/to/your/track.wav
```

Output is printed as JSON to stdout:

```json
{
  "file_reference": "path/to/your/track.wav",
  "duration_seconds": 210.5,
  "sample_rate_hz": 44100,
  "loudness_lufs": -18.3,
  "true_peak_dbtp": -6.0,
  "spectral_centroid_hz": 2150.7,
  "spectral_rolloff_hz": 4800.0,
  "spectral_flatness": 0.12,
  "rms_db": -15.4,
  "zero_crossing_rate": 0.087,
  "mfcc": [-210.5, 121.3, -34.2, 18.7, -11.2, 8.4, -5.1, 3.9, -2.7, 1.8, -1.2, 0.9, -0.6]
}
```

Save to a file instead:

```bash
python scripts/extract_features.py path/to/your/track.wav --output features.json
```

---

## Roadmap

- [ ] DAW plugin / OSC bridge to capture actions in real time
- [ ] Multi-engineer aggregation and preference modelling
- [ ] RLHF training loop integrating the action-state logs with a reward model
- [ ] Web UI for engineer feedback collection

---

## License

See [LICENSE](LICENSE).
