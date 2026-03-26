"""
Baseline Audio Feature Extractor for Audio-RLHF Sandbox.

Extracts objective audio features from a WAV file and outputs a JSON object
that conforms to the AudioState definition in schema/action_state_log.schema.json.

Features extracted:
    - Integrated loudness in LUFS (ITU-R BS.1770-4)
    - True peak level in dBTP
    - Spectral centroid in Hz
    - Spectral roll-off frequency in Hz
    - Spectral flatness (Wiener entropy)
    - RMS energy in dB
    - Zero-crossing rate
    - MFCCs (13 coefficients, mean across frames)

Usage:
    python scripts/extract_features.py <audio_file.wav>
    python scripts/extract_features.py <audio_file.wav> --output features.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
    import pyloudnorm as pyln
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}.\n"
        "Install requirements with:  pip install -r requirements.txt"
    )


def _load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file and return (samples, sample_rate).

    Supports any format handled by libsndfile (WAV, FLAC, AIFF, …).
    Multi-channel audio is converted to mono by averaging channels.
    """
    data, sample_rate = sf.read(str(file_path), always_2d=True)
    # Convert to mono
    mono = data.mean(axis=1)
    return mono.astype(np.float32), int(sample_rate)


def _measure_loudness(samples: np.ndarray, sample_rate: int) -> dict:
    """Measure ITU-R BS.1770 integrated loudness and true peak.

    Returns a dict with keys ``loudness_lufs`` and ``true_peak_dbtp``.
    """
    meter = pyln.Meter(sample_rate)
    # pyloudnorm expects shape (samples,) or (samples, channels)
    loudness_lufs = meter.integrated_loudness(samples)

    # True peak: upsample by 4x and measure peak
    upsampled = librosa.resample(samples, orig_sr=sample_rate, target_sr=sample_rate * 4)
    true_peak_linear = np.max(np.abs(upsampled))
    # Guard against log(0)
    true_peak_dbtp = float(20.0 * np.log10(true_peak_linear)) if true_peak_linear > 0 else -np.inf

    return {
        "loudness_lufs": round(float(loudness_lufs), 2),
        "true_peak_dbtp": round(true_peak_dbtp, 2),
    }


def _spectral_features(samples: np.ndarray, sample_rate: int) -> dict:
    """Extract spectral features using librosa.

    Returns a dict with spectral_centroid_hz, spectral_rolloff_hz, and
    spectral_flatness (all scalar means across frames).
    """
    # Spectral centroid – mean across frames
    centroid_frames = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
    spectral_centroid_hz = float(np.mean(centroid_frames))

    # Spectral roll-off (85% energy threshold) – mean across frames
    rolloff_frames = librosa.feature.spectral_rolloff(
        y=samples, sr=sample_rate, roll_percent=0.85
    )
    spectral_rolloff_hz = float(np.mean(rolloff_frames))

    # Spectral flatness – mean across frames
    flatness_frames = librosa.feature.spectral_flatness(y=samples)
    spectral_flatness = float(np.mean(flatness_frames))

    return {
        "spectral_centroid_hz": round(spectral_centroid_hz, 2),
        "spectral_rolloff_hz": round(spectral_rolloff_hz, 2),
        "spectral_flatness": round(float(np.clip(spectral_flatness, 0.0, 1.0)), 6),
    }


def _temporal_features(samples: np.ndarray, sample_rate: int) -> dict:
    """Extract temporal features: RMS (dB) and zero-crossing rate.

    Returns a dict with rms_db and zero_crossing_rate.
    """
    # RMS energy in dB
    rms_linear = float(np.sqrt(np.mean(samples ** 2)))
    rms_db = float(20.0 * np.log10(rms_linear)) if rms_linear > 0 else -np.inf

    # Zero-crossing rate – mean across frames
    zcr_frames = librosa.feature.zero_crossing_rate(y=samples)
    zero_crossing_rate = float(np.mean(zcr_frames))

    return {
        "rms_db": round(rms_db, 2),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
    }


def _mfcc_features(samples: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> dict:
    """Extract mean MFCCs across all frames.

    Returns a dict with key ``mfcc`` containing a list of ``n_mfcc`` floats.
    """
    mfcc_frames = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_means = [round(float(v), 4) for v in np.mean(mfcc_frames, axis=1)]
    return {"mfcc": mfcc_means}


def extract_features(file_path: str | Path) -> dict:
    """Extract all baseline audio features from *file_path*.

    Parameters
    ----------
    file_path:
        Path to the input audio file (WAV, FLAC, AIFF, etc.).

    Returns
    -------
    dict
        A dictionary conforming to the ``AudioState`` definition in
        ``schema/action_state_log.schema.json``.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    samples, sample_rate = _load_audio(file_path)
    duration_seconds = round(float(len(samples) / sample_rate), 4)

    features: dict = {
        "file_reference": str(file_path),
        "duration_seconds": duration_seconds,
        "sample_rate_hz": sample_rate,
    }

    # Suppress librosa UserWarnings about PySoundFile being unavailable on some platforms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features.update(_measure_loudness(samples, sample_rate))
        features.update(_spectral_features(samples, sample_rate))
        features.update(_temporal_features(samples, sample_rate))
        features.update(_mfcc_features(samples, sample_rate))

    return features


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract baseline audio features from an audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the input audio file (WAV, FLAC, AIFF, etc.).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write the JSON output. Defaults to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    features = extract_features(args.audio_file)
    json_output = json.dumps(features, indent=args.indent)

    if args.output:
        args.output.write_text(json_output, encoding="utf-8")
        print(f"Features written to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
