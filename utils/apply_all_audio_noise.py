import argparse
import shutil
import subprocess
from pathlib import Path
import tempfile
from typing import Optional
import json
import random

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

AUDIO_CORRUPTIONS = ["reverb", "compress", "jitter", "mp3", "snr_white", "bitcrushing", "bandlimit"]

# Use lossless PCM audio inside an MKV container by default and make this the
# non-optional project behavior. This ensures exact WAV extraction later.
AUDIO_CODEC = "pcm_s32le"
OUT_EXTENSION = ".mkv"  # enforce MKV output so WAV extraction can use exact PCM bytes

def _ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")

def _run(cmd: list[str]):
    subprocess.run(cmd, check=True)

def _iter_videos(inp: Path, recursive: bool):
    if inp.is_file():
        if inp.suffix.lower() in VIDEO_EXTS:
            yield inp
        return
    if recursive:
        for p in inp.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p
    else:
        for p in inp.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p

def _pre_af():
    return "dynaudnorm=f=150:g=15:p=0.95:m=10"

def _snr_to_anoisesrc_amplitude(_: float) -> str:
    return "1.0"

def _count_audio_streams(video: Path) -> int:
    """Return the number of audio streams in a video file using ffprobe. Returns 0 on error."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(video),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if not out:
            return 0
        lines = [l for l in out.splitlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


def _probe_audio_stream_props(video: Path, stream_idx: int) -> tuple[int, int] | None:
    """Return (channels, sample_rate) for the given audio stream index, or None on error.

    Uses ffprobe JSON output to reliably parse stream properties.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", f"a:{stream_idx}",
            "-show_entries", "stream=channels,sample_rate",
            "-of", "json",
            str(video),
        ]
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        streams = data.get("streams")
        # ensure streams is a list before indexing
        if not isinstance(streams, list) or len(streams) == 0:
            return None
        st = streams[0]
        # sample_rate may be returned as a string, channels as int
        channels = st.get("channels")
        sample_rate = st.get("sample_rate")
        if channels is None or sample_rate is None:
            return None
        try:
            return int(channels), int(sample_rate)
        except Exception:
            return None
    except Exception:
        return None


def _probe_audio_codec(video: Path, stream_idx: int) -> str | None:
    """Return the codec_name for the given audio stream index, or None on error."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", f"a:{stream_idx}",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            str(video),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if not out:
            return None
        return out.splitlines()[0].strip()
    except Exception:
        return None


def extract_audio_only(in_video: Path, out_wav: Path, overwrite: bool, sr: int):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # If the WAV already exists and we are not allowed to overwrite, skip extraction
    if out_wav.exists() and not overwrite:
        # caller expects the WAV to be present; skip silently to preserve behavior
        return

    audio_count = _count_audio_streams(in_video)
    if audio_count <= 0:
        # nothing to extract; create a short silent wav to avoid downstream errors
        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate={sr}",
            "-t", "0.1",
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]
        _run(cmd)
        return

    # choose the last stream (likely processed)
    stream_idx = audio_count - 1
    map_spec = f"0:a:{stream_idx}"

    # probe the stream properties so we extract without unnecessary resampling/downmixing
    props = _probe_audio_stream_props(in_video, stream_idx)
    if props is not None:
        channels, sample_rate = props
    else:
        # fallback to provided sr and mono
        channels, sample_rate = 1, sr

    # Build ffmpeg extraction command that preserves the stream's channels and sample rate
    codec = _probe_audio_codec(in_video, stream_idx)
    if codec is not None and codec.startswith("pcm"):
        # If the stream is PCM (lossless), copy the stream into the WAV container to preserve exact bytes
        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_video),
            "-vn", "-map", map_spec,
            "-c:a", "copy",
            str(out_wav),
        ]
    else:
        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_video),
            "-vn", "-map", map_spec,
            "-ac", str(channels), "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]
    _run(cmd)

def _apply_snr_white(in_path: Path, out_video: Path, severity: int, overwrite: bool):
    snr_levels = {1: 20, 2: 15, 3: 10, 4: 5, 5: 0}
    snr_db = snr_levels[severity]

    # gain to apply to noise relative to signal
    noise_gain = 10 ** (-snr_db / 20.0)

    pre = _pre_af()

    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=1.0:r=44100",
        "-filter_complex",
        f"[0:a]{pre},aresample=44100[a0];"
        f"[1:a]asetpts=N/SR/TB,volume={noise_gain},aresample=44100[ns];"
        "[a0][ns]amix=inputs=2:duration=shortest:normalize=0,"
        "alimiter=limit=0.98[aout]",
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC, "-ac", "1",
        "-shortest", str(out_video),
    ]
    _run(cmd)


def _af_reverb(severity: int) -> str:
    """Return ffmpeg audio filter string for reverb (aecho) by severity."""
    # Updated mappings: severity 1 unchanged; severity 2/3 shifted up;
    # severities 4 and 5 made more aggressive for stronger reverb.
    delays_map = {
        1: "80|120",
        2: "100|150",
        3: "120|200",
        4: "180|300",
        5: "260|420",
    }
    decays_map = {
        1: "0.25|0.20",
        2: "0.45|0.30",
        3: "0.65|0.40",
        4: "0.80|0.55",
        5: "0.90|0.70",
    }
    delays = delays_map[severity]
    decays = decays_map[severity]
    return f"aecho=0.8:0.9:{delays}:{decays}"


def _af_bitcrushing(severity: int) -> str:
    # Pregain pushes signal into distortion
    pregain = {1: 26, 2: 30, 3: 36, 4: 48, 5: 60}[severity]

    # tanh drive controls hardness of bitcrushing
    drive = {1: 5, 2: 8.0, 3: 10.0, 4: 16.0, 5: 32.0}[severity]

    # Bitcrush for extra degradation (fewer bits -> more obvious degradation)
    bits = {1: 7, 2: 6, 3: 4, 4: 1, 5: 1}[severity]

    if severity < 4:
        return (
            f"volume={pregain}dB,"
            f"aeval=0.5*tanh({drive}*val(0)),"
            f"acrusher=bits={bits}:mix=1,"
        )
    else:
        samples = 20 if severity == 5 else 8
        return (
            f"volume={pregain}dB,"
            f"aeval=0.5*tanh({drive}*val(0)),"
            f"acrusher=bits={bits}:samples={samples}:mix=1,"
        )



def _af_bandlimit(severity: int) -> str:
    """Return ffmpeg audio filter string for bandlimit (highpass+lowpass).

    This implementation is intentionally aggressive so the effect is audible:
    - tighter low/high cutoffs at higher severity
    - downsampling (resample) to a low rate for loss of high-frequency content
    - optional bitcrush (acrusher) at mid+ severities to add distortion
    """
    # Tuned cutoffs: severity 1=mild ... 5=severe
    # Updated mappings: keep severity 1 unchanged; severity 2 uses old severity-3;
    # severity 3 uses old severity-5; severities 4 and 5 made more aggressive.
    low = {1: 100, 2: 300, 3: 500, 4: 800, 5: 1000}[severity]
    high = {1: 6000, 2: 3000, 3: 1500, 4: 1000, 5: 500}[severity]

    # Aggressive resampling to remove HF content (lower sample rate for higher severity)
    resample_map = {1: 16000, 2: 8000, 3: 4000, 4: 2000, 5: 1000}
    rs = resample_map[severity]

    af_parts = [f"highpass=f={low}", f"lowpass=f={high}", f"aresample={rs}"]

    # Add a bitcrush/distortion at mid+ severities to make the result more 'corrupted'
    if severity >= 3:
        bits = {3: 8, 4: 6, 5: 4}[severity]
        af_parts.append(f"acrusher=bits={bits}:mix=1")

    # Upsample back to a reasonable working rate to avoid downstream errors
    af_parts.append("aresample=16000")

    return ",".join(af_parts)


def _apply_af_filter(in_path: Path, out_video: Path, af_str: str, overwrite: bool):
    """Run ffmpeg applying the provided audio filter string (prepends normalization)."""
    pre = _pre_af()
    af_full = f"{pre},{af_str}"
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "copy",
        "-af", af_full,
        "-c:a", AUDIO_CODEC, "-ac", "1",
        "-shortest", str(out_video),
    ]
    _run(cmd)


def _apply_reverb(in_path: Path, out_video: Path, severity: int, overwrite: bool):
    af = _af_reverb(severity)
    return _apply_af_filter(in_path, out_video, af, overwrite)


def _apply_bitcrushing(in_path: Path, out_video: Path, severity: int, overwrite: bool):
    af = _af_bitcrushing(severity)
    return _apply_af_filter(in_path, out_video, af, overwrite)


def _apply_bandlimit(in_path: Path, out_video: Path, severity: int, overwrite: bool):
    """Apply bandlimit by band-pass filtering and mixing in white noise.

    This uses a filter_complex with a second (lavfi) noise input so the
    effect is clearly audible even after downstream re-encoding.
    """
    af = _af_bandlimit(severity)

    # noise amplitude increases with severity
    noise_amp_map = {1: 0.01, 2: 0.05, 3: 0.20, 4: 0.30, 5: 0.60}
    noise_amp = noise_amp_map[severity]

    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude=1.0:r=44100",
        "-filter_complex",
        # apply normalization+bandlimit to original, prepare noise, mix, limiter
        f"[0:a]{_pre_af()},{af},aresample=44100[a0];"
        f"[1:a]asetpts=N/SR/TB,volume={noise_amp},aresample=44100[ns];"
        "[a0][ns]amix=inputs=2:duration=shortest:normalize=0,alimiter=limit=0.98[aout]",
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC, "-ac", "1",
        "-shortest", str(out_video),
    ]
    _run(cmd)


def _apply_mp3(in_path: Path, out_video: Path, severity: int, overwrite: bool):
    # Use noticeably lower bitrates for stronger audible degradation at higher severity
    br = {1: "96k", 2: "48k", 3: "16k", 4: "8k", 5: "4k"}[severity]

    # Add a small bandlimit + downsample before MP3 encode to increase perceptible
    # artifacts. The normalization from _pre_af() remains but we append resampling
    # and low/high cutoffs tuned by severity.
    pre = _pre_af()
    band_map = {
        1: (200, 6000),
        2: (400, 3000),
        3: (600, 1500),
        4: (800, 750),
        5: (1000, 375),
    }
    hp, lp = band_map[severity]
    # downsample to force MP3 encoder to work with fewer HF details
    target_sr = {1: 22050, 2: 16000, 3: 8000, 4: 5000, 5: 3000}[severity]
    mp3_af = f"{pre},highpass=f={hp},lowpass=f={lp},aresample={target_sr}"

    with tempfile.TemporaryDirectory() as td:
        tmp_mp3 = Path(td) / "tmp.mp3"

        # mix a small amount of noise into the MP3 to make compression artifacts
        # more audible; noise amplitude grows with severity
        noise_amp_map = {1: 0.02, 2: 0.10, 3: 0.30, 4: 0.45, 5: 0.80}
        noise_amp = noise_amp_map[severity]

        # 1) Extract audio -> normalize -> bandlimit/resample -> mix noise -> encode low-bitrate MP3
        _run([
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude=1.0:r=44100",
            "-filter_complex",
            # normalize+bandlimit the original, prepare noise, mix and limiter
            f"[0:a]{mp3_af},aresample=44100[a0];"
            f"[1:a]asetpts=N/SR/TB,volume={noise_amp},aresample=44100[ns];"
            "[a0][ns]amix=inputs=2:duration=shortest:normalize=0,alimiter=limit=0.95[aout]",
            "-map", "[aout]",
            "-vn",
            "-ac", "1",
            "-c:a", "libmp3lame", "-b:a", br,
            str(tmp_mp3),
        ])

        # 2) Mux original video + MP3 audio, re-encode audio to desired codec (PCM in MKV)
        _run([
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-i", str(tmp_mp3),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", AUDIO_CODEC, "-ac", "1",
            "-shortest",
            str(out_video),
        ])


def _get_media_duration(path: Path) -> float | None:
    """Return the duration in seconds for the media file, or None on error."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if not out:
            return None
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


def _apply_compress_and_silence(in_path: Path, out_video: Path, severity: int, overwrite: bool):
    """Speed up (compress) the audio by a severity-controlled factor, then
    prepend silence such that the resulting audio duration equals the original.

    The video stream is copied unchanged; audio is re-encoded to AUDIO_CODEC mono.
    """
    # map severity -> tempo factor (>1 means faster / shorter)
    tempo_map = {1: 1.3, 2: 1.6, 3: 2.00, 4: 3.0, 5: 4.0}
    factor = tempo_map.get(max(1, min(5, severity)), 2.00)

    # obtain original duration
    orig_dur = _get_media_duration(in_path)
    if orig_dur is None:
        # fallback: attempt a reasonable default (no silence)
        orig_dur = 0.0

    # sped audio duration after tempo change
    sped_dur = orig_dur / factor if orig_dur > 0 else 0.0
    silence_dur = max(0.0, orig_dur - sped_dur)

    # if silence is effectively zero, just speed the audio and leave it
    # (this keeps a simple pipeline for short inputs)
    if silence_dur <= 0.001:
        # apply atempo only and mux
        # note: atempo supports 0.5-2.0; our factors are in that range
        af = f"atempo={factor},aresample=44100"
        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy",
            "-af", af,
            "-c:a", AUDIO_CODEC, "-ac", "1",
            "-shortest", str(out_video),
        ]
        _run(cmd)
        return

    # create a silence input using lavfi and concat it with the sped audio
    # Inputs: 0 -> original file, 1 -> generated silence
    # We'll apply atempo to the original audio, then concat [silence][sped_audio]
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate=44100:d={silence_dur}",
        "-filter_complex",
        # apply atempo to original audio (input 0), ensure both streams have same sample rate
        f"[0:a]atempo={factor},aresample=44100[a1];"
        f"[1:a]aresample=44100[a0];"
        # concat silence (a0) + sped audio (a1)
        f"[a0][a1]concat=n=2:v=0:a=1[aout]",
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", AUDIO_CODEC, "-ac", "1",
        "-shortest",
        str(out_video),
    ]

    _run(cmd)


def _apply_temporal_jitter(in_path: Path, out_video: Path, severity: int, overwrite: bool):
    """Apply temporal jitter by splitting audio into short segments and permuting them.

    Strategy:
    - Extract audio segments into a temporary directory using ffmpeg's segment muxer.
    - Permute the list of segments according to severity (mild -> small local swaps, severe -> full shuffle).
    - Concat the permuted segments back into a single WAV using the concat demuxer and re-mux with the original video.

    This keeps the implementation dependency-free (uses ffmpeg) and is robust for different input lengths.
    """
    # severity -> target segment length (seconds): higher severity -> longer segments -> stronger temporal disruption
    seg_len_map = {1: 0.1, 2: 0.15, 3: 0.20, 4: 0.40, 5: 0.80}
    seg_len = seg_len_map.get(max(1, min(5, severity)), 0.2)

    # create temp dir to hold segments and intermediate files
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # 1) extract audio segments as WAV files
        seg_pattern = str(td_path / "seg%05d.wav")
        # Use a reasonable working sample rate and mono to make concat simple
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-vn",
            "-map", "0:a:0",
            "-f", "segment",
            "-segment_time", str(seg_len),
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            seg_pattern,
        ]
        _run(extract_cmd)

        # collect generated segments
        segs = sorted(td_path.glob("seg*.wav"))
        if not segs:
            # fallback: no audio segments produced (e.g., no audio) -> simply copy original
            _run([
                "ffmpeg", "-y" if overwrite else "-n",
                "-i", str(in_path),
                "-map", "0:v:0", "-map", "0:a:0",
                "-c:v", "copy",
                "-c:a", AUDIO_CODEC, "-ac", "1",
                "-shortest", str(out_video),
            ])
            return

        n = len(segs)

        # Determine permutation based on severity
        perm = list(range(n))
        if severity == 1:
            # tiny local swaps: swap 1% of adjacent pairs (at least 1 if n>1)
            swaps = max(1, n // 100) if n > 1 else 0
            for _ in range(swaps):
                i = random.randint(0, n - 2)
                perm[i], perm[i + 1] = perm[i + 1], perm[i]
        elif severity == 2:
            # small local swaps: 5% of adjacent pairs
            swaps = max(1, n * 5 // 100) if n > 1 else 0
            for _ in range(swaps):
                i = random.randint(0, n - 2)
                perm[i], perm[i + 1] = perm[i + 1], perm[i]
        elif severity == 3:
            # moderate randomness: perform a number of random swaps (~20%)
            swaps = max(1, n * 20 // 100)
            for _ in range(swaps):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                perm[i], perm[j] = perm[j], perm[i]
        elif severity == 4:
            # strong disruption: shuffle half of segments
            half = n // 2
            indices = list(range(n))
            subset = indices[:half]
            random.shuffle(subset)
            for k, idx in enumerate(subset):
                perm[k] = subset[k]
        else:
            # severity 5: full shuffle
            random.shuffle(perm)

        # build ordered list of segment file paths according to permutation
        perm_segs = [str(segs[i]) for i in perm]

        # write concat list file
        concat_list = td_path / "concat_list.txt"
        with concat_list.open("w") as f:
            for p in perm_segs:
                # paths must be escaped properly; use single quotes in ffmpeg concat list
                f.write(f"file '{p}'\n")

        # concat back into a single WAV
        out_wav = td_path / "jittered.wav"
        _run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(out_wav),
        ])

        # 3) Mux jittered audio with original video
        _run([
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-i", str(out_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", AUDIO_CODEC, "-ac", "1",
            "-shortest",
            str(out_video),
        ])


def apply_audio_corruption(in_path: Path, out_video: Path, corruption: str, severity: int, overwrite: bool) -> Path:
    """Apply the requested audio corruption by delegating to per-corruption helpers.

    Returns the actual path written (may have a different suffix when OUT_EXTENSION is set).
    """
    severity = max(1, min(5, severity))
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # If OUT_EXTENSION is set, ensure the output file uses that extension
    if OUT_EXTENSION is not None:
        effective_out = out_video.with_suffix(OUT_EXTENSION)
        effective_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        effective_out = out_video

    if corruption == "snr_white":
        _apply_snr_white(in_path, effective_out, severity, overwrite)
    elif corruption == "mp3":
        _apply_mp3(in_path, effective_out, severity, overwrite)
    elif corruption == "reverb":
        _apply_reverb(in_path, effective_out, severity, overwrite)
    elif corruption == "bitcrushing":
        _apply_bitcrushing(in_path, effective_out, severity, overwrite)
    elif corruption == "bandlimit":
        _apply_bandlimit(in_path, effective_out, severity, overwrite)
    elif corruption == "compress":
        _apply_compress_and_silence(in_path, effective_out, severity, overwrite)
    elif corruption == "jitter":
        _apply_temporal_jitter(in_path, effective_out, severity, overwrite)
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    return effective_out


def _ensure_video_audio_mono(video: Path, logger: Optional[callable] = None):
    """Ensure the given video file has a single mono audio stream.

    This remuxes/re-encodes the audio to AAC mono and replaces the original file
    atomically. If the video has no audio streams, the function does nothing.
    """
    try:
        count = _count_audio_streams(video)
        if count == 0:
            # nothing to do
            return

        # Prefer the last audio stream (likely the processed one) when remuxing
        if count > 1:
            audio_map = f"0:a:{count-1}"
        else:
            audio_map = "0:a:0"

        # If user requested lossless PCM output (AUDIO_CODEC == 'pcm_s16le') and the
        # chosen stream is already PCM and mono, skip remux/re-encode so the audio bytes
        # remain identical and can be copied into a WAV exactly.
        try:
            props = _probe_audio_stream_props(video, count - 1)
            codec = _probe_audio_codec(video, count - 1)
            if codec is not None and codec.startswith("pcm") and props is not None:
                ch, _ = props
                if ch == 1 and AUDIO_CODEC == "pcm_s16le":
                    if logger:
                        logger(f"[INFO] Audio already PCM mono, skipping remux for {video}")
                    return
        except Exception:
            # ignore probe errors and continue with remux
            pass

        # create a temporary output in the same directory to avoid cross-device moves
        with tempfile.NamedTemporaryFile(delete=False, dir=str(video.parent), suffix=video.suffix) as tf:
            tmp_path = Path(tf.name)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            # map video and the chosen audio stream (we downmix to mono)
            "-map", "0:v:0", "-map", audio_map,
            "-c:v", "copy",
            "-c:a", AUDIO_CODEC, "-ac", "1",
            str(tmp_path),
        ]
        _run(cmd)

        # replace original with tmp
        tmp_path.replace(video)
        if logger:
            logger(f"[INFO] Enforced mono audio for {video}")
    except Exception as e:
        # best-effort: don't crash the whole pipeline on a post-process failure
        if logger:
            logger(f"[WARN] Failed to enforce mono for {video}: {e}")
        return


def main():
    ap = argparse.ArgumentParser("Apply ALL audio perturbations to a video directory.")
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="Base output directory.")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--audio_sr", type=int, default=16000)
    # Note: lossless PCM-in-MKV is the default and cannot be disabled. We therefore
    # do not provide a CLI flag to turn it off.
    args = ap.parse_args()

    _ffmpeg()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted(list(_iter_videos(videos_dir, args.recursive)))
    if not vids:
        raise RuntimeError(f"No videos found in {videos_dir}")

    for corr in AUDIO_CORRUPTIONS:
        combo_root = out_dir / f"A={corr}_S={args.severity}"
        videos_out = combo_root
        audio_only_out = videos_out / "audio_only"
        videos_out.mkdir(parents=True, exist_ok=True)
        audio_only_out.mkdir(parents=True, exist_ok=True)

        for vid in vids:
            rel = vid.relative_to(videos_dir) if videos_dir.is_dir() else Path(vid.name)
            out_video = videos_out / rel
            out_video.parent.mkdir(parents=True, exist_ok=True)

            if out_video.exists() and not args.overwrite:
                continue

            # apply_audio_corruption may change the output file's suffix (e.g., when
            # OUT_EXTENSION is set in lossless mode). Capture the actual path it writes.
            processed_video = apply_audio_corruption(vid, out_video, corr, args.severity, args.overwrite)

            # Ensure the processed video's audio is mono before extracting WAV
            _ensure_video_audio_mono(processed_video, logger=print)

            # Produce an MP4 copy for downstream use (AAC mono). We keep the MKV as
            # the canonical, lossless container for WAV extraction, then remove it.
            mp4_copy = out_video.with_suffix('.mp4')
            mp4_copy.parent.mkdir(parents=True, exist_ok=True)
            try:
                _run([
                    "ffmpeg", "-y" if args.overwrite else "-n",
                    "-i", str(processed_video),
                    "-map", "0:v:0", "-map", "0:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-ac", "1", "-b:a", "128k",
                    str(mp4_copy),
                ])
            except Exception as e:
                print(f"[WARN] Failed to create MP4 copy for {processed_video}: {e}")

            out_wav = (audio_only_out / rel).with_suffix(".wav")
            extract_audio_only(processed_video, out_wav, args.overwrite, args.audio_sr)

            # Remove the temporary MKV (lossless) if an MP4 copy was created; keep the MP4
            # for downstream use. Best-effort deletion: warn on error but don't stop.
            try:
                if processed_video.exists():
                    processed_video.unlink()
                    print(f"[INFO] Removed temporary lossless file {processed_video}")
            except Exception as e:
                print(f"[WARN] Failed to remove temporary file {processed_video}: {e}")

        print(f"[OK] Finished audio corruption: {combo_root}")

    print("[DONE] All audio perturbations applied.")


if __name__ == "__main__":
    main()

