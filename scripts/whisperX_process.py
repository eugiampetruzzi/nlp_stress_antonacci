"""
This script is designed to be called by a cluster batch job (e.g., a shell script).
It processes a SINGLE audio file using the WhisperX pipeline.

It takes command-line arguments for the audio file, subject ID, and output
directory, then performs the following steps:
1.  Converts the audio file to .wav if needed (e.g., from .wma).
2.  Transcribes the audio using WhisperX (large-v2).
3.  Aligns the transcription for accurate timestamps.
4.  Performs speaker diarization using PyAnnote.
5.  Assigns speaker labels to the aligned transcription.
6.  Writes the final diarized transcript to a .txt file in the output directory.

All errors are printed to stderr to be captured by cluster logs.
"""

import os
import sys
import argparse
import subprocess
import torch
import whisperx

def convert_to_wav(input_file, output_file):
    """Converts an audio file to WAV format using FFmpeg."""
    try:
        # run ffmpeg command
        subprocess.run(
            ["ffmpeg", "-i", input_file, output_file],
            check=True,       # raise an exception on non-zero exit
            capture_output=True, # capture stdout/stderr
            text=True         # decode output as text
        )
        return True
    except subprocess.CalledProcessError as e:
        # print ffmpeg's error message to stderr for cluster logging
        print(f"FFmpeg conversion failed for {input_file}: {e.stderr}", file=sys.stderr)
        return False

def main():
    # setup argument parser to take inputs from the shell script
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize audio files using WhisperX."
    )
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("sub_id", help="Subject ID")
    parser.add_argument(
        "--pyannote_token", help="PyAnnote Hugging Face Token", required=True
    )
    parser.add_argument(
        "--min_speakers", default=None, help="Minimum number of speakers", type=int
    )
    parser.add_argument(
        "--max_speakers", default=None, help="Maximum number of speakers", type=int
    )
    parser.add_argument(
        "--output_dir", help="Path to output directory", required=True
    )
    args = parser.parse_args()

    audio_file = args.audio_file
    output_dir = args.output_dir
    subject_id = args.sub_id
    pyannote_hf_token = args.pyannote_token
    min_speakers = args.min_speakers
    max_speakers = args.max_speakers

    # check if gpu is available
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    elif torch.backends.mps.is_available(): # for apple silicon
        device = "mps"
        compute_type = "float16"
    else: # for cpu
        device = "cpu"
        compute_type = "int8"

    batch_size = 16 # reduce if low on gpu mem

    # convert .wma to .wav (if needed)
    if audio_file.lower().endswith(".wma"):
        # create a temporary wav file in the /tmp directory
        temp_wav_file = os.path.join("/tmp", f"{subject_id}.wav")
        if convert_to_wav(audio_file, temp_wav_file):
            audio_file = temp_wav_file  # use the converted file
        else:
            print(f"Skipping {audio_file} due to conversion failure.", file=sys.stderr)
            return  # exit, skipping transcription

    try:
        # 1. transcribe with original whisper (batched)
        model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # 2. align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False
        )

        # 3. assign speaker labels - run pyannote
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=pyannote_hf_token, device=device
        )

        # add min/max number of speakers if specified
        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers if min_speakers else None,
            max_speakers=max_speakers if max_speakers else None
        )

        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 4. write output to a subject-specific file
        output_file = os.path.join(
            output_dir, f"{subject_id}_whisperx_transcript.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment["text"].strip()
                line = f"{start_time}\t{end_time}\t{speaker}: {text}\n"
                f.write(line)

    except Exception as e:
        print(f"Error processing {audio_file}: {e}", file=sys.stderr)

    finally:
        # clean up temporary wav file if one was created
        if ('temp_wav_file' in locals() and
            audio_file == temp_wav_file and
            os.path.exists(temp_wav_file)):
            
            os.remove(temp_wav_file)

if __name__ == "__main__":
    main()