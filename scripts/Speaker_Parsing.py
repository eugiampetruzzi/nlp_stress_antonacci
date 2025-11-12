"""
This script prepares the raw, diarized transcripts for NLP analysis.

It corresponds to the part of the methods where interviewer speech is
removed. The script reads from a folder of diarized transcripts
(e.g., 'WhisperX_Carina_Revised') that have been cleaned by the GPT-4o
process to have 'Interviewer' and 'Participant' labels.

It parses each file and extracts *only* the lines spoken by 'Participant',
concatenates them into a single block of text, and saves this
participant-only speech to a new .txt file in the output folder
(e.g., 'Participant_Parse').
"""

import os
import re

def extract_participant_speech(input_file_path, output_file_path):
    """
    Reads a diarized transcript, extracts lines from 'Participant',
    and writes them as a single paragraph to a new file.
    """
    participant_lines = []

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip():
                    continue

                # regex to capture: [start_time] [end_time] [Speaker]: [Utterance]
                match = re.match(r'^(\d+\.\d+)\s+(\d+\.\d+)\s+(\w+):\s*(.+)$', line.strip())
                
                if match:
                    start_time, end_time, speaker, utterance = match.groups()
                    if speaker.lower() == 'participant':
                        participant_lines.append(utterance.strip())

        full_paragraph = ' '.join(participant_lines)

        # clean up extra spacing around punctuation
        full_paragraph = re.sub(r'\s+([,.?!])', r'\1', full_paragraph)

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(full_paragraph + '\n')
            
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
    except Exception as e:
        print(f"An error occurred processing {input_file_path}: {e}")

def main():
    """
    Main function to run the parsing process for all files in a folder.
    """
    # define relative paths
    input_folder = os.path.join(
        "data", "WhisperX_Carina_Revised"
    )
    output_folder = os.path.join(
        "data", "Participant_Parse"
    )

    # create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        transcript_files = sorted(os.listdir(input_folder))
        print(f"Found {len(transcript_files)} files in {input_folder}.")
    except FileNotFoundError:
        print(f"Error: Input folder not found at {input_folder}")
        print("Please check the 'input_folder' path.")
        return

    file_count = 0
    for filename in transcript_files:
        if not filename.endswith(".txt"):
            continue
            
        input_file_path = os.path.join(input_folder, filename)
        
        # assumes filename format 'ELS_XXX...'
        subject_id = filename[:7]
        output_file_name = f"{subject_id}_participant_parsing.txt"
        output_file_path = os.path.join(output_folder, output_file_name)

        extract_participant_speech(input_file_path, output_file_path)
        file_count += 1

    print(f"\nSuccessfully processed {file_count} transcript files.")
    print(f"Output saved to: {output_folder}")

if __name__ == "__main__":
    main()