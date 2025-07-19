#!/usr/bin/env python3
import os
import sys
import difflib

import ollama
import spacy

# ─── Parameters ─────────────────────────────────────────────────────────────
CHUNK_CHARS_LIMIT  = 2500           # Max character count for each text chunk
OUTPUT_PERCENT     = 50             # % of the chunk that will be written to output
OVERLAP_PERCENT    = (100 - OUTPUT_PERCENT)  # % of overlap between chunks
MODEL_NAME         = "gemma3n"      # Name of the Ollama model
MISMATCH_THRESHOLD = 0.03           # Threshold for text similarity mismatch
# ─────────────────────────────────────────────────────────────────────────────

# Set up a blank multilingual SpaCy model with sentence segmentation
nlp = spacy.blank("xx")
nlp.add_pipe("sentencizer")

# Truncates the output file from the given index onward
def truncate_file(path: str, cutoff_index: int):
    with open(path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0)
        f.write(content[:cutoff_index])
        f.truncate()

# Prints the beginning and end of a string to help with debugging
def print_start_end(string, ratio):
    print(f"{string[0:round(len(string)*ratio)]}…{string[-round(len(string)*ratio):round(len(string))]}")

# Splits the text into chunks with overlapping areas for context preservation
def chunk_sentences(string_list, chunk_word_limit, overlap_percent):
    chunk_length = chunk_word_limit - (chunk_word_limit * overlap_percent / 100)

    chunks = []
    current_chunk = []
    current_length = 0
    old_length = 0
    
    for s in string_list:
        substring_list = [s.text for s in nlp(s).sents]
        if len(substring_list) > 0:
            substring_list[-1] += "\n"
        else:
            substring_list = ["\n"]
        for ss in substring_list:
            corrected_len = len(ss) - ss.count("\n")
            if corrected_len > (chunk_word_limit - chunk_length) / 2:
                raise ValueError(f"String too long, increase CHUNK_CHARS_LIMIT: {ss} ({corrected_len} > {chunk_word_limit - chunk_length})")

            if current_length + corrected_len <= chunk_length:
                current_chunk.append(ss)
                current_length += corrected_len
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [ss]
                old_length = current_length 
                current_length = corrected_len

            if old_length <= chunk_word_limit and len(chunks) > 0:
                chunks[-1] += f" {ss}"
                old_length += corrected_len

    if len(current_chunk) > 0:
        chunks.append(" ".join(current_chunk))

    return chunks

# Sends a text chunk to the Ollama model for basic correction (spelling, formatting)
def correct_chunk_with_ollama(chunk: str) -> str:
    prompt = (
        "Ora ti fornisco una porzione di testo.\n"
        "Correggi solo errori ortografici e di battitura, accenti sbagliati e problemi di formattazione se ci sono.\n"
        "Non riformulare frasi e non aggiungere grassetti o corsivi.\n"
        "Originale:\n"
        f"{chunk}\n"
        f"Corretto:\n"
    )
    print_start_end(chunk, 1/20)
    resp = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    message = resp["message"]["content"].strip()
    print_start_end(message, 1/20)
    return message

# Compares overlapping segments and finds the best matching similarity index
def find_similarity(last_overlap, corrected_segmented):
    best_similarity = 0
    best_old_start_index = 0
    best_start_index = 0
    best_end_index = 0
    for old_start_index in range(len(last_overlap)): 
        s1 = " ".join([s.text for s in last_overlap[old_start_index:]])
        for start_index in range(len(corrected_segmented)): 
            for end_index in range(start_index, len(corrected_segmented)):
                s2 = " ".join([s.text for s in corrected_segmented[start_index:end_index]])
                similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
                if best_similarity < similarity:
                    best_similarity = similarity
                    best_old_start_index = old_start_index
                    best_start_index = start_index
                    best_end_index = end_index
    return best_similarity, best_old_start_index, best_start_index, best_end_index

# Main processing logic for correcting the full file
def correct_file(input_path: str, output_path: str):
    if not os.path.isfile(input_path):
        print(f"❌ File '{input_path}' not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    if os.path.exists(output_path):
        os.remove(output_path)

    # Split text into overlapping paragraphs
    paragraphs = full_text.split('\n')
    paragraphs = chunk_sentences(paragraphs, CHUNK_CHARS_LIMIT, OVERLAP_PERCENT)
    max_overlap_size = CHUNK_CHARS_LIMIT * OVERLAP_PERCENT / 100

    # Data structures to track chunk progress and positions
    last_overlap = [None] * len(paragraphs)
    paragraph_file_position_last_char = [0] * len(paragraphs)
    last_segment_end_index = [0] * len(paragraphs)

    idx = 0
    while idx < len(paragraphs):
        paragraph = paragraphs[idx]
        end_segment_index = 0
        matching_found = False
        similarity_retry = 0

        while(not matching_found):
            print(f"📝 Correcting chunk {idx}/{len(paragraphs)} …")
            corrected = correct_chunk_with_ollama(paragraph)
            corrected_segmented = [s for s in nlp(corrected).sents]

            if idx == 0:
                # First chunk does not need overlap matching
                matching_found = True
            else:
                similarity, old_start_index, start_index, end_index = find_similarity(last_overlap[idx-1], corrected_segmented)
                print(f"🧬 Best similarity [{old_start_index}:{len(corrected_segmented)}-1]/{len(corrected_segmented)} [{start_index}:{end_index}]/{len(corrected_segmented)} : {similarity}\n")
                if similarity >= 1 - MISMATCH_THRESHOLD:
                    print(f"✅ Similarity accepted\n")
                    _, _, _, end_segment_index = find_similarity(last_overlap[idx-1][:last_segment_end_index[idx-1]], corrected_segmented)
                    end_segment_index += 1
                    matching_found = True
                
                if not matching_found:
                    similarity_retry += 1
                    print(f"🧪 Similarity not found, best chance:")
                    print("─────────────────────────────────────────────────────────────────────")
                    print(" ".join([s.text for s in last_overlap[idx-1]]))
                    print("─────────────────────────────────────────────────────────────────────")
                    print("\nvs\n")
                    print("─────────────────────────────────────────────────────────────────────")
                    print(" ".join([s.text for s in corrected_segmented]))
                    print("─────────────────────────────────────────────────────────────────────")
                    if similarity_retry >= 3:
                        # If similarity match fails too many times, go back one chunk
                        print("❌ Similarity retied too many times, reverting to the chunk before.")
                        idx -= 1
                        if idx > 0:
                            truncate_file(output_path, paragraph_file_position_last_char[idx - 1])
                        else:
                            os.remove(output_path)
                        break 

        if matching_found:
            # Write corrected chunk to output file
            with open(output_path, "a", encoding="utf-8") as out:
                if end_segment_index == 0:
                    start_write_char = corrected_segmented[end_segment_index].start_char
                else:
                    start_write_char = corrected_segmented[end_segment_index - 1].end_char
                if last_segment_end_index[idx] == len(corrected_segmented) - 1:
                    end_write_char = corrected_segmented[last_segment_end_index[idx]].end_char
                else:
                    end_write_char = corrected_segmented[last_segment_end_index[idx] + 1].start_char

                writing_segment = corrected[start_write_char:end_write_char]
                out.write(writing_segment)
                paragraph_file_position_last_char[idx] = (paragraph_file_position_last_char[idx] if len(paragraph_file_position_last_char)>0 else 0) + len(writing_segment)
            print(f"✅ Done! Output in: {output_path}")
                
            # Save overlap for the next chunk
            for i in reversed(range(len(corrected_segmented))):
                if corrected_segmented[-1].end_char - corrected_segmented[i].start_char + 1 > max_overlap_size:
                    last_overlap[idx] = corrected_segmented[i:]
                    last_segment_end_index[idx] -= i
                    break

            idx += 1

# ─── Command Line Interface ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python correttore.py input.txt output.txt")
    else:
        if os.path.exists(sys.argv[2]):
            os.remove(sys.argv[2])
        correct_file(sys.argv[1], sys.argv[2])
