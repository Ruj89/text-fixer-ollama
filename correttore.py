#!/usr/bin/env python3
import os
import sys
import difflib
import shutil

import ollama
import spacy

# ─── Parameters ─────────────────────────────────────────────────────────────
CHUNK_CHARS_LIMIT  = 2500           # Max character count for each text chunk
OUTPUT_PERCENT     = 50             # % of the chunk that will be written to output
OVERLAP_PERCENT    = (100 - OUTPUT_PERCENT)  # % of overlap between chunks
MODEL_NAME         = "gemma3n"      # Name of the Ollama model
MISMATCH_THRESHOLD = 0.03           # Threshold for text similarity mismatch
MAX_OVERLAP_SIZE = CHUNK_CHARS_LIMIT * OVERLAP_PERCENT / 100
# ─────────────────────────────────────────────────────────────────────────────

# Set up a blank multilingual SpaCy model with sentence segmentation
nlp = spacy.blank("xx")
nlp.add_pipe("sentencizer")

def backup_file(src_path, step):
    folder_file = os.path.dirname(src_path)
    name_file = os.path.basename(src_path)
    name, est = os.path.splitext(name_file)
    new_name = f"{step}_{name}{est}"
    dest_path = os.path.join(folder_file, new_name)
    shutil.copy2(src_path, dest_path)

# Truncates the output file from the given index onward
def truncate_file(path: str, cutoff_index: int):
    with open(path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0)
        f.write(content[:-cutoff_index])
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
def find_similarity(sentences1, sentences2):
    best_similarity = 0
    best_sentences1_start_index = 0
    best_sentences2_start_index = 0
    best_sentences2_end_index = 0
    for sentences1_start_index in range(len(sentences1)): 
        s1 = " ".join([s.text for s in sentences1[sentences1_start_index:]])
        for sentences2_start_index in range(len(sentences2)): 
            for sentences2_end_index in range(sentences2_start_index, len(sentences2)):
                s2 = " ".join([s.text for s in sentences2[sentences2_start_index:sentences2_end_index]])
                similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
                if best_similarity < similarity:
                    best_similarity = similarity
                    best_sentences1_start_index = sentences1_start_index
                    best_sentences2_start_index = sentences2_start_index
                    best_sentences2_end_index = sentences2_end_index
    return best_similarity, best_sentences1_start_index, best_sentences2_start_index, best_sentences2_end_index

# Main processing logic for correcting the full file
def correct_file(input_path: str, output_path: str):
    step=0
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

    # Data structures to track chunk progress and positions
    iteration_data = [{
        "paragraph": paragraph,
        "corrected_segmented": [],
        "tail_overlap_index_start": 0, 
        "written_indexes": {"start" : 0, "end": 0},
    } for paragraph in paragraphs]

    idx = 0
    while idx < len(iteration_data):
        matching_found = False
        similarity_retry = 0

        while(not matching_found):
            print(f"📝 Correcting chunk {idx}/{len(paragraphs)} …")
            corrected = correct_chunk_with_ollama(iteration_data[idx]["paragraph"])
            iteration_data[idx]["corrected_segmented"] = [s for s in nlp(corrected).sents]

            if idx == 0:
                # First chunk does not need overlap matching
                matching_found = True
            else:
                last_overlap = iteration_data[idx - 1]["corrected_segmented"][iteration_data[idx - 1]["tail_overlap_index_start"]:]
                similarity, old_start_index, start_index, end_index = find_similarity(last_overlap, iteration_data[idx]["corrected_segmented"])
                print(f"🧬 Best similarity [{old_start_index}:{len(iteration_data[idx]["corrected_segmented"])}-1]/{len(iteration_data[idx]["corrected_segmented"])} [{start_index}:{end_index}]/{len(iteration_data[idx]["corrected_segmented"])} : {similarity}\n")
                if similarity >= 1 - MISMATCH_THRESHOLD:
                    print(f"✅ Similarity accepted\n")
                    last_written_overlap = iteration_data[idx - 1]["corrected_segmented"][iteration_data[idx - 1]["tail_overlap_index_start"]:iteration_data[idx - 1]["written_indexes"]["end"]]
                    _, _, _, iteration_data[idx]["written_indexes"]["start"] = find_similarity(last_written_overlap, iteration_data[idx]["corrected_segmented"])
                    iteration_data[idx]["written_indexes"]["start"] += 1
                    matching_found = True
                
                if not matching_found:
                    similarity_retry += 1
                    print(f"🧪 Similarity not found, best chance:")
                    print("─────────────────────────────────────────────────────────────────────")
                    print(" ".join([s.text for s in last_overlap]))
                    print("─────────────────────────────────────────────────────────────────────")
                    print("\nvs\n")
                    print("─────────────────────────────────────────────────────────────────────")
                    print(" ".join([s.text for s in iteration_data[idx]["corrected_segmented"]]))
                    print("─────────────────────────────────────────────────────────────────────")
                    if similarity_retry >= 3:
                        # If similarity match fails too many times, go back one chunk
                        print("❌ Similarity retied too many times, reverting to the chunk before.")
                        backup_file(output_path, step)
                        step += 1
                        idx -= 1
                        if idx > 0:
                            start_write_char = iteration_data[idx]["corrected_segmented"][iteration_data[idx]["written_indexes"]["start"]].start_char
                            if idx > 0:
                                start_write_char -= 1
                            end_write_char = iteration_data[idx]["corrected_segmented"][iteration_data[idx]["written_indexes"]["end"]].end_char
                            truncate_file(output_path, end_write_char - start_write_char)
                        else:
                            os.remove(output_path)
                        break 

        if matching_found:
            # Save overlap for the next chunk
            i = 0
            for i in reversed(range(len(iteration_data[idx]["corrected_segmented"]))):
                if iteration_data[idx]["corrected_segmented"][-1].end_char - iteration_data[idx]["corrected_segmented"][i].start_char + 1 <= MAX_OVERLAP_SIZE:    
                    iteration_data[idx]["tail_overlap_index_start"] = i
                else:
                    break
            
            if idx < len(iteration_data) - 1:
                overlap_sentences_count = len(iteration_data[idx]["corrected_segmented"]) - iteration_data[idx]["tail_overlap_index_start"]
                iteration_data[idx]["written_indexes"]["end"] = iteration_data[idx]["tail_overlap_index_start"] + round(overlap_sentences_count/2)
            else:
                iteration_data[idx]["written_indexes"]["end"] = len(iteration_data[idx]["corrected_segmented"]) - 1

                
            start_write_char = iteration_data[idx]["corrected_segmented"][iteration_data[idx]["written_indexes"]["start"]].start_char
            if iteration_data[idx]["written_indexes"]["start"] > 0:
                start_write_char -= 1
            end_write_char = iteration_data[idx]["corrected_segmented"][iteration_data[idx]["written_indexes"]["end"]].end_char
            writing_segment = corrected[start_write_char:end_write_char]

            if idx > 0:
                backup_file(output_path, step)
            step += 1


            # Write corrected chunk to output file
            with open(output_path, "a", encoding="utf-8") as out:
                out.write(writing_segment)
            print(f"✅ Done! Output in: {output_path}")
                

            idx += 1

# ─── Command Line Interface ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python correttore.py input.txt output.txt")
    else:
        if os.path.exists(sys.argv[2]):
            os.remove(sys.argv[2])
        correct_file(sys.argv[1], sys.argv[2])
