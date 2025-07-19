#!/usr/bin/env python3
"""
Corregge un TXT a blocchi sovrapposti preservando esattamente
la formattazione (a-capo compresi) e senza inserire newline extra.
"""

import os
import re
import sys
import ollama
import pysbd

# ─── Parametri ───────────────────────────────────────────────────────────────
CHUNK_WORD_LIMIT   = 200          # parole totali (overlap incluso)
OUTPUT_PERCENT     = 75
OVERLAP_PERCENT    = (100 - OUTPUT_PERCENT) / 2  # 12,5 %
MODEL_NAME         = "gemma3n"
MISMATCH_THRESHOLD = 0.01         # 1 %
# ─────────────────────────────────────────────────────────────────────────────

def concat_strings(strings, max_length=1500):
    """
    Concatena un array di stringhe in stringhe di lunghezza massima specificata.

    Args:
        strings (list): Lista di stringhe da concatenare.
        max_length (int): Lunghezza massima delle stringhe concatenate. Default: 1500.

    Returns:
        list: Lista di stringhe concatenate, ciascuna di lunghezza massima max_length.
    """
    result = []
    current_concat = ""

    for string in strings:
        # Se aggiungendo la stringa si supera il limite
        if len(current_concat) + len(string) > max_length:
            result.append(current_concat)  # Aggiungi la concatenazione corrente
            current_concat = string       # Inizia una nuova concatenazione
        else:
            current_concat += " " + string      # Continua a concatenare

    if current_concat:  # Aggiungi l'ultimo blocco, se esiste
        result.append(current_concat)

    return result

def tokenize(text):
    seg = pysbd.Segmenter(language="en", clean=True)
    blocks = seg.segment(text)
    blocks = concat_strings(blocks, CHUNK_WORD_LIMIT)
    return blocks

def join(tokens):
    return ''.join(tokens)

def split_with_overlap(text, total_words, overlap_pct):
    tok       = split_blocks(text, CHUNK_WORD_LIMIT)
    overlap   = max(1, round(total_words * overlap_pct / 100))
    step      = round(total_words * OUTPUT_PERCENT / 100)
    chunks, ranges = [], []
    i = 0
    while i < len(tok):
        start = max(i - overlap, 0)
        end   = min(i + step + overlap, len(tok))
        chunk_tokens = tok[start:end]
        chunks.append(join(chunk_tokens))

        out_start = overlap if start else 0
        out_end   = len(chunk_tokens) - overlap if end != len(tok) else len(chunk_tokens)
        ranges.append((out_start, out_end))
        i += step
    return chunks, ranges

def correct_chunk_with_ollama(chunk: str) -> str:
    prompt = (
        "Ora ti fornisco una porzione di testo HTML. \n"
        "Correggi solo errori ortografici, accenti sbagliati e problemi di formattazione. "
        "Non modificare tag HTML o riformulare frasi.\n"
        "Originale:\n"
        f"{chunk}\n"
        f"Corretto:\n"
    )
    resp = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()

def compare_overlap(a_tokens, b_tokens) -> float:
    length = min(len(a_tokens), len(b_tokens))
    if length == 0:
        return 1.0
    mism = sum(
        1 for x, y in zip(a_tokens[:length], b_tokens[:length])
        if x.strip() != y.strip()
    )
    return mism / length

def correct_file(input_path: str, output_path: str):
    if not os.path.isfile(input_path):
        print(f"❌ File '{input_path}' not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks, ranges = split_with_overlap(full_text, CHUNK_WORD_LIMIT, OVERLAP_PERCENT)
    corrected_chunks, kept_slices = [], []

    # Passaggio 1: correzione e append immediato (senza newline extra)
    for idx, (chunk, (s, e)) in enumerate(zip(chunks, ranges), 1):
        print(f"📝 Correcting chunk {idx}/{len(chunks)} …")
        try:
            corrected = correct_chunk_with_ollama(chunk)
        except Exception as err:
            print(f"   ⚠️  Error: {err} — using original")
            corrected = chunk

        corrected_chunks.append(corrected)
        tokens = tokenize(corrected)
        kept   = join(tokens[s:e])
        kept_slices.append(kept)

        with open(output_path, "a", encoding="utf-8") as out:
            out.write(kept)                 # <── niente "\n\n"

    # Passaggio 2: verifica overlap e rigenera se >1 %
    overlap_len = max(1, round(CHUNK_WORD_LIMIT * OVERLAP_PERCENT / 100))
    for i in range(1, len(corrected_chunks)):
        end_prev   = tokenize(corrected_chunks[i - 1])[-overlap_len:]
        start_curr = tokenize(corrected_chunks[i])[:overlap_len]
        mismatch   = compare_overlap(end_prev, start_curr)

        if mismatch > MISMATCH_THRESHOLD:
            print(f"🔄  Mismatch {mismatch:.2%} fra chunk {i} e {i+1} — rigenero …")
            for j in (i - 1, i):
                try:
                    regen = correct_chunk_with_ollama(chunks[j])
                    corrected_chunks[j] = regen
                    s, e = ranges[j]
                    kept_slices[j] = join(tokenize(regen)[s:e])
                except Exception as err:
                    print(f"   ⚠️  Regen error on chunk {j+1}: {err}")

    # Passaggio 3: riscrittura finale senza separatori aggiuntivi
    print("\n💾 Rewriting final output …")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(''.join(kept_slices))      # <── concatenazione pura
    print(f"✅ Done! Output in: {output_path}")

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python correct_text_overlap_check.py input.txt output.txt")
    else:
        if os.path.exists(sys.argv[2]):
            os.remove(sys.argv[2])           # ripartenza pulita
        correct_file(sys.argv[1], sys.argv[2])
