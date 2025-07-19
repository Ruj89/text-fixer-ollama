import os
import sys
import ollama
import pysbd

CHUNK_WORD_LIMIT = 2000
MODEL_NAME = "gemma3n"  # Sostituisci con il modello che hai installato

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

def split_blocks(text, words_per_block):
    seg = pysbd.Segmenter(language="en", clean=True)
    blocks = seg.segment(text)
    blocks = concat_strings(blocks, CHUNK_WORD_LIMIT)
    return blocks

def correggi_blocco_ollama(testo_blocco):
    prompt = (
        "Ora ti fornisco una porzione di testo HTML. \n"
        "Correggi solo gli errori ortografici, gli accenti sbagliati e problemi di formattazione utilizzando se possibile il formato HTML. "
        "Del testo non cambiare il modo in cui è scritto. Ecco il testo:\n\n"
        f"{testo_blocco}\n\nTesto corretto:"
    )

    response = ollama.chat(model=MODEL_NAME, messages=[
        {"role": "user", "content": prompt}
    ])

    return response['message']['content'].strip()

def fix_file(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"File '{input_path}' not exists.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        testo = f.read()

    blocchi = split_blocks(testo, CHUNK_WORD_LIMIT)
    testo_finale = ""

    for idx, blocco in enumerate(blocchi):
        print(f"Fixing block {idx + 1} of {len(blocchi)}...")
        try:
            testo_corretto = correggi_blocco_ollama(blocco)
            testo_finale += testo_corretto + "\n\n"
        except Exception as e:
            print(f"Block {idx + 1} error: {e}")
            testo_finale += blocco + "\n\n"  # fallback: testo originale

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(testo_finale.strip())

    print(f"\n✅ Fixed text saved in: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use: python correttore.py input.txt output.txt")
    else:
        fix_file(sys.argv[1], sys.argv[2])
