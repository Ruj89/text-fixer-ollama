import os
import sys
import ollama

CHUNK_WORD_LIMIT = 2000
MODEL_NAME = "gemma3n"  # Sostituisci con il modello che hai installato

def dividi_in_blocchi(testo, parole_per_blocco):
    parole = testo.split()
    blocchi = [
        ' '.join(parole[i:i + parole_per_blocco])
        for i in range(0, len(parole), parole_per_blocco)
    ]
    return blocchi

def correggi_blocco_ollama(testo_blocco):
    prompt = (
        "Correggi solo gli errori ortografici, gli accenti sbagliati e problemi di formattazione. "
        "Non cambiare stile, tono o il modo in cui è scritto. Ecco il testo:\n\n"
        f"{testo_blocco}\n\nTesto corretto:"
    )

    response = ollama.chat(model=MODEL_NAME, messages=[
        {"role": "user", "content": prompt}
    ])

    return response['message']['content'].strip()

def correggi_file(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"Il file '{input_path}' non esiste.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        testo = f.read()

    blocchi = dividi_in_blocchi(testo, CHUNK_WORD_LIMIT)
    testo_finale = ""

    for idx, blocco in enumerate(blocchi):
        print(f"Correggo blocco {idx + 1} di {len(blocchi)}...")
        try:
            testo_corretto = correggi_blocco_ollama(blocco)
            testo_finale += testo_corretto + "\n\n"
        except Exception as e:
            print(f"Errore nel blocco {idx + 1}: {e}")
            testo_finale += blocco + "\n\n"  # fallback: testo originale

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(testo_finale.strip())

    print(f"\n✅ Testo corretto salvato in: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python correggi_testo_chunk_ollama.py input.txt output.txt")
    else:
        correggi_file(sys.argv[1], sys.argv[2])
