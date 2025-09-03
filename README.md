# correttore.py

*A chunk-based spelling & formatting corrector powered by Ollama*

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [License](#license)
7. [Author](#author)

<a id="features"></a>

## Features

* **Context-aware chunking** – splits long files into overlapping text blocks to preserve context while respecting model token limits.
* **Multilingual support** – uses a blank spaCy *xx* pipeline for lightweight sentence segmentation in any language.
* **LLM-powered correction** – each chunk is sent to an Ollama model (`gemma3n` by default) for plain spelling, accent and basic formatting fixes **without** re-writing style or meaning.
* **Smart overlap merge** – similarity matching (via `difflib`) removes duplicate sentences between consecutive chunks.
* **CLI-ready** – run the script directly from the command line with a single command.

<a id="quick-start"></a>

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Ruj89/text-fixer-ollama
cd text-fixer-ollama

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt      # or: pip install ollama spacy

# Make sure the Ollama server is running, then pull the model you want
ollama serve             # starts the local Ollama daemon (if not already running)
ollama pull gemma3n       # download the default model

# Correct a file
python correttore.py input.txt output.txt
```

<a id="installation"></a>

## Installation

1. **Prerequisites**

   * Python ≥ 3.9 (tested with 3.11)
   * [Ollama](https://ollama.ai) installed and the `ollama` CLI available in your `$PATH`.
   * A local model compatible with your hardware (default: `gemma3n`).

2. **Set up a virtual environment** *(recommended)*

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *`requirements.txt`*

   ```text
   ollama>=0.5.1
   spacy>=3.8.7
   ```

4. **Download (or change) the model**

   ```bash
   # List available local models
   ollama list

   # Pull gemma3n (default used in the script)
   ollama pull gemma3n
   ```

   > **Tip:** You can change the model by editing the `MODEL_NAME` constant in *correttore.py*.

<a id="usage"></a>

## Usage

```bash
python correttore.py <input_file> <output_file>
```

Examples:

```bash
# Correct a plain-text novel\python correttore.py war_and_peace.txt war_and_peace_fixed.txt

# Overwrite the original (⚠️ irreversible!)
python correttore.py draft.md draft.md
```

Upon completion you will see progress messages like:

```
📝 Correcting chunk 3/12 …
✅ Done! Output in: corrected.txt
```

<a id="configuration"></a>

## Configuration

Tweak the top-level constants in *correttore.py* to fit your project:

| Constant             | Purpose                                          | Default     |
| -------------------- | ------------------------------------------------ | ----------- |
| `CHUNK_CHARS_LIMIT`  | Max characters per chunk sent to the LLM         | `2500`      |
| `OUTPUT_PERCENT`     | % of the chunk written before the overlap begins | `50`        |
| `OVERLAP_PERCENT`    | Chunk overlap (derived from `OUTPUT_PERCENT`)    | `50`        |
| `MODEL_NAME`         | Ollama model to use                              | `"gemma3n"` |
| `MISMATCH_THRESHOLD` | Allowed mismatch between overlapping segments    | `0.05`      |

> **Note:** If you set a far smaller `CHUNK_CHARS_LIMIT`, the script will run more requests but use less GPU/VRAM per call.

<a id="license"></a>

## License

This project is licensed under the terms of the **MIT License**:

```
MIT License

Copyright (c) 2025 Ruj89

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<a id="author"></a>

## Author

**Ruj89**
Feel free to open an issue or pull request if you find a bug or have a feature request!
