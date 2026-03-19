# PaperMind 📄
### Chat with any research paper. See the answers.

Upload a PDF research paper, ask questions in plain English, and get answers with **auto-generated visuals** — flowcharts, bar charts, timelines, concept maps — drawn from the actual paper content.

---

## The Problem It Solves

Reading research papers is hard. You get walls of dense text, complex methodologies, and tables of numbers. Existing tools (ChatGPT, NotebookLM) give you more text walls. PaperMind gives you **visual answers** — because understanding comes faster when you can see it.

## Demo

Upload your IEEE paper → Ask "explain the methodology" → Get an animated flowchart  
Ask "what are the key results?" → Get a bar chart with actual numbers from the paper  
Ask "compare the models" → Get a side-by-side comparison card  

## Visual Types

| Question Type | Visual Generated |
|--------------|-----------------|
| Methodology / Pipeline | Flowchart |
| Results / Accuracy | Bar Chart |
| Concepts / Definitions | Concept Map |
| History / Phases | Timeline |
| Model Comparisons | Comparison Cards |
| Structured Data | Table |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude claude-sonnet-4-20250514 (Anthropic) |
| PDF Parsing | PyMuPDF |
| Backend | Python Flask |
| Visuals | Pure SVG + HTML Canvas (no chart libraries) |
| Prompt Engineering | Structured JSON output with visual type classification |

## Setup

```bash
git clone https://github.com/GitBit09/papermind
cd papermind
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your_key_here   # Windows
# export ANTHROPIC_API_KEY=your_key_here  # Mac/Linux
python app.py
# Visit http://localhost:5000
```

## How It Works

1. PDF text is extracted with PyMuPDF
2. User asks a question in natural language
3. Claude reads the paper and the question, then decides:
   - What's the best answer (2-4 sentences)
   - What visual type fits (flowchart/barchart/timeline/etc)
   - What data to put in the visual (extracted from the paper)
4. Frontend renders the visual as animated SVG

## Prompt Engineering

The core innovation is forcing the LLM to output **structured JSON with a visual specification** alongside the answer — not just text. The system prompt classifies question intent and maps it to visual type automatically.

## License
MIT
