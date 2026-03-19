from flask import Flask, render_template, request, jsonify, session
import anthropic
import json
import re
import os
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are PaperMind, an expert research paper analyst. You answer questions about research papers with both explanation AND a visual specification.

For EVERY response, you must return EXACTLY this JSON format (no markdown, no extra text):
{
  "answer": "Clear, concise explanation in 2-4 sentences. Be specific to the paper content.",
  "visual_type": "flowchart|timeline|barchart|concept_map|table|comparison|none",
  "visual_title": "Short title for the visual",
  "visual_data": <see formats below>
}

VISUAL TYPE FORMATS:

flowchart - for processes, methodologies, pipelines:
"visual_data": {
  "nodes": [{"id": "1", "label": "Step Name", "type": "start|process|decision|end"}],
  "edges": [{"from": "1", "to": "2", "label": "optional"}]
}

timeline - for chronological events, phases, history:
"visual_data": {
  "events": [{"year": "2024", "label": "Event Name", "description": "short desc"}]
}

barchart - for numerical comparisons, results, metrics:
"visual_data": {
  "labels": ["Model A", "Model B"],
  "values": [52.5, 74.8],
  "unit": "%",
  "color_highlight": 3
}

concept_map - for relationships between ideas/terms:
"visual_data": {
  "center": "Main Concept",
  "branches": [
    {"label": "Branch 1", "children": ["Child A", "Child B"]},
    {"label": "Branch 2", "children": ["Child C"]}
  ]
}

table - for structured comparisons:
"visual_data": {
  "headers": ["Col1", "Col2", "Col3"],
  "rows": [["val1", "val2", "val3"]]
}

comparison - for two things side by side:
"visual_data": {
  "left": {"title": "Option A", "points": ["point1", "point2"]},
  "right": {"title": "Option B", "points": ["point1", "point2"]}
}

none - for simple factual answers:
"visual_data": null

RULES:
- ALWAYS prefer a visual over none. Ask yourself: can this be shown visually?
- Pick the visual type that best matches the question intent
- Keep node labels short (max 4 words)
- Extract REAL data from the paper — don't make up numbers
- For methodology questions → flowchart
- For results/accuracy questions → barchart  
- For "what is X" questions → concept_map
- For comparison questions → comparison or table
- For history/phases → timeline"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_paper():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files supported'}), 400
    
    try:
        import pymupdf
        pdf_bytes = file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Limit to ~8000 chars to stay in context
        text = text[:8000]
        
        # Store in session
        session['paper_text'] = text
        session['paper_name'] = file.filename
        session['chat_history'] = []
        
        # Get paper summary
        summary_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Paper content:\n{text}\n\nQuestion: Give me a one-sentence summary of what this paper is about, and list 4 good questions someone might ask about it."
            }]
        )
        
        raw = summary_msg.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        summary_data = json.loads(raw)
        
        return jsonify({
            'success': True,
            'paper_name': file.filename,
            'page_count': len(doc),
            'summary': summary_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    paper_text = session.get('paper_text', '')
    if not paper_text:
        return jsonify({'error': 'No paper uploaded'}), 400
    
    chat_history = session.get('chat_history', [])
    
    # Build messages
    messages = []
    
    # Add history (last 4 exchanges)
    for h in chat_history[-4:]:
        messages.append({"role": "user", "content": h['q']})
        messages.append({"role": "assistant", "content": h['a']})
    
    # Add current question
    messages.append({
        "role": "user",
        "content": f"Paper content:\n{paper_text}\n\nQuestion: {question}"
    })
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        
        raw = response.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        
        # Save to history
        chat_history.append({'q': question, 'a': raw})
        session['chat_history'] = chat_history
        
        return jsonify({'success': True, 'result': result})
        
    except json.JSONDecodeError as e:
        return jsonify({'error': 'Failed to parse response'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
