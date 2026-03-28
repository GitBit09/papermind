from flask import Flask, render_template, request, jsonify, make_response
from groq import Groq
import json
import re
import os
from datetime import datetime

app = Flask(__name__)

# Store paper text in memory (avoids cookie size issues)
paper_store = {}

SYSTEM_PROMPT = """You are PaperMind, an expert research paper analyst. You answer questions about research papers with both explanation AND a visual specification.

For EVERY response, you must return EXACTLY this JSON format (no markdown, no extra text):
{
  "answer": "Clear, concise explanation in 2-4 sentences. Be specific to the paper content.",
  "visual_type": "flowchart|timeline|barchart|concept_map|table|comparison|none",
  "visual_title": "Short title for the visual",
  "visual_data": {}
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
  "color_highlight": 0
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
- ALWAYS prefer a visual over none
- Pick the visual type that best matches the question intent
- Keep node labels short (max 4 words)
- Extract REAL data from the paper, do not make up numbers
- For methodology questions use flowchart
- For results/accuracy questions use barchart
- For what is X questions use concept_map
- For comparison questions use comparison or table
- For history/phases use timeline
- Return ONLY valid JSON, no extra text, no markdown fences"""


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

    api_key = request.headers.get('X-API-Key', '')
    if not api_key or not api_key.startswith('gsk_'):
        return jsonify({'error': 'Valid Groq API key required (starts with gsk_)'}), 401

    try:
        import pymupdf
        pdf_bytes = file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text()

        text = text[:6000]

        session_id = api_key[-8:]
        paper_store[session_id] = {
            'text': text,
            'name': file.filename,
            'history': [],
            'qa_log': []
        }

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Paper content:\n{text}\n\nQuestion: Give me a one-sentence summary of what this paper is about, and list 4 good questions someone might ask about it."}
            ]
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        summary_data = json.loads(raw)

        return jsonify({
            'success': True,
            'paper_name': file.filename,
            'page_count': len(doc),
            'summary': summary_data,
            'session_id': session_id
        })

    except json.JSONDecodeError:
        return jsonify({'error': 'Failed to parse AI response'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '').strip()
    session_id = data.get('session_id', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    api_key = request.headers.get('X-API-Key', '')
    if not api_key or not api_key.startswith('gsk_'):
        return jsonify({'error': 'Valid Groq API key required (starts with gsk_)'}), 401

    if not session_id or session_id not in paper_store:
        return jsonify({'error': 'No paper uploaded. Please upload a paper first.'}), 400

    store = paper_store[session_id]
    paper_text = store['text']
    chat_history = store['history']

    try:
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for h in chat_history[-4:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a']})

        messages.append({
            "role": "user",
            "content": f"Paper content:\n{paper_text}\n\nQuestion: {question}"
        })

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            messages=messages
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)

        chat_history.append({'q': question, 'a': raw})
        store['history'] = chat_history[-10:]

        store['qa_log'].append({
            'question': question,
            'answer': result.get('answer', ''),
            'visual_type': result.get('visual_type', 'none'),
            'visual_title': result.get('visual_title', ''),
            'visual_data': result.get('visual_data', None),
            'timestamp': datetime.now().strftime('%H:%M')
        })

        return jsonify({'success': True, 'result': result})

    except json.JSONDecodeError:
        return jsonify({'error': 'Failed to parse AI response'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/report', methods=['GET'])
def download_report():
    """Generate and return a downloadable HTML report of the full session."""
    session_id = request.args.get('session_id', '')

    if not session_id or session_id not in paper_store:
        return jsonify({'error': 'No session found.'}), 400

    store = paper_store[session_id]
    paper_name = store.get('name', 'Unknown Paper')
    qa_log = store.get('qa_log', [])

    if not qa_log:
        return jsonify({'error': 'No questions asked yet. Ask some questions first.'}), 400

    now = datetime.now().strftime('%B %d, %Y at %I:%M %p')

    qa_blocks = ''
    for i, entry in enumerate(qa_log, 1):
        visual_html = build_visual_html(
            entry.get('visual_type', 'none'),
            entry.get('visual_title', ''),
            entry.get('visual_data', None)
        )
        qa_blocks += f'''
        <div class="qa-block">
            <div class="q-num">Q{i}</div>
            <div class="q-text">{entry["question"]}</div>
            <div class="a-text">{entry["answer"]}</div>
            {visual_html}
            <div class="ts">{entry["timestamp"]}</div>
        </div>
        '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PaperMind Report — {paper_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#fafaf9;color:#1a1a1a;padding:40px 20px;}}
.wrap{{max-width:780px;margin:0 auto;}}
.rpt-header{{background:#1e3a8a;color:white;border-radius:14px;padding:34px 38px;margin-bottom:26px;}}
.rpt-header .badge{{display:inline-block;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.2);color:#bfdbfe;font-size:10px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;padding:3px 10px;border-radius:20px;margin-bottom:12px;}}
.rpt-header h1{{font-size:24px;font-weight:700;letter-spacing:-0.5px;margin-bottom:5px;}}
.rpt-header .pname{{font-size:13px;opacity:0.65;margin-bottom:3px;}}
.rpt-header .meta{{font-size:11px;opacity:0.45;}}
.stats{{display:flex;gap:12px;margin-top:16px;}}
.stat{{background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.15);border-radius:9px;padding:7px 15px;font-size:11px;color:rgba(255,255,255,0.7);}}
.stat span{{font-size:20px;font-weight:700;color:white;display:block;}}
.qa-block{{background:white;border-radius:11px;padding:22px 26px;margin-bottom:16px;border:1px solid #e8e5e0;border-left:4px solid #2563eb;position:relative;box-shadow:0 1px 4px rgba(0,0,0,0.04);}}
.q-num{{position:absolute;top:-10px;left:18px;background:#2563eb;color:white;font-size:10px;font-weight:700;padding:2px 8px;border-radius:9px;letter-spacing:0.4px;}}
.q-text{{font-size:14px;font-weight:600;color:#1a1a1a;margin-bottom:8px;line-height:1.5;}}
.a-text{{font-size:13px;color:#4b5563;line-height:1.7;margin-bottom:12px;}}
.ts{{font-size:10px;color:#9ca3af;margin-top:8px;}}
.vbox{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:14px 16px;margin-top:10px;}}
.vbox h4{{font-size:10px;font-weight:600;color:#2563eb;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:10px;}}
table.vt{{width:100%;border-collapse:collapse;font-size:12px;}}
table.vt th{{background:#eff6ff;color:#1d4ed8;padding:7px 10px;text-align:left;font-weight:600;border:1px solid #bfdbfe;}}
table.vt td{{padding:7px 10px;border:1px solid #e2e8f0;color:#374151;}}
table.vt tr:nth-child(even) td{{background:#f8fafc;}}
.br{{margin:5px 0;}}.bl{{font-size:11px;color:#6b7280;margin-bottom:2px;}}.bt{{background:#e5e7eb;border-radius:5px;height:19px;overflow:hidden;}}
.bf{{height:100%;background:linear-gradient(90deg,#2563eb,#60a5fa);border-radius:5px;display:flex;align-items:center;padding-left:7px;color:white;font-size:10px;font-weight:600;min-width:32px;}}
.bf.hl{{background:linear-gradient(90deg,#f59e0b,#fbbf24);}}
.ti{{display:flex;gap:12px;margin-bottom:8px;align-items:flex-start;}}.ty{{background:#2563eb;color:white;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;min-width:44px;text-align:center;white-space:nowrap;}}.tb{{font-size:12px;color:#374151;line-height:1.5;}}
.fw{{display:flex;flex-direction:column;align-items:flex-start;gap:2px;}}.fn{{display:inline-block;background:#eff6ff;border:1.5px solid #93c5fd;border-radius:6px;padding:4px 11px;font-size:11px;font-weight:600;color:#1d4ed8;}}.fn.start,.fn.end{{background:#2563eb;color:white;border-color:#2563eb;border-radius:18px;}}.fn.decision{{background:#fef9c3;border-color:#fde047;color:#854d0e;}}.fa{{color:#94a3b8;font-size:13px;margin:2px 5px;}}
.cc{{display:inline-block;background:#2563eb;color:white;border-radius:7px;padding:5px 13px;font-size:12px;font-weight:700;margin-bottom:8px;}}.cb{{margin-left:16px;border-left:2px solid #bfdbfe;padding-left:12px;margin-bottom:6px;}}.cbl{{font-size:11px;font-weight:600;color:#1d4ed8;margin-bottom:3px;}}.cch{{display:inline-block;background:#eff6ff;border:1px solid #bfdbfe;border-radius:4px;padding:2px 7px;font-size:10px;color:#374151;margin:2px;}}
.cg{{display:grid;grid-template-columns:1fr 1fr;gap:10px;}}.co{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;padding:11px;}}.ct{{font-size:11px;font-weight:700;color:#2563eb;margin-bottom:6px;}}.cp{{font-size:11px;color:#4b5563;padding:3px 0;border-bottom:1px solid #f1f5f9;}}.cp::before{{content:'→ ';color:#2563eb;}}
.footer{{text-align:center;color:#9ca3af;font-size:11px;margin-top:26px;padding-top:16px;border-top:1px solid #e5e7eb;}}
@media print{{body{{background:white;}}.qa-block{{break-inside:avoid;box-shadow:none;}}}}
</style>
</head>
<body>
<div class="wrap">
<div class="rpt-header">
  <div class="badge">◆ PaperMind Report</div>
  <h1>Research Analysis Report</h1>
  <div class="pname">📄 {paper_name}</div>
  <div class="meta">Generated on {now}</div>
  <div class="stats"><div class="stat"><span>{len(qa_log)}</span>Questions Asked</div></div>
</div>
{qa_blocks}
<div class="footer">Generated by <strong>PaperMind</strong> &nbsp;·&nbsp; Powered by Groq &amp; LLaMA 3.3 &nbsp;·&nbsp; {now}</div>
</div>
</body>
</html>'''

    resp = make_response(html)
    safe = re.sub(r'[^\w\-.]', '_', paper_name.replace('.pdf', ''))
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    resp.headers['Content-Disposition'] = f'attachment; filename="PaperMind_Report_{safe}.html"'
    return resp


def build_visual_html(vtype, vtitle, vdata):
    if vtype == 'none' or not vdata:
        return ''
    title = f'<h4>{vtitle}</h4>' if vtitle else ''
    try:
        if vtype == 'table':
            hdrs = ''.join(f'<th>{h}</th>' for h in vdata.get('headers', []))
            rows = ''.join(
                '<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
                for r in vdata.get('rows', [])
            )
            return f'<div class="vbox">{title}<table class="vt"><thead><tr>{hdrs}</tr></thead><tbody>{rows}</tbody></table></div>'

        if vtype == 'barchart':
            labels = vdata.get('labels', [])
            values = vdata.get('values', [])
            unit = vdata.get('unit', '')
            hl = vdata.get('color_highlight', -1)
            mx = max(values, default=1)
            bars = ''
            for i, (lb, v) in enumerate(zip(labels, values)):
                css_class = 'bf hl' if i == hl else 'bf'
                width = f'{(v / mx * 100):.1f}%'
                bars += (
                    f'<div class="br">'
                    f'<div class="bl">{lb}</div>'
                    f'<div class="bt"><div class="{css_class}" style="width:{width}">{v}{unit}</div></div>'
                    f'</div>'
                )
            return f'<div class="vbox">{title}{bars}</div>'

        if vtype == 'timeline':
            items = ''
            for e in vdata.get('events', []):
                items += (
                    f'<div class="ti">'
                    f'<div class="ty">{e.get("year", "")}</div>'
                    f'<div class="tb"><strong>{e.get("label", "")}</strong> — {e.get("description", "")}</div>'
                    f'</div>'
                )
            return f'<div class="vbox">{title}{items}</div>'

        if vtype == 'flowchart':
            nodes = {n['id']: n for n in vdata.get('nodes', [])}
            edges = vdata.get('edges', [])
            order = []
            for e in edges:
                if e['from'] not in order:
                    order.append(e['from'])
                if e['to'] not in order:
                    order.append(e['to'])
            for nid in nodes:
                if nid not in order:
                    order.append(nid)
            flow = ''
            for i, nid in enumerate(order):
                node = nodes.get(nid, {})
                ntype = node.get('type', 'process')
                nlabel = node.get('label', nid)
                flow += f'<span class="fn {ntype}">{nlabel}</span>'
                if i < len(order) - 1:
                    flow += '<span class="fa">↓</span>'
            return f'<div class="vbox">{title}<div class="fw">{flow}</div></div>'

        if vtype == 'concept_map':
            center = vdata.get('center', '')
            bhtml = ''
            for b in vdata.get('branches', []):
                children_html = ''.join(
                    f'<span class="cch">{c}</span>'
                    for c in b.get('children', [])
                )
                bhtml += (
                    f'<div class="cb">'
                    f'<div class="cbl">↳ {b.get("label", "")}</div>'
                    f'<div>{children_html}</div>'
                    f'</div>'
                )
            return f'<div class="vbox">{title}<div class="cc">{center}</div>{bhtml}</div>'

        if vtype == 'comparison':
            left = vdata.get('left', {})
            right = vdata.get('right', {})
            lpts = ''.join(f'<div class="cp">{p}</div>' for p in left.get('points', []))
            rpts = ''.join(f'<div class="cp">{p}</div>' for p in right.get('points', []))
            return (
                f'<div class="vbox">{title}'
                f'<div class="cg">'
                f'<div class="co"><div class="ct">{left.get("title", "")}</div>{lpts}</div>'
                f'<div class="co"><div class="ct">{right.get("title", "")}</div>{rpts}</div>'
                f'</div></div>'
            )

    except Exception:
        pass
    return ''


if __name__ == '__main__':
    app.run(debug=True, port=5000)