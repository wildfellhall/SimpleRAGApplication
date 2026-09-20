"""Live local-only benchmark. Requires npm run dev; no model mocks."""
import json
import time
from pathlib import Path
import httpx

BASE = 'http://127.0.0.1:8008'
QUESTION = 'How is the class doing with combining like terms? Who needs support and what should we practice?'
rows=[]
with httpx.Client(timeout=150) as client:
    for name, body in [('class_question',{'question':QUESTION}),('repeat_class_question',{'question':QUESTION}),
                       ('student_question',{'question':'How is Maya doing with fractions? Give her exact skill correctness and one next step.','student_id':'STU-1001'})]:
        start=time.perf_counter()
        first=None
        result=None
        with client.stream('POST',BASE+'/api/chat/stream',json=body) as response:
            response.raise_for_status()
            event=''
            for line in response.iter_lines():
                if line.startswith('event:'):event=line[6:].strip()
                if not line.startswith('data:'):continue
                data=json.loads(line[5:])
                if event=='delta' and first is None:first=(time.perf_counter()-start)*1000
                if event=='error':raise RuntimeError(data['detail'])
                if event=='done':result=data
        assert result and result['answer']
        row={'name':name,'roundtrip_ms':round((time.perf_counter()-start)*1000,1),'first_text_ms':round(first,1),
             'metrics':result['metrics'],'usage':result['usage'],'retrieval':result['retrieval'],
             'source_ids':[s['id'] for s in result['sources']],'answer':result['answer']}
        rows.append(row)
        print(json.dumps(row),flush=True)
Path('/private/tmp/forma-benchmark.json').write_text(json.dumps(rows,indent=2))
