"""Live local-model check that changing the task advances a RAG conversation."""
import json
import os
import re
from pathlib import Path
import httpx
from backend.conversation import repetition_issue

base=os.getenv('FORMA_BASE_URL','http://127.0.0.1:8008').rstrip('/')
history=[
    {'role':'user','content':'Plan a ten-minute fraction activity for Julian without calculators.'},
    {'role':'assistant','content':
        'Suggested activity: Use paper strips to model fraction addition. Spend two minutes showing '
        'equivalent fractions, five minutes practicing addition with common denominators, and three '
        'minutes checking understanding independently. Ask Julian to explain why the denominator '
        'stays the same when adding equal parts. Keep the activity calculator-free.',
     'scope':{'student_ids':['STU-1020'],'skill_ids':['S06']}},
]
questions=[
    'Write exactly two exit-ticket questions, followed by an answer key. Do not restate the activity plan.',
    'Replace only the second question with an easier fraction addition problem. Give the new question and its answer.',
    'Now draft a short note to Julian’s parent explaining how to practice this skill at home. Address the parent directly.',
]
results=[]
with httpx.Client(timeout=240) as client:
    for question in questions:
        result=None
        with client.stream('POST',base+'/api/chat/stream',json={'question':question,'history':history}) as response:
            response.raise_for_status()
            event=''
            for line in response.iter_lines():
                if line.startswith('event:'):event=line[6:].strip()
                if not line.startswith('data:'):continue
                data=json.loads(line[5:])
                if event=='error':raise AssertionError(data)
                if event=='status':print(data['message'],flush=True)
                if event=='done':result=data
        assert result and not result['metrics'].get('fallback'),result
        answer=result['answer']
        assert repetition_issue(answer,question,history) is None
        assert result['retrieval']['student_ids']==['STU-1020']
        assert result['retrieval']['skill_ids']==['S06']
        if len(results)<2:
            assert re.search(r'\d+\s*/\s*\d+|\\(?:d?frac)\{\d+\}\{\d+\}',answer),answer
            assert re.search(r'answer|solution|=',answer,re.I),answer
        else:
            assert re.search(r'\b(?:dear|your child|you can|at home)\b',answer,re.I),answer
        record={'question':question,'answer':answer,'metrics':result['metrics']}
        results.append(record)
        print(json.dumps(record),flush=True)
        history.extend([{'role':'user','content':question},{'role':'assistant','content':answer,
                        'scope':{key:result['retrieval'][key] for key in ['student_ids','skill_ids']}}])
        Path('/private/tmp/forma-followups-live.json').write_text(json.dumps(results,indent=2))
print('Exit ticket, focused revision, and parent note follow-ups passed.',flush=True)
