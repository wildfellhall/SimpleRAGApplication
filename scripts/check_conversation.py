"""Exercise a real local-model conversation; no generation or embedding mocks."""
import json
import os
import re
import time
from pathlib import Path
import httpx
from backend.conversation import answer_issue

base=os.getenv('FORMA_BASE_URL','http://127.0.0.1:8008').rstrip('/')
questions=[
    'How is Julian doing with fractions? I have ten minutes and no calculators. Suggest a focused teaching approach.',
    'Turn that into an activity with two practice problems and their answers.',
    'Make the second problem easier and explain how to use it.',
    'What constraints did I give you at the beginning, and how does the activity respect them?',
    'What about Noah with the same skill? How would you adapt the activity for him?',
]
history=[]
results=[]
with httpx.Client(timeout=240) as client:
    for turn,question in enumerate(questions,start=1):
        started=time.monotonic()
        result=None
        visible=''
        with client.stream('POST',base+'/api/chat/stream',json={'question':question,'history':history}) as response:
            response.raise_for_status()
            event=''
            for line in response.iter_lines():
                if line.startswith('event:'):event=line[6:].strip()
                if not line.startswith('data:'):continue
                data=json.loads(line[5:])
                if event=='status':print(json.dumps({'turn':turn,'status':data['message']}),flush=True)
                if event=='error':raise AssertionError(data)
                if event=='delta':visible+=data['text']
                if event=='done':result=data
        assert result and visible==result['answer']
        assert not result['metrics'].get('fallback'),result['answer']
        assert answer_issue(visible,question,bool(history),'stop') is None
        assert not re.search(r'<(?:think|analysis)>|the user (?:asks|wants)|we need to answer',visible,re.I)
        assert result['retrieval']['student_ids']==(['STU-1002'] if turn==5 else ['STU-1020'])
        assert result['retrieval']['skill_ids']==['S06']
        if turn==4:
            assert re.search(r'\b(?:ten|10)\b',visible,re.I) and 'calculator' in visible.lower(),visible
        record={'turn':turn,'question':question,'answer':visible,'seconds':round(time.monotonic()-started,1),
                'metrics':result['metrics'],'conversation':result['conversation'],'scope':{'student_ids':result['retrieval']['student_ids'],'skill_ids':result['retrieval']['skill_ids']}}
        results.append(record)
        print(json.dumps(record),flush=True)
        history.extend([{'role':'user','content':question},{'role':'assistant','content':visible,'scope':record['scope']}])
        Path('/private/tmp/forma-conversation-live.json').write_text(json.dumps(results,indent=2))
print('Five consecutive RAG conversation turns passed.',flush=True)
