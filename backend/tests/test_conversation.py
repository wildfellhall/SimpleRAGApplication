import json
import re
import httpx
import pytest
from fastapi.testclient import TestClient
from backend import conversation, rag, vectors
from backend.main import app
from backend.tests.test_app import isolated_db

GOOD = ('Julian can practice fraction addition with paper strips, using a ten-minute activity without calculators. '
        'First model a common denominator, then have him explain two practice examples. Finish with a short independent check '
        'and use his explanation to decide whether another worked example would help. [STU-1020]')


def model_script(monkeypatch, replies, counter=None):
    requests=[]
    templates=[]
    original=httpx.AsyncClient
    def respond(request):
        body=json.loads(request.content) if request.content else {}
        if request.url.path=='/props':return httpx.Response(200,json={'default_generation_settings':{'n_ctx':8192}})
        if request.url.path=='/apply-template':
            templates.append(body)
            return httpx.Response(200,json={'prompt':json.dumps(body['messages'])})
        if request.url.path=='/tokenize':
            count=counter(body['content']) if counter else len(body['content'])//3
            return httpx.Response(200,json={'tokens':[0]*count})
        requests.append(body)
        reply=replies[min(len(requests)-1,len(replies)-1)]
        parts=reply.get('parts',[{'content':reply.get('content','')}])
        events=[{'choices':[{'delta':part,'finish_reason':None}]} for part in parts]
        events.append({'choices':[{'delta':{},'finish_reason':reply.get('finish','stop')}],'usage':{'prompt_tokens':500,'completion_tokens':100}})
        raw=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
        return httpx.Response(200,content=raw,headers={'content-type':'text/event-stream'})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(respond),**kwargs))
    return requests,templates


def completion(response):
    return json.loads(next(frame for frame in response.text.split('\n\n') if frame.startswith('event: done')).split('data: ',1)[1])


@pytest.mark.parametrize('raw,expected',[
    ('<think>secret reasoning</think>Useful answer.','Useful answer.'),
    ('<think>never closes',''),
    ('secret tail</think>Useful answer.','Useful answer.'),
    ('<analysis>secret</analysis>Useful answer.','Useful answer.'),
    ('<|channel|>analysis hidden<|channel|>final<|message|>Useful answer.','Useful answer.'),
])
def test_reasoning_markers_are_removed(raw,expected):
    assert conversation.clean_answer(raw)==expected


@pytest.mark.parametrize('first',[
    {'content':''},
    {'parts':[{'content':'<thi'},{'content':'nk>SECRET REASONING'},{'content':'</think>'}]},
    {'parts':[{'reasoning_content':'SECRET REASONING'}]},
    {'content':'We need to answer the user carefully. First we should provide a summary of the records.'},
    {'content':'Sure.'},
    {'content':'[STU-9999] '*30},
    {'content':'### Thinking\nI should compose a careful response based on the supplied records before giving an answer to the teacher.'},
    {'content':'An unfinished lesson plan that never gets to the promised practice activities.','finish':'length'},
])
def test_invalid_attempt_is_retried_without_leaking_to_stream(monkeypatch,first):
    requests,_=model_script(monkeypatch,[first,{'content':GOOD}])
    with TestClient(app) as client:
        response=client.post('/api/chat/stream',json={'question':'Plan a fraction activity for Julian.'})
        result=completion(response)
    assert len(requests)==2
    assert 'SECRET REASONING' not in response.text and 'We need to answer' not in response.text
    assert 'Sure.' not in response.text and '<think>' not in response.text
    assert result['answer']==GOOD
    assert result['metrics']['recovered'] and result['metrics']['generation_attempts']==2
    assert not result['metrics']['fallback']
    assert result['metrics']['recovery_reasons']
    assert all(r['max_tokens']==1200 and r['reasoning_format']=='deepseek' and r['chat_template_kwargs']=={'enable_thinking':False,'preserve_thinking':True} for r in requests)
    assert all({key:r[key] for key in conversation.SAMPLING_OPTIONS}==conversation.SAMPLING_OPTIONS for r in requests)
    assert requests[1]['cache_prompt'] is False


def test_valid_answer_with_reasoning_is_clean_before_first_visible_delta(monkeypatch):
    requests,_=model_script(monkeypatch,[{'parts':[{'content':'<thi'},{'content':'nk>PRIVATE'},{'content':'</think>'+GOOD,'reasoning_content':'MORE PRIVATE'}]}])
    with TestClient(app) as client:
        response=client.post('/api/chat/stream',json={'question':'Plan a fraction activity for Julian.'})
    assert len(requests)==1
    assert 'PRIVATE' not in response.text
    assert completion(response)['answer']==GOOD
    deltas=[json.loads(f.split('data: ',1)[1])['text'] for f in response.text.split('\n\n') if f.startswith('event: delta')]
    assert ''.join(deltas)==GOOD


def test_repeated_empty_attempts_return_honest_grounded_fallback_not_cache(monkeypatch):
    requests,_=model_script(monkeypatch,[{'content':'<think>no final answer</think>'}])
    with TestClient(app) as client:
        body={'question':'Plan fraction practice for Julian.'}
        for _ in range(2):
            result=client.post('/api/chat',json=body).json()
            assert result['metrics']['fallback'] and not result['metrics']['cached']
            assert 'could not produce a reliable conversational response' in result['answer']
            assert '[STU-1020]' in result['answer'] and len(result['answer'])>100
            assert 'no final answer' not in result['answer']
    assert len(requests)==4


def test_context_denial_retries_and_preserves_teacher_constraints(monkeypatch):
    requests,_=model_script(monkeypatch,[{'content':"I don't have access to earlier conversation history. Please repeat all of your constraints before I can suggest a lesson plan that meets your requirements."},{'content':GOOD}])
    history=[{'role':'user','content':'Help Julian with fractions. Ten minutes, no calculators.'},{'role':'assistant','content':'We can use fraction strips for this lesson.'}]
    with TestClient(app) as client:
        result=client.post('/api/chat',json={'question':'What constraints did I give you?','history':history}).json()
    assert len(requests)==2 and not result['metrics']['fallback']
    assert 'Ten minutes, no calculators.' in str(requests[0]['messages'])
    assert result['retrieval']['student_ids']==['STU-1020'] and result['retrieval']['skill_ids']==['S06']


def test_direct_fact_does_not_require_an_artificially_long_answer(monkeypatch):
    requests,_=model_script(monkeypatch,[{'content':'Julian has 70% correctness in fractions [STU-1020].'}])
    with TestClient(app) as client:
        result=client.post('/api/chat',json={'question':'What is Julian’s exact correctness in fractions?'}).json()
    assert len(requests)==1 and not result['metrics']['fallback']


def test_scope_survives_multiple_followups_and_can_change():
    history=[{'role':'user','content':'Tell me about Maya and fractions.'}]
    for question in ['Suggest a teaching activity.','Give two examples.','Make the second one easier.','How long should that take?']:
        _,meta=rag.retrieve_with_metadata(question,history=history)
        assert meta['student_ids']==['STU-1001'] and meta['skill_ids']==['S06']
        history.extend([{'role':'user','content':question},{'role':'assistant','content':'Use paper strips for the activity.'}])
    _,meta=rag.retrieve_with_metadata('Now focus on integers.',history=history)
    assert meta['student_ids']==['STU-1001'] and meta['skill_ids']==['S05']
    _,meta=rag.retrieve_with_metadata('What about Noah?',history=history)
    assert meta['student_ids']==['STU-1002'] and meta['skill_ids']==['S06']
    _,meta=rag.retrieve_with_metadata('Show the whole class across all skills.',history=history)
    assert meta['student_ids']==[] and meta['skill_ids']==[]
    _,meta=rag.retrieve_with_metadata('Show the whole class across all skills.',history=history[:1])
    assert meta['student_ids']==[] and meta['skill_ids']==[] and 'Previous question' not in meta['query']
    _,ids,skills=rag.plan_query('New topic: explain the distributive property.',None,history,vectors.get_index().documents)
    assert ids==[] and skills==['S02']


def test_semantically_resolved_skill_is_carried_by_assistant_scope():
    history=[{'role':'user','content':'Help Julian work with unlike denominators.'},
             {'role':'assistant','content':GOOD,'scope':{'student_ids':['STU-1020'],'skill_ids':['S06']}}]
    _,meta=rag.retrieve_with_metadata('Give him a follow-up exercise.',history=history)
    assert meta['student_ids']==['STU-1020'] and meta['skill_ids']==['S06']


def test_old_teacher_requirements_survive_context_compaction_and_token_budget(monkeypatch):
    requests,templates=model_script(monkeypatch,[{'content':GOOD}],counter=lambda prompt:len(prompt)//2)
    history=[{'role':'user','content':'Help Julian with fractions. Ten minutes and no calculators. Use paper strips.'}]
    for i in range(16):
        history.extend([{'role':'assistant','content':f'Activity version {i}. '+('Practice examples and explanations. '*40)},
                        {'role':'user','content':f'Revise example {i}, keeping the original constraints. '+('Consider the lesson. '*15)}])
    with TestClient(app) as client:
        result=client.post('/api/chat',json={'question':'Recall my original calculator and time constraints, then refine the activity.','history':history}).json()
    prompt=str(requests[0]['messages'])
    assert 'Ten minutes and no calculators' in prompt
    assert 'Revise example 15' in prompt
    assert result['conversation']['history_messages']==33 and result['conversation']['compacted']
    assert result['conversation']['prompt_tokens']+1200+256<=8192
    assert len(templates)>1


def test_guided_followup_keeps_scope_but_can_request_a_lesson_plan(monkeypatch):
    requests,_=model_script(monkeypatch,[{'content':GOOD}])
    with TestClient(app) as client:
        result=client.post('/api/chat',json={'question':'Turn the summary into an activity.','student_id':'STU-1020',
            'filters':{'skill_id':'S06','intent':'summary'},'history':[{'role':'user','content':'Summarize Julian’s fractions.'},{'role':'assistant','content':GOOD}]}).json()
    assert 'fulfill the latest teacher request' in requests[0]['messages'][0]['content']
    assert 'Previous question: Summarize' in result['retrieval']['query']
    assert result['retrieval']['skill_ids']==['S06']


def test_retrieved_problem_context_includes_the_canonical_answer():
    docs=rag.retrieve('How is Julian doing with fractions?')
    problems=[d for d in docs if d['kind']=='problem']
    assert problems
    for problem in problems:
        assert 'correct_answer='+problem['data']['answer'] in problem['content']


def test_explicit_one_word_requests_can_be_short():
    assert conversation.answer_issue('Yes.','Answer yes or no: may I use paper strips?',False,'stop') is None
    assert conversation.answer_issue('Use a worksheet.','Plan a full lesson using average time per problem.',False,'stop')=='unhelpfully short answer'


FOLLOWUP = ('Suggested exit ticket: Ask Julian to solve 1/2 + 1/4 independently, then explain why '
            'the denominator stays four. The answer is 3/4. Use his explanation to check whether '
            'he understands equivalent fractions before choosing the next lesson.')
HISTORY = [{'role':'user','content':'Plan a ten-minute fraction activity for Julian without calculators.'},
           {'role':'assistant','content':GOOD}]


@pytest.mark.parametrize('repeated',[
    GOOD,
    '**'+GOOD.replace('[STU-1020]','[S06]')+'**',
    GOOD.replace('can practice','should practice').replace('paper strips','fraction strips'),
])
def test_copied_followup_is_hidden_and_retried_for_latest_task(monkeypatch,repeated):
    requests,_=model_script(monkeypatch,[{'content':repeated},{'content':FOLLOWUP}])
    question='Now write an exit ticket with its answer.'
    with TestClient(app) as client:
        response=client.post('/api/chat/stream',json={'question':question,'history':HISTORY})
    result=completion(response)
    assert result['answer']==FOLLOWUP and result['metrics']['generation_attempts']==2
    assert result['metrics']['recovery_reasons']==['repeated earlier answer instead of addressing the latest request']
    assert 'ten-minute activity' not in response.text
    for request in requests:
        assert request['messages'][-1]=={'role':'user','content':question}
        assert 'EVIDENCE:' in request['messages'][0]['content']
    assert 'discarded draft repeated' in requests[1]['messages'][0]['content']
    assert requests[1]['cache_prompt'] is False


def test_persistent_repetition_returns_uncached_fallback(monkeypatch):
    requests,_=model_script(monkeypatch,[{'content':GOOD}])
    with TestClient(app) as client:
        body={'question':'Give two new worked examples.','history':HISTORY}
        for _ in range(2):
            result=client.post('/api/chat',json=body).json()
            assert result['metrics']['fallback'] and not result['metrics']['cached']
            assert 'could not produce a reliable conversational response' in result['answer']
            assert result['answer']!=GOOD
    assert len(requests)==4


@pytest.mark.parametrize('question,answer',[
    ('Please repeat that answer.',GOOD),
    ('Show me that again.',GOOD),
    (HISTORY[0]['content'],GOOD),
    ('Make the activity five minutes instead of ten.',GOOD.replace('ten-minute','five-minute')),
    ('Change the 10-minute activity to 5 minutes.',GOOD.replace('ten-minute','5-minute')),
    ('Summarize that in one sentence.',GOOD.split('First model')[0]),
    ('Now write an exit ticket with its answer.',FOLLOWUP),
    ('Add an exit ticket to the plan.',GOOD+'\n\n'+FOLLOWUP),
])
def test_valid_continuations_and_deliberate_repetition_are_allowed(question,answer):
    assert conversation.repetition_issue(answer,question,HISTORY) is None


def test_repetition_is_detected_beyond_last_answer_and_with_negative_repeat_request():
    history=HISTORY+[{'role':'user','content':'Now write an exit ticket.'},{'role':'assistant','content':FOLLOWUP}]
    assert conversation.repetition_issue(GOOD,"Don't repeat the activity. Explain a different approach.",history)
    assert conversation.repetition_issue(GOOD,'Don’t repeat the activity.',history)
    assert conversation.repetition_issue(GOOD,'Can you not repeat the activity?',history)
    assert conversation.repetition_issue(GOOD,'Repeat the plan but add an exit ticket.',history)
    assert conversation.repetition_issue('Julian has 70% correctness in fractions.','Remind me of his correctness.',history) is None


def test_distinct_followups_are_not_served_a_previous_cached_answer(monkeypatch):
    requests,_=model_script(monkeypatch,[{'content':FOLLOWUP},{'content':GOOD}])
    with TestClient(app) as client:
        first={'question':'Give an exit ticket.','history':HISTORY}
        second={'question':'Repeat the activity plan.','history':HISTORY}
        assert client.post('/api/chat',json=first).json()['answer']==FOLLOWUP
        assert client.post('/api/chat',json=second).json()['answer']==GOOD
        assert client.post('/api/chat',json=first).json()['metrics']['cached']
    assert len(requests)==2


def test_template_keeps_structure_without_preserving_actual_reasoning(monkeypatch):
    requests,templates=model_script(monkeypatch,[{'content':FOLLOWUP}])
    history=[HISTORY[0],{'role':'assistant','content':'<think>PRIVATE DRAFT</think>'+GOOD}]
    with TestClient(app) as client:
        result=client.post('/api/chat',json={'question':'Write an exit ticket.','history':history}).json()
    assert result['answer']==FOLLOWUP
    assert 'PRIVATE DRAFT' not in str(requests) and '<think>' not in str(requests)
    assert requests[0]['messages'][-2]['content']==GOOD
    assert templates[0]['chat_template_kwargs']==requests[0]['chat_template_kwargs']==conversation.TEMPLATE_OPTIONS
