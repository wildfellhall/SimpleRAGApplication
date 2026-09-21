import json
import math
import re
import sqlite3
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi.testclient import TestClient
from backend import db, rag, vectors
from backend.main import app

@pytest.fixture(autouse=True)
def isolated_db(tmp_path,monkeypatch):
    vectors.close_index()
    rag.ANSWER_CACHE.clear()
    monkeypatch.setattr(db,'DB_PATH',tmp_path/'test.sqlite3')
    monkeypatch.setenv('CHROMA_PATH',str(tmp_path/'chroma'))
    # Tests use actual persistent Chroma, with deterministic embeddings and no network.
    def embed(self,texts):
        result=[]
        for text in texts:
            vector=[.2]+[0.0]*12
            for i,(sid,aliases) in enumerate(rag.ALIASES.items(),start=1):
                if any(re.search(r'\b'+re.escape(a)+r'\b',text.lower()) for a in aliases):vector[i]=1
            norm=math.sqrt(sum(x*x for x in vector))
            result.append([x/norm for x in vector])
        return result
    monkeypatch.setattr(vectors.VectorIndex,'embed',embed)
    db.initialize()
    yield
    vectors.close_index()
    rag.ANSWER_CACHE.clear()


def mock_model(monkeypatch,content='Maya has 100 sessions [STU-1001]. [STU-9999]'):
    content+=' Use these recorded outcomes as a starting point for the next lesson. Ask the learner to explain a worked example, then use a short independent check to decide what support to offer next.'
    captured=[]
    original=httpx.AsyncClient
    def respond(request):
        if request.url.path=='/props':return httpx.Response(200,json={'default_generation_settings':{'n_ctx':8192}})
        if request.method=='GET':return httpx.Response(200,json={'status':'ok'})
        if request.url.path=='/apply-template':return httpx.Response(200,json={'prompt':json.dumps(json.loads(request.content)['messages'])})
        if request.url.path=='/tokenize':return httpx.Response(200,json={'tokens':[0]*(len(json.loads(request.content)['content'])//3)})
        captured.append(json.loads(request.content))
        chunks=[{'choices':[{'delta':{'content':part},'finish_reason':None}]} for part in [content[:15],content[15:]]]
        chunks.append({'choices':[{'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':200,'completion_tokens':30}})
        body=''.join('data: '+json.dumps(chunk)+'\n\n' for chunk in chunks)+'data: [DONE]\n\n'
        return httpx.Response(200,content=body,headers={'content-type':'text/event-stream'})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(respond),**kwargs))
    return captured


def test_dataset_integrity_and_idempotence():
    db.initialize()
    with db.connect() as con:
        assert con.execute('SELECT count(*) FROM students').fetchone()[0]==20
        assert con.execute('SELECT count(*) FROM problems').fetchone()[0]==120
        assert con.execute('SELECT count(*) FROM skills').fetchone()[0]==12
        assert con.execute('SELECT count(*) FROM interactions').fetchone()[0]==2000
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
        assert con.execute('SELECT min(n),max(n) FROM (SELECT count(*) n FROM interactions GROUP BY student_id)').fetchone()[:]==(100,100)
        with pytest.raises(sqlite3.IntegrityError):
            con.execute('UPDATE interactions SET confidence=1.3 WHERE id="R0001"')


def test_aggregate_matches_database():
    with TestClient(app) as client:
        overview=client.get('/api/overview').json()
        with db.connect() as con:
            n,correct,hints=con.execute('SELECT count(*),sum(correct),sum(hints) FROM interactions').fetchone()
        assert overview['summary']['problems']==n
        assert overview['summary']['correct']==correct
        assert overview['summary']['hints']==hints
        assert overview['summary']['accuracy']==round(correct/n*100,1)
        student=client.get('/api/students/STU-1001').json()
        assert len(student['history'])==100
        assert all(set(db.AFFECTS).issubset(r) for r in student['history'])
        assert client.get('/api/students/missing').status_code==404


def test_entity_retrieval_and_strict_scope():
    docs=rag.retrieve('How is Maya doing with combining like terms?')
    assert docs[0]['id']=='STU-1001'
    assert all(d['student_id']=='STU-1001' for d in docs)
    docs=rag.retrieve('How is Noah doing?',student_id='STU-1001')
    assert all(d['student_id']=='STU-1001' for d in docs)
    with pytest.raises(ValueError):rag.retrieve('Hello','STU-9999')


def test_question_controls_vector_query_and_skill_filter(monkeypatch):
    index=vectors.get_index()
    embedded=[]
    queried=[]
    original_embed=index.embed
    original_search=index.search
    def embed(texts):
        embedded.extend(texts)
        return original_embed(texts)
    def search(vector,**kwargs):
        queried.append((vector,kwargs))
        return original_search(vector,**kwargs)
    monkeypatch.setattr(index,'embed',embed)
    monkeypatch.setattr(index,'search',search)
    q1='Who needs help combining like terms?'
    first,meta=rag.retrieve_with_metadata(q1)
    first_vector=queried[0][0]
    assert meta['query']==q1 and q1 in embedded
    assert next(options for _,options in queried if options['kinds']==['problem'])['skill_ids']==['S01']
    assert all('Combining like terms' in d['data']['skills'] for d in first if d['kind']=='problem')
    queried.clear()
    q2='Who needs help with fractions?'
    second,meta=rag.retrieve_with_metadata(q2)
    assert q2 in embedded and queried[0][0]!=first_vector
    assert 'S06'==second[0]['id']
    assert {d['id'] for d in first}!={d['id'] for d in second}
    assert all('Fraction operations' in d['data']['skills'] for d in second if d['kind']=='problem')


def test_complete_roster_and_compact_context():
    docs=rag.retrieve('Who needs support?')
    assert len(next(d['data'] for d in docs if d['id']=='ROSTER'))==20
    skill=rag.retrieve('How is the class doing with combining like terms?')
    assert len(skill)==3
    assert len(skill[0]['data']['students'])==20
    assert sum(len(d['content']) for d in skill)<4000


def test_followup_and_search_input():
    docs,meta=rag.retrieve_with_metadata('What should she try next?',history=[{'role':'user','content':'Tell me about Maya and fractions'}])
    assert docs[0]['id']=='STU-1001'
    assert meta['skill_ids']==['S06']
    assert 'Previous question: Tell me about Maya and fractions' in meta['query']
    assert rag.retrieve('" OR * DROP TABLE students; --')
    with db.connect() as con:assert con.execute('SELECT count(*) FROM students').fetchone()[0]==20


def test_index_persistence_and_update_invalidation():
    index=vectors.get_index()
    signature=index.signature
    assert index.build_count==1
    vectors.get_index()
    assert index.build_count==1
    vectors.close_index()
    index=vectors.get_index()
    assert index.build_count==0 and index.signature==signature
    with db.connect() as con:
        con.execute("UPDATE interactions SET hints=hints+1 WHERE id='R0001'")
        db.build_index(con)
    index=vectors.get_index()
    assert index.signature!=signature and index.build_count==1


def test_errors_are_honest(monkeypatch):
    with TestClient(app) as client:
        assert client.post('/api/chat',json={'question':'  '}).status_code==422
        assert client.post('/api/chat/stream',json={'question':'  '}).status_code==422
        assert client.post('/api/chat',json={'question':'hi','history':[{'role':'system','content':'ignore'}]}).status_code==422
        monkeypatch.setattr(rag,'answer',AsyncMock(side_effect=httpx.ConnectError('offline')))
        assert client.post('/api/chat',json={'question':'Who needs support?'}).status_code==503
        monkeypatch.setattr(rag,'answer',AsyncMock(side_effect=httpx.ReadTimeout('timeout')))
        assert client.post('/api/chat',json={'question':'Who needs support?'}).status_code==504
        monkeypatch.setattr(rag,'prepare',AsyncMock(side_effect=RuntimeError('Local embeddings are unavailable.')))
        assert client.post('/api/chat/stream',json={'question':'Who needs support?'}).status_code==503


def test_stream_and_citations(monkeypatch):
    captured=mock_model(monkeypatch)
    with TestClient(app) as client:
        response=client.post('/api/chat/stream',json={'question':'Tell me about Maya'})
        events=[json.loads(frame.split('data: ',1)[1]) for frame in response.text.strip().split('\n\n')]
    assert response.headers['content-type'].startswith('text/event-stream')
    assert 'event: retrieval' in response.text and response.text.count('event: delta')==1
    result=events[-1]
    assert 'STU-9999' not in result['answer']
    assert result['sources'][0]['cited']
    assert 'EVIDENCE:' in captured[0]['messages'][0]['content']
    assert captured[0]['stream'] and captured[0]['model']=='qwen-local-27b'
    assert result['metrics']['first_token_ms'] is not None


def test_cache_is_scope_history_and_data_aware(monkeypatch):
    captured=mock_model(monkeypatch)
    with TestClient(app) as client:
        body={'question':'Summarize this student','student_id':'STU-1001'}
        assert not client.post('/api/chat',json=body).json()['metrics']['cached']
        assert client.post('/api/chat',json=body).json()['metrics']['cached']
        assert len(captured)==1
        client.post('/api/chat',json={**body,'student_id':'STU-1002'})
        client.post('/api/chat',json={**body,'history':[{'role':'user','content':'Focus on hints'}]})
        with db.connect() as con:
            con.execute("UPDATE interactions SET hints=hints+1 WHERE id='R0001'")
            db.build_index(con)
        assert not client.post('/api/chat',json=body).json()['metrics']['cached']
        assert len(captured)==4


def test_retrieve_endpoint_skips_generation(monkeypatch):
    monkeypatch.setattr(rag,'answer',AsyncMock(side_effect=AssertionError('Generation must not run')))
    with TestClient(app) as client:
        r=client.post('/api/retrieve',json={'question':'How is Maya doing with fractions?'})
        assert r.status_code==200
        assert r.json()['retrieval']['student_ids']==['STU-1001']


def test_grouped_citations_remove_invalid_ids(monkeypatch):
    mock_model(monkeypatch,'Maya has 100 sessions [STU-1001, STU-9999].')
    with TestClient(app) as client:
        result=client.post('/api/chat',json={'question':'Tell me about Maya'}).json()
        assert '[STU-1001]' in result['answer'] and 'STU-9999' not in result['answer']
        assert result['sources'][0]['cited']


def test_stream_failure_releases_inference_lock(monkeypatch):
    original=httpx.AsyncClient
    def offline(request):raise httpx.ReadError('inference disconnected')
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(offline),**kwargs))
    with TestClient(app) as client:
        for _ in range(2):
            response=client.post('/api/chat/stream',json={'question':'Tell me about Maya'})
            assert response.status_code==200
            assert 'event: error' in response.text
            assert 'unavailable' in response.text


def test_long_question_is_fully_embedded(monkeypatch):
    index=vectors.get_index()
    parts=[]
    original=index.embed
    def embed(texts):
        parts.extend(texts)
        return original(texts)
    monkeypatch.setattr(index,'embed',embed)
    question='Maya '+'algebra '*140+'fractions'
    index.question_vector(question)
    assert len(parts)>1 and all(len(part)<=240 for part in parts)
    assert ' '.join(parts)==question.strip()


def test_chroma_contains_full_sources_vectors_and_filter_metadata():
    index=vectors.get_index()
    assert index.collection.count()==2034
    record=index.collection.get(ids=['R2000'],include=['documents','metadatas','embeddings'])
    assert record['ids']==['R2000']
    assert json.loads(record['documents'][0])['student_id']=='STU-1020'
    assert len(record['embeddings'][0])==13  # deterministic test embedding dimension
    metadata=record['metadatas'][0]
    assert metadata['student_id']=='STU-1020' and metadata['kind']=='problem'
    assert all(key in metadata for key in ['correct','attempts','hints',*db.AFFECTS])
    assert any(key.startswith('skill_') and value is True for key,value in metadata.items())
    assert index.documents['R2000']['content']==record['documents'][0]
    docs,meta=rag.retrieve_with_metadata('How is Julian doing with fractions?')
    assert meta['database']=='Chroma (local)' and meta['student_ids']==['STU-1020']
    assert all(d['student_id']=='STU-1020' for d in docs)
    assert index.search(index.question_vector('fractions'),student_ids=['STU-9999'])==[]


def test_legacy_migration_preserves_existing_records():
    with db.connect() as con:
        con.execute("DELETE FROM interactions WHERE CAST(substr(id,2) AS INTEGER)>432")
        con.execute("DELETE FROM problem_skills WHERE CAST(substr(problem_id,2) AS INTEGER)>36")
        con.execute("DELETE FROM problems WHERE CAST(substr(id,2) AS INTEGER)>36")
        con.execute("DELETE FROM students WHERE id IN ('STU-1019','STU-1020')")
        con.execute('DELETE FROM dataset_migrations')
        con.execute("UPDATE interactions SET hints=5 WHERE id='R0001'")
        original=[tuple(r) for r in con.execute('SELECT * FROM interactions ORDER BY id')]
    db.initialize()
    db.initialize()
    with db.connect() as con:
        preserved=[tuple(r) for r in con.execute("SELECT * FROM interactions WHERE CAST(substr(id,2) AS INTEGER)<=432 ORDER BY id")]
        assert preserved==original
        assert con.execute('SELECT count(*) FROM interactions').fetchone()[0]==2000
        assert con.execute('SELECT count(DISTINCT prompt) FROM problems').fetchone()[0]==120
        assert not con.execute('SELECT id FROM problems WHERE id NOT IN (SELECT problem_id FROM interactions)').fetchall()
        for row in con.execute("SELECT content FROM documents WHERE kind='student'"):
            data=json.loads(row['content'])
            assert data['first_12']['problems']==data['last_12']['problems']==12


def test_failed_reindex_keeps_previous_chroma_collection(monkeypatch):
    index=vectors.get_index()
    original_name=index.collection.name
    signature=index.signature
    with db.connect() as con:
        con.execute("UPDATE interactions SET hints=hints+1 WHERE id='R0001'")
        db.build_index(con)
    def offline(texts):raise RuntimeError('embedding service unavailable')
    monkeypatch.setattr(index,'embed',offline)
    with pytest.raises(RuntimeError):index.ensure()
    assert index.signature==signature
    assert index.client.get_collection(original_name,embedding_function=None).count()==2034


def test_skill_question_uses_skill_statistics_not_overall_rate():
    docs=rag.retrieve('How is Julian doing with fractions? Give exact correctness.')
    profile=docs[0]
    assert profile['id']=='STU-1020' and profile['skill_scope']
    assert 'skill-specific correctness' in profile['content']
    assert 'Fraction operations' in profile['content']
    assert 'accuracy='+str(profile['data']['accuracy']) not in profile['content']
    assert 'First 12 sessions' not in profile['content']
    overall=rag.retrieve('How is Julian doing with fractions compared with his overall results?')[0]
    assert 'Separately, across ALL skills:' in overall['content']


def test_insight_preview_uses_skill_accuracy_and_inclusive_bounds(monkeypatch):
    with TestClient(app) as client:
        index=vectors.get_index()
        monkeypatch.setattr(index,'question_vector',lambda *_:pytest.fail('Preview must not embed a question'))
        all_students=client.post('/api/insights/preview',json={}).json()
        assert all_students['matching_count']==20 and all_students['problem_sessions']==2000
        body={'student_id':'STU-1020','skill_id':'S06','accuracy_min':70,'accuracy_max':70,'intent':'summary','focus':'Hints and confidence'}
        result=client.post('/api/insights/preview',json=body).json()
        assert result['matching_count']==1 and result['problem_sessions']==10
        assert result['students'][0]['accuracy']==70
        assert result['question'].startswith('Summarize') and 'Hints and confidence' in result['question']
        assert result['accuracy_basis']=='Fraction operations'
        assert client.post('/api/insights/preview',json={**body,'skill_id':None}).json()['matching_count']==0


def test_insight_validation():
    with TestClient(app) as client:
        for invalid in [{'accuracy_min':80,'accuracy_max':60},{'accuracy_min':-1},{'accuracy_max':101},{'skill_id':'S99'},{'intent':'unknown'}]:
            assert client.post('/api/insights/preview',json=invalid).status_code==422
            assert client.post('/api/retrieve',json={'question':'Help','filters':invalid}).status_code==422
        assert client.post('/api/insights/preview',json={'student_id':'missing'}).status_code==400
        assert client.post('/api/insights/preview',json={'focus':'x'*301}).status_code==422


def test_guided_retrieval_enforces_cohort_and_exact_aggregates(monkeypatch):
    from backend import insights
    index=vectors.get_index()
    searches=[]
    original=index.search
    def search(vector,**options):
        searches.append(options)
        return original(vector,**options)
    monkeypatch.setattr(index,'search',search)
    filters=insights.InsightFilters(skill_id='S06',accuracy_max=60,intent='overview').model_dump()
    docs,meta=rag.retrieve_with_metadata('Overview of fractions',filters=filters)
    cohort=docs[0]['data']
    ids={s['id'] for s in cohort['students']}
    assert 0<len(ids)<20 and all(s['accuracy']<=60 for s in cohort['students'])
    assert all(set(s['student_ids'])==ids and s['skill_ids']==['S06'] for s in searches)
    assert meta['matching_count']==len(ids)
    assert not any(d['kind'] in ['class','roster','student','skill'] for d in docs)
    assert all(d['student_id'] in ids and 'Fraction operations' in d['data']['skills'] for d in docs[1:])
    with db.connect() as con:
        rows=[r for r in db.histories(con) if r['student_id'] in ids and 'Fraction operations' in r['skills']]
    assert cohort['summary']==db.aggregate(rows)
    assert cohort['summary']['problems']==sum(s['problems'] for s in cohort['students'])
    scoped,meta=rag.retrieve_with_metadata('Discuss Maya and integers',student_id='STU-1020',filters={**filters,'accuracy_max':100})
    assert meta['student_ids']==['STU-1020'] and meta['skill_ids']==['S06']
    assert scoped[0]['data']['summary']['accuracy']==70
    assert 'Maya' not in scoped[0]['content']


def test_no_matching_students_skips_embedding_search_and_generation(monkeypatch):
    with TestClient(app) as client:
        index=vectors.get_index()
        def forbidden(*args,**kwargs):pytest.fail('Empty cohorts must not search or generate')
        monkeypatch.setattr(index,'question_vector',forbidden)
        monkeypatch.setattr(index,'search',forbidden)
        monkeypatch.setattr(httpx,'AsyncClient',forbidden)
        body={'question':'Recommend next steps','student_id':'STU-1020','filters':{'skill_id':'S06','accuracy_min':99}}
        result=client.post('/api/chat',json=body).json()
        assert result['sources']==[] and result['retrieval']['matching_count']==0
        assert result['metrics']['generated'] is False
        assert result['answer'].startswith('No students match')
        assert 'event: done' in client.post('/api/chat/stream',json=body).text


def test_guided_intent_cache_and_followup_filters(monkeypatch):
    captured=mock_model(monkeypatch,'Julian has 70% correctness in fractions [FILTERED].')
    with TestClient(app) as client:
        body={'question':'Describe the selected records','student_id':'STU-1020','filters':{'skill_id':'S06','accuracy_min':65,'accuracy_max':75,'intent':'summary'}}
        result=client.post('/api/chat',json=body).json()
        assert result['sources'][0]['id']=='FILTERED' and result['sources'][0]['cited']
        assert 'Summarize exact correctness' in captured[0]['messages'][0]['content']
        assert client.post('/api/chat',json=body).json()['metrics']['cached']
        changed={**body,'filters':{**body['filters'],'intent':'overview'}}
        assert not client.post('/api/chat',json=changed).json()['metrics']['cached']
        assert 'Give an overview' in captured[1]['messages'][0]['content']
        followup=client.post('/api/retrieve',json={**body,'question':'What about his confidence?','history':[{'role':'user','content':body['question']}]}).json()
        assert followup['retrieval']['student_ids']==['STU-1020']
        assert followup['retrieval']['skill_ids']==['S06']
        assert followup['sources'][0]['data']['summary']['accuracy']==70


def test_timing_migration_preserves_existing_observations_and_is_idempotent():
    with db.connect() as con:
        expected_times={r['id']:r['time_taken_seconds'] for r in con.execute('SELECT * FROM interactions')}
        con.execute('ALTER TABLE interactions DROP COLUMN time_taken_seconds')
        con.execute('DELETE FROM dataset_migrations WHERE version=3')
        original=[tuple(r) for r in con.execute('SELECT * FROM interactions ORDER BY id')]
    db.initialize()
    with db.connect() as con:
        rows=con.execute('SELECT * FROM interactions ORDER BY id').fetchall()
        assert [tuple(r)[:-1] for r in rows]==original
        assert {r['id']:r['time_taken_seconds'] for r in rows}==expected_times
        assert all(isinstance(r['time_taken_seconds'],int) and r['time_taken_seconds']>0 for r in rows)
        con.execute("UPDATE interactions SET time_taken_seconds=123 WHERE id='R0001'")
        for invalid in [0,-5]:
            with pytest.raises(sqlite3.IntegrityError):
                con.execute("UPDATE interactions SET time_taken_seconds=? WHERE id='R0002'",(invalid,))
    db.initialize()
    db.initialize()
    with db.connect() as con:
        assert con.execute("SELECT time_taken_seconds FROM interactions WHERE id='R0001'").fetchone()[0]==123
        assert con.execute('SELECT count(*) FROM dataset_migrations WHERE version=3').fetchone()[0]==1
        assert json.loads(con.execute("SELECT content FROM documents WHERE id='R0001'").fetchone()[0])['time_taken_seconds']==123


def test_timing_totals_and_skill_averages_match_full_sessions():
    with TestClient(app) as client:
        overview=client.get('/api/overview').json()
        student=client.get('/api/students/STU-1020').json()
        with db.connect() as con:
            count,total=con.execute('SELECT count(*),sum(time_taken_seconds) FROM interactions').fetchone()
        assert overview['summary']['total_time_seconds']==total
        assert overview['summary']['avg_time_seconds']==round(total/count,1)
        for skill in student['skills']:
            rows=[r for r in student['history'] if skill['name'] in r['skills']]
            seconds=sum(r['time_taken_seconds'] for r in rows)
            assert skill['total_time_seconds']==seconds
            assert skill['avg_time_seconds']==round(seconds/len(rows),1)
        assert db.aggregate([])['avg_time_seconds']==0
        assert db.aggregate([])['total_time_seconds']==0


def test_timing_is_in_chroma_vectors_metadata_and_prompt(monkeypatch):
    captured=mock_model(monkeypatch,'The timing is recorded in seconds [STU-1020].')
    with TestClient(app) as client:
        index=vectors.get_index()
        result=index.collection.get(ids=['R2000','STU-1020'],include=['documents','metadatas'])
        for content,metadata in zip(result['documents'],result['metadatas']):
            data=json.loads(content)
            fields=['time_taken_seconds'] if metadata['kind']=='problem' else ['total_time_seconds','avg_time_seconds']
            assert all(metadata[f]==data[f] for f in fields)
        assert 'Time taken ' in vectors.embedding_text(index.documents['R2000'])
        matches=index.collection.get(where={'time_taken_seconds':{'$gt':200}},include=['documents'])
        assert matches['ids'] and all(json.loads(d)['time_taken_seconds']>200 for d in matches['documents'])
        response=client.post('/api/chat',json={'question':'What is Julian’s average time taken for fractions?','student_id':'STU-1020'}).json()
        profile=response['sources'][0]
        assert 'avg_time_seconds' in profile['content'] and 'total_time_seconds' in profile['content']
        assert 'First 12 sessions' not in profile['content']
        assert 'avg_time_seconds' in captured[0]['messages'][0]['content']
        assert 'not model response latency' in captured[0]['messages'][0]['content']
        total_prompt=rag.retrieve('Give Julian’s total time spent on fractions.','STU-1020')[0]['content']
        assert 'total_time_seconds' in total_prompt and 'Separately, across ALL skills' not in total_prompt
        signature=index.signature
        with db.connect() as con:
            con.execute("UPDATE interactions SET time_taken_seconds=time_taken_seconds+10 WHERE id='R2000'")
            db.build_index(con)
        updated=vectors.get_index()
        assert updated.signature!=signature
        record=updated.collection.get(ids=['R2000'],include=['documents','metadatas'])
        assert record['metadatas'][0]['time_taken_seconds']==json.loads(record['documents'][0])['time_taken_seconds']


def test_guided_timing_uses_weighted_matching_sessions_only():
    with TestClient(app) as client:
        filters={'skill_id':'S06','accuracy_min':0,'accuracy_max':60,'intent':'overview'}
        preview=client.post('/api/insights/preview',json=filters).json()
        result=client.post('/api/retrieve',json={'question':'How much time do these students spend on fractions?','filters':filters}).json()
        cohort=result['sources'][0]['data']
        ids={s['id'] for s in preview['students']}
        with db.connect() as con:
            rows=[r for r in db.histories(con) if r['student_id'] in ids and 'Fraction operations' in r['skills']]
        total=sum(r['time_taken_seconds'] for r in rows)
        average=round(total/len(rows),1)
        assert preview['total_time_seconds']==cohort['summary']['total_time_seconds']==total
        assert preview['avg_time_seconds']==cohort['summary']['avg_time_seconds']==average
        assert f'Average time per problem session: {average} seconds' in result['sources'][0]['content']
        assert all('time_taken_seconds=' in d['content'] for d in result['sources'][1:])
        empty=client.post('/api/insights/preview',json={**filters,'student_id':'STU-1020'}).json()
        assert empty['matching_count']==empty['avg_time_seconds']==empty['total_time_seconds']==0
