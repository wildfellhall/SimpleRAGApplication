"""Persistent local Chroma database; questions and evidence share a local embedding model."""
from collections import OrderedDict
import hashlib
import json
import os
import threading
import textwrap
from pathlib import Path

import httpx
import chromadb
from chromadb.config import Settings
from . import db

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-minilm:22m')
EMBEDDING_URL = os.getenv('EMBEDDING_URL', 'http://127.0.0.1:11434').rstrip('/')
COLLECTION = 'learning_evidence'
INDEX_VERSION = 'forma-chroma-v2-session-time'


def embedding_text(doc):
    """Short semantic descriptions avoid silently truncating long aggregate JSON."""
    data = json.loads(doc['content'])
    if doc['kind'] == 'problem':
        affect = ', '.join(f'{key} {data[key]}' for key in db.AFFECTS)
        return (f"{data['student_name']} {data['student_id']}. Practice problem: {data['prompt']}. "
                f"Skills: {', '.join(data['skills'])}. {'Correct' if data['correct'] else 'Incorrect, struggling, needs support'}. "
                f"Attempts {data['attempts']}, hints {data['hints']}. Time taken {data['time_taken_seconds']} seconds. {affect}.")
    if doc['kind'] == 'skill':
        return (f"Class performance in {data['skill']}. {data['keywords']}. "
                'Which students need help with this skill? Correctness, practice, difficulty, hints, confusion, '
                'determination, confidence, frustration, time spent answering, average time per problem and next steps.')
    if doc['kind'] == 'student':
        return (f"{data['name']} {data['id']} learning progress and performance. Strengths, weaknesses, "
                'skills, improvement over time, correctness, attempts, hints, confusion, determination, confidence, frustration, time taken answering problems. '
                + ', '.join(s['name'] for s in data['skills']))
    return (doc['title'] + '. Class overview, all students, compare progress, who needs support, '
            'correctness, skills, hints, confusion, determination, confidence, frustration, total and average time spent on problems.')


class VectorIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        self.collection = None
        self.http = httpx.Client(timeout=httpx.Timeout(90, connect=5))
        self.lock = threading.RLock()
        self.query_cache = OrderedDict()
        self.signature = None
        self.documents = {}
        self.build_count = 0

    def embed(self, texts):
        try:
            response = self.http.post(EMBEDDING_URL + '/api/embed', json={
                'model': EMBEDDING_MODEL, 'input': texts, 'truncate': False, 'keep_alive': '30m'})
            response.raise_for_status()
            vectors = response.json()['embeddings']
            if len(vectors) != len(texts) or not vectors or not vectors[0]:
                raise ValueError('Embedding service returned invalid vectors')
            return vectors
        except httpx.HTTPError as exc:
            raise RuntimeError(f'Local embeddings are unavailable. Start Ollama and run ollama pull {EMBEDDING_MODEL}.') from exc

    def ensure(self):
        with self.lock, db.connect() as con:
            documents = [dict(r) for r in con.execute('SELECT * FROM documents ORDER BY id')]
            signature = hashlib.sha256((INDEX_VERSION + EMBEDDING_MODEL + json.dumps(documents, sort_keys=True)).encode()).hexdigest()
            if signature == self.signature:
                return signature
            manifest = self.path / 'forma-index.json'
            try:
                saved = json.loads(manifest.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                saved = {}
            name = COLLECTION + '_' + signature[:16]
            names = {c.name for c in self.client.list_collections()}
            collection = self.client.get_collection(name,embedding_function=None) if name in names else None
            if saved.get('signature') != signature or collection is None or collection.count()!=len(documents):
                embedded = []
                for start in range(0, len(documents), 32):
                    embedded.extend(self.embed([embedding_text(d) for d in documents[start:start+32]]))
                # Publish a new complete collection before retiring the previous version.
                if collection is not None:
                    self.client.delete_collection(name)
                collection = self.client.create_collection(name,embedding_function=None,
                    configuration={'hnsw':{'space':'cosine'}},metadata={'signature':signature,'embedding_model':EMBEDDING_MODEL})
                metadatas = []
                for doc in documents:
                    data = json.loads(doc['content'])
                    skills = data.get('skills', []) if isinstance(data,dict) else []
                    skill_ids = [skill['id'] for skill in skills if isinstance(skill,dict)]
                    if doc['kind']=='problem':
                        skill_ids = [sid for sid,skill_name,_,_ in db.SKILLS if skill_name in skills]
                    if doc['kind']=='skill':
                        skill_ids = [doc['id']]
                    metadata = {'kind':doc['kind'],'student_id':doc['student_id'] or '', 'title':doc['title']}
                    metadata.update({f'skill_{sid}':True for sid in skill_ids})
                    if doc['kind']=='problem':
                        metadata.update({key:data[key] for key in ['problem_id','occurred_at','correct','attempts','hints','time_taken_seconds',*db.AFFECTS]})
                    if isinstance(data,dict) and 'total_time_seconds' in data:
                        metadata.update({key:data[key] for key in ['total_time_seconds','avg_time_seconds']})
                    metadatas.append(metadata)
                for start in range(0,len(documents),128):
                    batch=documents[start:start+128]
                    collection.upsert(ids=[d['id'] for d in batch],embeddings=embedded[start:start+128],
                        documents=[d['content'] for d in batch],metadatas=metadatas[start:start+128])
                temporary=manifest.with_suffix('.tmp')
                temporary.write_text(json.dumps({'signature':signature,'model':EMBEDDING_MODEL,'collection':name,
                    'documents':len(documents),'dimensions':len(embedded[0]),'database':'Chroma'}))
                temporary.replace(manifest)
                for old_name in names:
                    if old_name.startswith(COLLECTION+'_') and old_name!=name:
                        self.client.delete_collection(old_name)
                self.build_count += 1
            self.collection = collection
            self.signature = signature
            # Retrieval evidence is read back from Chroma, including full source documents.
            stored = collection.get(include=['documents','metadatas'])
            self.documents = {id:{'id':id,'kind':metadata['kind'],'student_id':metadata['student_id'] or None,
                'title':metadata['title'],'content':content} for id,metadata,content in
                zip(stored['ids'],stored['metadatas'],stored['documents'])}
            self.query_cache.clear()
            return signature

    def question_vector(self, question):
        with self.lock:
            if question in self.query_cache:
                self.query_cache.move_to_end(question)
                return self.query_cache[question]
            # Embed every part of a long question; MiniLM has a short context window.
            parts = textwrap.wrap(question,width=240,break_long_words=True,break_on_hyphens=False) or [question]
            embedded = self.embed(parts)
            weights = [len(part) for part in parts]
            vector = [sum(value*weight for value,weight in zip(values,weights))/sum(weights) for values in zip(*embedded)]
            self.query_cache[question] = vector
            if len(self.query_cache)>256:
                self.query_cache.popitem(last=False)
            return vector

    def search(self, vector, *, kinds=None, student_ids=None, skill_ids=None, limit=3):
        conditions = []
        for field,values in [('kind',kinds),('student_id',student_ids)]:
            if values:
                conditions.append({field:{'$in':values}})
        if skill_ids:
            skills = [{f'skill_{sid}':{'$eq':True}} for sid in skill_ids]
            conditions.append(skills[0] if len(skills)==1 else {'$or':skills})
        where = conditions[0] if len(conditions)==1 else {'$and':conditions} if conditions else None
        with self.lock:
            result = self.collection.query(query_embeddings=[vector],where=where,
                n_results=limit,include=['distances'])
        return [{'id':id,'score':round(1-distance,5)} for id,distance in zip(result['ids'][0],result['distances'][0])]

    def close(self):
        self.http.close()
        self.collection = None
        self.client.close()
        self.client = None


_index = None
_index_lock = threading.Lock()


def get_index():
    global _index
    path = Path(os.getenv('CHROMA_PATH',os.getenv('VECTOR_PATH',str(db.DB_PATH.parent/'chroma'))))
    with _index_lock:
        if _index is None or _index.path != path:
            if _index is not None:
                _index.close()
            _index = VectorIndex(path)
        _index.ensure()
        return _index


def close_index():
    global _index
    with _index_lock:
        if _index:
            _index.close()
            _index = None


if __name__ == '__main__':
    db.initialize()
    with db.connect() as con:
        db.build_index(con)
    index = get_index()
    print(json.dumps({'documents':len(index.documents),'model':EMBEDDING_MODEL,'path':str(index.path)}))
    close_index()
