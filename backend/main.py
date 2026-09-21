import asyncio
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Literal
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .db import ROOT, aggregate, build_index, connect, histories, initialize, skill_stats
from . import rag, vectors, insights

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    initialize()
    with connect() as con:
        build_index(con)
    try:
        await asyncio.to_thread(vectors.get_index)
        app.state.retrieval_error = None
    except (RuntimeError, ValueError) as exc:
        # Keep student browsing available when the embedding service is down.
        logger.warning('Vector index not ready: %s', exc)
        app.state.retrieval_error = str(exc)
    yield
    vectors.close_index()
    rag.ANSWER_CACHE.clear()

app = FastAPI(title='Forma · Local learning insights', lifespan=lifespan)
chat_lock = asyncio.Lock()

class ConversationScope(BaseModel):
    student_ids:list[str]=Field(default_factory=list,max_length=20)
    skill_ids:list[str]=Field(default_factory=list,max_length=12)

class Message(BaseModel):
    role: Literal['user','assistant']
    content: str = Field(min_length=1,max_length=16000)
    scope: ConversationScope | None = None

class ChatRequest(BaseModel):
    filters: insights.InsightFilters | None = None
    question: str = Field(min_length=1,max_length=4000)
    student_id: str | None = None
    history: list[Message] = Field(default_factory=list,max_length=1000)

@app.get('/api/health')
async def health():
    online = await rag.model_health()
    return {'status':'ok','model_online':online,'model':rag.MODEL_LABEL,'local':True,
            'retrieval':'Chroma local + '+vectors.EMBEDDING_MODEL,'retrieval_error':getattr(app.state,'retrieval_error',None)}

@app.get('/api/overview')
def overview():
    with connect() as con:
        rows = histories(con)
        students = [dict(s) for s in con.execute('SELECT * FROM students ORDER BY name')]
        dates = con.execute('SELECT min(occurred_at),max(occurred_at) FROM interactions').fetchone()
        date_range = ' – '.join(datetime.fromisoformat(d).strftime('%b %d, %Y') for d in dates)
        problem_count = con.execute('SELECT count(*) FROM problems').fetchone()[0]
        return {'problem_count':problem_count,'students':[{**s,**aggregate([r for r in rows if r['student_id']==s['id']])} for s in students],
                'summary':aggregate(rows),'skills':skill_stats(rows),'date_range':date_range,'synthetic':True}

@app.get('/api/students/{student_id}')
def student(student_id: str):
    with connect() as con:
        s = con.execute('SELECT * FROM students WHERE id=?',(student_id,)).fetchone()
        if not s: raise HTTPException(404,'Student not found')
        rows = histories(con,student_id)
        return {**dict(s),'summary':aggregate(rows),'skills':skill_stats(rows),'history':rows}


def error_response(exc):
    if isinstance(exc,httpx.TimeoutException):
        return HTTPException(504,'The local model took too long. Try a more focused question.')
    if isinstance(exc,httpx.HTTPError):
        return HTTPException(503,'The local model is unavailable. Start it with npm run model, then try again.')
    if isinstance(exc,RuntimeError):
        return HTTPException(503,str(exc))
    return HTTPException(400,str(exc))


def validate(request):
    question = request.question.strip()
    if not question: raise HTTPException(422,'Please enter a question.')
    return question


@app.post('/api/insights/preview')
async def preview_insight(request:insights.InsightPreviewRequest):
    def build():
        index=vectors.get_index()
        return insights.preview(index.documents,request)
    try:
        return await asyncio.to_thread(build)
    except (RuntimeError,ValueError) as exc:
        raise error_response(exc)


@app.post('/api/retrieve')
async def retrieve(request:ChatRequest):
    """Inspect exactly how a question queries Chroma, without running generation."""
    try:
        evidence, metadata = await asyncio.to_thread(rag.retrieve_with_metadata,validate(request),request.student_id,[m.model_dump() for m in request.history],request.filters.model_dump() if request.filters else None)
        app.state.retrieval_error = None
        return {'sources':evidence,'retrieval':metadata}
    except (httpx.HTTPError,RuntimeError,ValueError) as exc:
        raise error_response(exc)


@app.post('/api/chat')
async def chat(request: ChatRequest):
    question = validate(request)
    if chat_lock.locked(): raise HTTPException(429,'The local model is answering another question. Try again shortly.')
    try:
        async with chat_lock:
            result = await rag.answer(question,request.student_id,[m.model_dump() for m in request.history],request.filters.model_dump() if request.filters else None)
            app.state.retrieval_error = None
            return result
    except (httpx.HTTPError,RuntimeError,ValueError) as exc:
        raise error_response(exc)


@app.post('/api/chat/stream')
async def chat_stream(request:ChatRequest):
    question = validate(request)
    if chat_lock.locked(): raise HTTPException(429,'The local model is answering another question. Try again shortly.')
    await chat_lock.acquire()
    try:
        prepared = await rag.prepare(question,request.student_id,[m.model_dump() for m in request.history],request.filters.model_dump() if request.filters else None)
        app.state.retrieval_error = None
    except BaseException as exc:
        chat_lock.release()
        if isinstance(exc,(httpx.HTTPError,RuntimeError,ValueError)):
            raise error_response(exc)
        raise
    async def events():
        try:
            async for event in rag.generate(prepared):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'],ensure_ascii=False)}\n\n"
        except (httpx.HTTPError,RuntimeError,ValueError) as exc:
            yield 'event: error\ndata: '+json.dumps({'detail':error_response(exc).detail})+'\n\n'
        finally:
            chat_lock.release()
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

if (ROOT/'frontend/dist').exists():
    app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')
