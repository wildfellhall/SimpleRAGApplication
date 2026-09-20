"""Explicit teacher controls and exact cohort selection for guided insights."""
import json
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from .db import SKILLS, aggregate

METRICS = ('problems','correct','accuracy','attempts','hints','confusion','determination','confidence','frustration')
INTENTS = {
    'recommendations': 'Give 2–3 specific teaching recommendations, each tied to the recorded evidence. Distinguish suggestions from observations.',
    'summary': 'Summarize exact correctness, attempt and hint totals, and recorded affect averages concisely. Focus on observations rather than teaching recommendations.',
    'overview': 'Give an overview of the matching group: its size, exact correctness, and key performance and affect patterns. Do not describe it as the whole class unless everyone matches.',
}

class InsightFilters(BaseModel):
    skill_id: str | None = None
    accuracy_min: float = Field(default=0,ge=0,le=100)
    accuracy_max: float = Field(default=100,ge=0,le=100)
    intent: Literal['recommendations','summary','overview'] = 'recommendations'

    @model_validator(mode='after')
    def valid_selection(self):
        if self.accuracy_min>self.accuracy_max:
            raise ValueError('Minimum accuracy must not exceed maximum accuracy.')
        if self.skill_id and self.skill_id not in {s[0] for s in SKILLS}:
            raise ValueError('Unknown skill ID')
        return self

class InsightPreviewRequest(InsightFilters):
    student_id: str | None = None
    focus: str = Field(default='',max_length=300)


def select_students(documents,student_id,filters):
    profiles = [json.loads(d['content']) for d in documents.values() if d['kind']=='student']
    if student_id and student_id not in {p['id'] for p in profiles}:
        raise ValueError('Unknown student ID')
    selected=[]
    for profile in profiles:
        if student_id and profile['id']!=student_id:continue
        stats = next((s for s in profile['skills'] if s['id']==filters['skill_id']),None) if filters.get('skill_id') else profile
        if stats and stats['problems'] and filters['accuracy_min']<=stats['accuracy']<=filters['accuracy_max']:
            selected.append({'id':profile['id'],'name':profile['name'],**{k:stats[k] for k in METRICS}})
    return sorted(selected,key=lambda s:(s['accuracy'],s['name']))


def skill_name(filters):
    return next((s[1] for s in SKILLS if s[0]==filters.get('skill_id')),'all skills')


def scope_text(student_id,filters,documents):
    name = json.loads(documents[student_id]['content'])['name'] if student_id else 'students'
    return f"{name} with {filters['accuracy_min']:g}–{filters['accuracy_max']:g}% accuracy in {skill_name(filters)}"


def preview(documents,request):
    filters = InsightFilters(**request.model_dump()).model_dump()
    students = select_students(documents,request.student_id,filters)
    subject = scope_text(request.student_id,filters,documents)
    verb={'recommendations':'Recommend teaching next steps for','summary':'Summarize the learning records for','overview':'Give a learning overview for'}[request.intent]
    question=f'{verb} {subject}. Use only matching students and cite the evidence.'
    if request.focus.strip():question+=' Additional focus: '+request.focus.strip()
    return {'question':question,'filters':filters,'students':students,'matching_count':len(students),
            'problem_sessions':sum(s['problems'] for s in students),'accuracy_basis':skill_name(filters)}


def cohort_document(documents,students,student_id,filters):
    ids={s['id'] for s in students}
    records=[]
    name=skill_name(filters)
    for doc in documents.values():
        if doc['kind']!='problem' or doc['student_id'] not in ids:continue
        row=json.loads(doc['content'])
        if not filters.get('skill_id') or name in row['skills']:records.append(row)
    data={'scope':scope_text(student_id,filters,documents),'student_count':len(students),
          'accuracy_basis':name,'accuracy_range':[filters['accuracy_min'],filters['accuracy_max']],
          'summary':aggregate(records),'students':students}
    return {'id':'FILTERED','kind':'cohort','student_id':student_id,'title':'Selected students · filtered learning summary',
            'data':data,'selection':'explicit student, skill, and accuracy filters','score':None,'skill_scope':False}
