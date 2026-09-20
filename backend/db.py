"""Versioned synthetic learning records and exact aggregate evidence for Chroma."""
import json
import os
import random
import sqlite3
from fractions import Fraction
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv('DATABASE_PATH', str(ROOT / 'data' / 'students.sqlite3')))
SKILLS = [
    ('S01', 'Combining like terms', 'like terms simplify expressions coefficients', [('3x + 5x - 2', '8x - 2'), ('7a - 2a + 4', '5a + 4'), ('2y + 6 + 3y - 1', '5y + 5')]),
    ('S02', 'Distributive property', 'expand parentheses distribution', [('3(x + 4)', '3x + 12'), ('-2(3y - 5)', '-6y + 10'), ('4(2a + 1) - a', '7a + 4')]),
    ('S03', 'One-step equations', 'solve equations inverse operations', [('x + 7 = 15', 'x = 8'), ('4x = 28', 'x = 7'), ('y - 9 = -3', 'y = 6')]),
    ('S04', 'Two-step equations', 'solve equations isolate variable', [('2x + 3 = 17', 'x = 7'), ('5y - 4 = 21', 'y = 5'), ('x/3 + 2 = 6', 'x = 12')]),
    ('S05', 'Integer operations', 'negative numbers signed arithmetic', [('-8 + 13', '5'), ('-4 × -6', '24'), ('9 - 15', '-6')]),
    ('S06', 'Fraction operations', 'fractions common denominators', [('1/3 + 1/4', '7/12'), ('3/4 × 2/3', '1/2'), ('5/6 - 1/3', '1/2')]),
    ('S07', 'Ratios and proportions', 'ratio proportion equivalent rate', [('2/5 = x/20', 'x = 8'), ('3 notebooks cost $12. Cost of 5?', '$20'), ('Simplify the ratio 12:18', '2:3')]),
    ('S08', 'Percentages', 'percent discount percentage', [('25% of 80', '20'), ('A $50 shirt is 20% off. Sale price?', '$40'), ('15 is what percent of 60?', '25%')]),
    ('S09', 'Order of operations', 'pemdas arithmetic parentheses', [('3 + 4 × 2', '11'), ('(12 - 4) / 2 + 3', '7'), ('2² + 3 × 5', '19')]),
    ('S10', 'Evaluating expressions', 'substitute substitution variables', [('2x + 5 when x = 3', '11'), ('a² - 2 when a = 4', '14'), ('3m - n when m = 2, n = 5', '1')]),
    ('S11', 'Inequalities', 'inequality less greater number line', [('x + 3 > 8', 'x > 5'), ('2x ≤ 10', 'x ≤ 5'), ('-3x > 12', 'x < -4')]),
    ('S12', 'Coordinate plane', 'coordinates graph plotting ordered pairs', [('Which quadrant contains (3, -2)?', 'IV'), ('Reflect (2, 5) across the y-axis', '(-2, 5)'), ('Distance from (1, 2) to (1, 7)', '5')]),
]
NAMES = ['Maya Chen', 'Noah Williams', 'Aisha Patel', 'Liam Garcia', 'Sofia Martinez', 'Ethan Brooks', 'Amara Okafor', 'Oliver Kim', 'Isabella Rossi', 'Lucas Thompson', 'Zoe Anderson', 'Arjun Shah', 'Chloe Nguyen', 'Mateo Rivera', 'Grace Park', 'Elijah Johnson', 'Layla Hassan', 'Henry Wilson']
NAMES += ['Audrey Bell', 'Julian Reed']
SESSIONS_PER_STUDENT = 100
DATASET_VERSION = 2
AFFECTS = ('confusion', 'determination', 'confidence', 'frustration')


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys = ON')
    return con


def initialize():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS students(id TEXT PRIMARY KEY, name TEXT NOT NULL, grade INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY, name TEXT NOT NULL, keywords TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS problems(id TEXT PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS problem_skills(problem_id TEXT REFERENCES problems(id), skill_id TEXT REFERENCES skills(id), PRIMARY KEY(problem_id, skill_id));
        CREATE TABLE IF NOT EXISTS interactions(
            id TEXT PRIMARY KEY, student_id TEXT NOT NULL REFERENCES students(id),
            problem_id TEXT NOT NULL REFERENCES problems(id), occurred_at TEXT NOT NULL,
            correct INTEGER NOT NULL CHECK(correct IN (0,1)), attempts INTEGER NOT NULL CHECK(attempts BETWEEN 1 AND 6),
            hints INTEGER NOT NULL CHECK(hints BETWEEN 0 AND 5),
            confusion REAL NOT NULL CHECK(confusion BETWEEN 0 AND 1),
            determination REAL NOT NULL CHECK(determination BETWEEN 0 AND 1),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
            frustration REAL NOT NULL CHECK(frustration BETWEEN 0 AND 1));
        CREATE INDEX IF NOT EXISTS interactions_student ON interactions(student_id, occurred_at);
        CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, kind TEXT NOT NULL, student_id TEXT, title TEXT NOT NULL, content TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(id UNINDEXED, content, tokenize='porter unicode61');
        ''')
        con.execute('CREATE TABLE IF NOT EXISTS dataset_migrations(version INTEGER PRIMARY KEY)')
        if con.execute('SELECT count(*) FROM students').fetchone()[0]:
            expand_dataset(con)
            return
        rng = random.Random(20260920)
        for sid, name, keywords, problems in SKILLS:
            con.execute('INSERT INTO skills VALUES(?,?,?)', (sid, name, keywords))
            for i, (prompt, answer) in enumerate(problems):
                pid = f'P{(int(sid[1:])-1)*3+i+1:03}'
                con.execute('INSERT INTO problems VALUES(?,?,?)', (pid, prompt, answer))
                con.execute('INSERT INTO problem_skills VALUES(?,?)', (pid, sid))
        # Multi-skill problems genuinely require both skills.
        for pid, sid in [('P006', 'S01'), ('P012', 'S06'), ('P018', 'S09'), ('P023', 'S07'), ('P030', 'S05'), ('P033', 'S05')]:
            con.execute('INSERT INTO problem_skills VALUES(?,?)', (pid, sid))
        for i, name in enumerate(NAMES[:18]):
            student = f'STU-{1001+i}'
            con.execute('INSERT INTO students VALUES(?,?,?)', (student, name, 7 if i % 3 else 8))
            strength = .49 + (i % 6) * .073
            for day in range(24):
                skill = (day + i * 3) % 12
                pid = f'P{skill*3+(day//12+i)%3+1:03}'
                chance = strength + (.1 if day >= 12 else -.04) - (.19 if skill in [0,1,3,5,10] else 0)
                correct = rng.random() < chance
                attempts = rng.choices([1,2,3,4], [5,3,1,1] if correct else [1,2,4,3])[0]
                hints = min(5, max(0, attempts - 1 + rng.choice([-1,0,1])))
                confusion = round(min(.98, max(.05, (.25 if correct else .62) + rng.uniform(-.18,.2))), 2)
                frustration = round(min(.98, max(.03, confusion * .7 + hints * .065 + rng.uniform(-.15,.12))), 2)
                confidence = round(min(.98, max(.05, 1 - confusion + rng.uniform(-.12,.12))), 2)
                determination = round(rng.uniform(.45,.95), 2)
                timestamp = (datetime(2026,8,24,9,0) + timedelta(days=day, minutes=i*3)).isoformat()
                con.execute('INSERT INTO interactions VALUES(?,?,?,?,?,?,?,?,?,?,?)', (f'R{i*24+day+1:04}', student, pid, timestamp, int(correct), attempts, hints, confusion, determination, confidence, frustration))
        expand_dataset(con)



def extra_problems(skill_id):
    """Seven exact-answer variants per skill; preserve the original 36 problem IDs."""
    result = []
    for n in range(1, 8):
        a, b = n+4, n+6
        variants = {
            'S01': (f'{a}x + {b}x - {n}', f'{a+b}x - {n}'),
            'S02': (f'{a}(x + {b})', f'{a}x + {a*b}'),
            'S03': (f'x + {a} = {a+b}', f'x = {b}'),
            'S04': (f'{a}x + {b} = {a*n+b}', f'x = {n}'),
            'S05': (f'-{a} + {a+b}', str(b)),
            'S06': (f'1/{a} + 1/{a+1}', str(Fraction(1,a)+Fraction(1,a+1))),
            'S07': (f'{a}/{b} = x/{b*4}', f'x = {a*4}'),
            'S08': (f'{n*5}% of 200', str(n*10)),
            'S09': (f'{n} + {a} × {b}', str(n+a*b)),
            'S10': (f'{a}x + {b} when x = {n}', str(a*n+b)),
            'S11': (f'x + {a} > {a+b}', f'x > {b}'),
            'S12': (f'Reflect ({a}, {b}) across the x-axis', f'({a}, -{b})'),
        }
        result.append(variants[skill_id])
    return result


def expand_dataset(con):
    """Add history exactly once, without modifying existing student/session IDs or values."""
    if con.execute('SELECT 1 FROM dataset_migrations WHERE version=?',(DATASET_VERSION,)).fetchone():
        return
    for index, (sid, _, _, _) in enumerate(SKILLS):
        for j,(prompt,answer) in enumerate(extra_problems(sid)):
            pid = f'P{37+index*7+j:03}'
            con.execute('INSERT OR IGNORE INTO problems VALUES(?,?,?)',(pid,prompt,answer))
            con.execute('INSERT OR IGNORE INTO problem_skills VALUES(?,?)',(pid,sid))
    pools = {sid:[r[0] for r in con.execute('SELECT problem_id FROM problem_skills WHERE skill_id=? ORDER BY problem_id',(sid,))] for sid,_,_,_ in SKILLS}
    next_record = max(int(r[0][1:]) for r in con.execute('SELECT id FROM interactions'))+1
    for i,name in enumerate(NAMES):
        student = f'STU-{1001+i}'
        con.execute('INSERT OR IGNORE INTO students VALUES(?,?,?)',(student,name,7 if i%3 else 8))
        existing = con.execute('SELECT count(*) FROM interactions WHERE student_id=?',(student,)).fetchone()[0]
        rng = random.Random(20260920+i)
        for day in range(max(0, SESSIONS_PER_STUDENT-existing)):
            skill = (day+i*3)%len(SKILLS)
            pool = pools[SKILLS[skill][0]]
            pid = pool[(day//len(SKILLS)+i)%len(pool)]
            chance = .40+(i%6)*.075+day*.0015-(.13 if skill in (0,1,3,5,10) else 0)
            correct = rng.random()<chance
            attempts = rng.choices([1,2,3,4,5],[5,3,1,1,1] if correct else [1,2,4,3,2])[0]
            hints = min(5,max(0,attempts-1+rng.choice([-1,0,1])))
            confusion = round(min(.98,max(.05,(.25 if correct else .62)+rng.uniform(-.18,.2))),2)
            frustration = round(min(.98,max(.03,confusion*.7+hints*.065+rng.uniform(-.15,.12))),2)
            confidence = round(min(.98,max(.05,1-confusion+rng.uniform(-.12,.12))),2)
            determination = round(rng.uniform(.45,.95),2)
            timestamp = (datetime(2026,6,9,9)+timedelta(days=day,minutes=i*3)).isoformat()
            con.execute('INSERT INTO interactions VALUES(?,?,?,?,?,?,?,?,?,?,?)',(f'R{next_record:04}',student,pid,timestamp,int(correct),attempts,hints,confusion,determination,confidence,frustration))
            next_record += 1
    con.execute('INSERT INTO dataset_migrations VALUES(?)',(DATASET_VERSION,))
    build_index(con)

def histories(con, student_id=None):
    rows = con.execute('''SELECT i.*, s.name student_name, p.prompt, p.answer,
        group_concat(sk.name, '|') skills FROM interactions i
        JOIN students s ON s.id=i.student_id JOIN problems p ON p.id=i.problem_id
        JOIN problem_skills ps ON ps.problem_id=p.id JOIN skills sk ON sk.id=ps.skill_id
        WHERE (? IS NULL OR i.student_id=?) GROUP BY i.id ORDER BY i.occurred_at DESC''', (student_id, student_id)).fetchall()
    return [{**dict(r), 'correct': bool(r['correct']), 'skills': r['skills'].split('|')} for r in rows]


def aggregate(rows):
    n = len(rows)
    return {'problems': n, 'correct': sum(r['correct'] for r in rows),
            'accuracy': round(100 * sum(r['correct'] for r in rows) / n, 1) if n else 0,
            'attempts': sum(r['attempts'] for r in rows), 'hints': sum(r['hints'] for r in rows),
            **{a: round(sum(r[a] for r in rows) / n, 2) if n else 0 for a in AFFECTS}}


def skill_stats(rows):
    return [{'id': sid, 'name': name, **aggregate([r for r in rows if name in r['skills']])} for sid, name, _, _ in SKILLS]


def build_index(con):
    con.execute('DELETE FROM documents')
    con.execute('DELETE FROM documents_fts')
    def add(id, kind, sid, title, content):
        con.execute('INSERT INTO documents VALUES(?,?,?,?,?)', (id, kind, sid, title, content))
        con.execute('INSERT INTO documents_fts VALUES(?,?)', (id, content))
    rows = histories(con)
    count = con.execute('SELECT count(*) FROM students').fetchone()[0]
    dates = con.execute('SELECT min(occurred_at),max(occurred_at) FROM interactions').fetchone()
    add('CLASS', 'class', None, f'Class overview · all {count} students', json.dumps({'scope':f'all {count} fictional students, {dates[0][:10]} to {dates[1][:10]}', **aggregate(rows), 'skills':skill_stats(rows)}))
    for s in con.execute('SELECT * FROM students ORDER BY id').fetchall():
        sr = [r for r in rows if r['student_id']==s['id']]
        chronology = sorted(sr, key=lambda r:r['occurred_at'])
        data = {**dict(s), **aggregate(sr), 'first_12':aggregate(chronology[:12]), 'last_12':aggregate(chronology[-12:]), 'skills':skill_stats(sr)}
        add(s['id'], 'student', s['id'], s['name']+' · learning summary', json.dumps(data))
    roster = [{'id':s['id'], 'name':s['name'], **aggregate([r for r in rows if r['student_id']==s['id']])} for s in con.execute('SELECT * FROM students ORDER BY id')]
    add('ROSTER', 'roster', None, f'All {count} students · comparable outcomes', json.dumps(roster))
    for sid, name, keywords, _ in SKILLS:
        sr = [r for r in rows if name in r['skills']]
        data = {'skill':name, 'keywords':keywords, **aggregate(sr), 'students':[{'id':s['id'], 'name':s['name'], **aggregate([r for r in sr if r['student_id']==s['id']])} for s in con.execute('SELECT * FROM students')]}
        add(sid, 'skill', None, name+' · class skill summary', json.dumps(data))
    for r in rows:
        add(r['id'], 'problem', r['student_id'], f"{r['student_name']} · {r['problem_id']} · {r['occurred_at'][:10]}", json.dumps(r))


if __name__ == '__main__':
    initialize()
    with connect() as con:
        print(json.dumps({'database':str(DB_PATH), 'students':con.execute('SELECT count(*) FROM students').fetchone()[0], 'interactions':con.execute('SELECT count(*) FROM interactions').fetchone()[0]}))
