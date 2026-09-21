import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ArrowUp, ArrowUpRight, BookOpen, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Database, GraduationCap, Layers3, LoaderCircle, MessageSquare, Plus, Search, ShieldCheck, SlidersHorizontal, Sparkles, Users, X } from 'lucide-react'
import Markdown from 'react-markdown'
import { streamChat } from './stream'
import InsightBuilder, { type InsightFilters } from './InsightBuilder'

type Stats = { total_time_seconds:number; avg_time_seconds:number; problems:number; correct:number; accuracy:number; attempts:number; hints:number; confusion:number; determination:number; confidence:number; frustration:number }
type Student = Stats & {id:string; name:string; grade:number}
type Skill = Stats & {id:string; name:string}
type RecordRow = {time_taken_seconds:number; id:string; problem_id:string; prompt:string; answer:string; occurred_at:string; correct:boolean; attempts:number; hints:number; skills:string[]; confusion:number; determination:number; confidence:number; frustration:number}
type Detail = {id:string; name:string; grade:number; summary:Stats; skills:Skill[]; history:RecordRow[]}
type Overview = {problem_count:number; students:Student[]; summary:Stats; skills:Skill[]; date_range:string}
type Source = {id:string; kind:string; title:string; cited:boolean; data:Record<string,unknown>|Record<string,unknown>[]}
type Retrieval = {student_ids?:string[];skill_ids?:string[];query:string; retrieval_ms:number; matches:{id:string;score:number}[]}
type Metrics = {cached:boolean; total_ms:number; first_token_ms:number|null}
type ChatMessage = {complete?:boolean;failed?:boolean;retrieval?:Retrieval; metrics?:Metrics; role:'user'|'assistant'; content:string; sources?:Source[]; model?:string; truncated?:boolean}
const affects = ['confusion','determination','confidence','frustration'] as const
const prompts = [
  {icon:Users, label:'Find a starting point', text:'Which students could use the most support?', question:'Which students could use the most support? Use correctness and confusion, and suggest a next step.'},
  {icon:Layers3, label:'Look closer at a skill', text:'How is the class doing with like terms?', question:'How is the class doing with combining like terms? Who needs support and what should we practice?'},
  {icon:Sparkles, label:'Understand the experience', text:'Who is showing determination despite difficulty?', question:'Which students show high determination despite difficulty? Use the recorded affect and correctness scores.'},
]
function duration(seconds:number){ const minutes=Math.floor(seconds/60);const remainder=Math.round(seconds%60);return minutes?`${minutes}m ${remainder}s`:`${remainder}s` }
function initials(name:string) { return name.split(' ').map(w=>w[0]).join('') }
async function api<T>(url:string, options?:RequestInit):Promise<T> {
  const r=await fetch(url,options)
  if(!r.ok) { let message='Something went wrong. Please try again.'; try {const d=await r.json(); if(typeof d.detail==='string')message=d.detail} catch { /* preserve readable fallback */ } throw new Error(message) }
  return r.json()
}
function App(){
  const [overview,setOverview]=useState<Overview|null>(null)
  const [health,setHealth]=useState<{model_online:boolean;model:string}|null>(null)
  const [builderOpen,setBuilderOpen]=useState(false)
  const [appliedFilters,setAppliedFilters]=useState<InsightFilters|null>(null)
  const [view,setView]=useState<'chat'|'students'|'skills'>('chat')
  const [studentId,setStudentId]=useState('')
  const [detail,setDetail]=useState<Detail|null>(null)
  const [query,setQuery]=useState('')
  const [search,setSearch]=useState('')
  const [messages,setMessages]=useState<ChatMessage[]>([])
  const [busy,setBusy]=useState(false)
  const [progress,setProgress]=useState('Finding relevant learning records…')
  const [error,setError]=useState('')
  const [loadError,setLoadError]=useState('')
  const [source,setSource]=useState<Source|null>(null)
  const [about,setAbout]=useState(false)
  const end=useRef<HTMLDivElement>(null)
  const input=useRef<HTMLTextAreaElement>(null)
  const dialog=useRef<HTMLElement>(null)
  const active=overview?.students.find(s=>s.id===studentId)
  useEffect(()=>{api<Overview>('/api/overview').then(setOverview).catch(e=>setLoadError(e.message)); const check=()=>api<{model_online:boolean;model:string}>('/api/health').then(setHealth).catch(()=>setHealth({model_online:false,model:'Local model'})); check(); const timer=setInterval(check,30000);return()=>clearInterval(timer)},[])
  useEffect(()=>{ if(!studentId){setDetail(null);return} let current=true; setDetail(null); api<Detail>('/api/students/'+studentId).then(d=>{if(current)setDetail(d)}).catch(e=>{if(current)setError(e.message)}); return()=>{current=false} },[studentId])
  useEffect(()=>{end.current?.scrollIntoView({behavior:'smooth',block:'end'})},[messages,busy])
  useEffect(()=>{
    if(!source&&!about)return
    const previous=document.activeElement as HTMLElement|null
    function keydown(event:KeyboardEvent){
      if(event.key==='Escape'){setSource(null);setAbout(false)}
      if(event.key==='Tab'){
        const elements=dialog.current?.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex="0"]')
        if(!elements?.length)return
        const first=elements[0],last=elements[elements.length-1]
        if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus()}
        else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus()}
      }
    }
    document.addEventListener('keydown',keydown)
    return()=>{document.removeEventListener('keydown',keydown);previous?.focus()}
  },[source,about])
  function newChat(){if(busy)return;setAppliedFilters(null);setBuilderOpen(false);setMessages([]);setQuery('');setError('');setView('chat');input.current?.focus()}
  function changeStudent(id:string){if(busy)return;setAppliedFilters(null);setStudentId(id);setMessages([]);setError('')}
  async function send(question=query,chosenFilters?:InsightFilters,retry=false){
    if(!question.trim()||busy)return
    const filters=chosenFilters??appliedFilters
    setBuilderOpen(false)
    if(chosenFilters)setAppliedFilters(chosenFilters)
    const retryIndex=messages.map(m=>m.role).lastIndexOf('user')
    const base=chosenFilters?[]:retry&&retryIndex>=0?messages.slice(0,retryIndex):messages
    const previous=base.filter(m=>!m.failed&&m.content.trim()&&(m.role==='user'||m.complete||Boolean(m.model))).map(({role,content,retrieval})=>({role,content,scope:role==='assistant'&&retrieval?{student_ids:retrieval.student_ids||[],skill_ids:retrieval.skill_ids||[]}:undefined}))
    const assistantIndex=base.length+1
    setMessages([...base,{role:'user',content:question.trim()},{role:'assistant',content:''}]);setQuery('');setError('');setBusy(true);setView('chat')
    setProgress('Finding relevant learning records…')
    try {
      await streamChat({question:question.trim(),student_id:studentId||null,history:previous,filters},({event,data})=>{
        if(event==='retrieval'){
          setProgress('Relevant records found. Preparing your answer…')
          setMessages(m=>m.map((message,i)=>i===assistantIndex?{...message,sources:data.sources as Source[],retrieval:data.retrieval as Retrieval}:message))
        }else if(event==='status'){
          setProgress(String(data.message))
        }else if(event==='delta'){
          setProgress('Checking the completed answer…')
        }else if(event==='done'){
          setMessages(m=>m.map((message,i)=>i===assistantIndex?{role:'assistant',complete:true,content:String(data.answer),sources:data.sources as Source[],model:String(data.model),truncated:Boolean(data.truncated),retrieval:data.retrieval as Retrieval,metrics:data.metrics as Metrics}:message))
        }
      })
    }catch(e){setError(e instanceof Error?e.message:'Unable to generate an answer.');setMessages(m=>m.map((message,i)=>i===assistantIndex||i===assistantIndex-1?{...message,failed:true,...(i===assistantIndex?{content:''}:{})}:message))}finally{setBusy(false)}
  }
  const summary=detail?.summary||overview?.summary
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="Forma home"><span className="brand-mark"><i/><i/><i/></span>forma<span className="brand-dot">.</span></a>
      <div className="workspace"><span className="workspace-icon"><GraduationCap size={18}/></span><div>Learning workspace<small>Pre-algebra & early algebra</small></div></div>
      <button className="new-chat" onClick={newChat} disabled={busy}><Plus size={17}/> New conversation <span>↗</span></button>
      <p className="nav-label">WORKSPACE</p>
      <nav aria-label="Main navigation">
        <button className={view==='chat'?'selected':''} onClick={()=>setView('chat')}><MessageSquare size={17}/> Learning assistant</button>
        <button className={view==='students'?'selected':''} onClick={()=>setView('students')}><Users size={17}/> Students <span className="nav-count">{overview?.students.length||'—'}</span></button>
        <button className={view==='skills'?'selected':''} onClick={()=>setView('skills')}><BookOpen size={17}/> Skill library <span className="nav-count">12</span></button>
      </nav>
      <div className="sidebar-note"><div className="note-icon"><ShieldCheck size={18}/></div><strong>A little more understanding.</strong><p>Turn learning records into thoughtful next steps.</p><span>Private by design. Runs locally.</span></div>
      <div className="sidebar-bottom"><div className="avatar teacher">T</div><div>Teacher workspace<small>Synthetic classroom</small></div><button title="About this workspace" aria-label="About this workspace" onClick={()=>setAbout(true)}><CircleHelp size={17}/></button></div>
    </aside>
    <div className="main-shell">
      <header><div className="breadcrumb">Workspace <ChevronRight size={14}/> <strong>{view==='chat'?'Learning assistant':view==='students'?'Students':'Skill library'}</strong></div><div className="header-right"><span className="synthetic-badge">Demo data</span><span className={'status '+(health?.model_online?'online':'')}><i/>{health?.model_online?'Local model connected':health?'Model offline':'Connecting'}</span></div></header>
      {loadError?<div role="alert" className="error page-error">Could not load the classroom. {loadError} <button onClick={()=>location.reload()}>Reload</button></div>:!overview?<div className="loading-page"><LoaderCircle className="spin"/> Loading your classroom…</div>:<>
      {view==='chat'?<main className="chat-page">
        <div className="chat-toolbar"><div><span className="tiny-dot"/> YOUR TEACHING COMPANION</div><div className="toolbar-actions"><button className="builder-toggle" type="button" disabled={busy} aria-expanded={builderOpen} onClick={()=>setBuilderOpen(!builderOpen)}><SlidersHorizontal size={14}/>{builderOpen?'Back to chat':'Build an insight'}</button>{!builderOpen&&<label className="scope-select"><Users size={14}/><select aria-label="Conversation student scope" value={studentId} disabled={busy} onChange={e=>changeStudent(e.target.value)}><option value="">All students</option>{overview.students.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select><ChevronDown size={13}/></label>}</div></div>
        <div className={'conversation '+(messages.length?'has-messages':'')}>
          {builderOpen?<InsightBuilder students={overview.students} skills={overview.skills} studentId={studentId} onStudentChange={changeStudent} initialFilters={appliedFilters} busy={busy} onGenerate={(question,filters)=>send(question,filters)}/>:!messages.length?<div className="welcome"><div className="welcome-emblem"><Layers3 size={27} strokeWidth={1.4}/></div><div className="eyebrow">EVERY LEARNER HAS A STORY</div><h1>A clearer picture<br/>of learning<span>.</span></h1><p className="welcome-description">{active?<>Explore {active.name.split(' ')[0]}’s progress, understand the challenges,<br className="desktop-break"/> and find a thoughtful next step.</>:<>Ask about progress, uncover patterns, and find the next<br className="desktop-break"/> step for your students. All grounded in their learning.</>}</p>
            <div className="class-context"><span><Users size={14}/>{active?active.name:overview.students.length+' students'}</span><i/><span>{summary?.problems} problem sessions</span><i/><span>12 skills</span></div>
            <div className="suggestions">{prompts.map(({icon:Icon,label,text,question},i)=><button key={label} onClick={()=>send(active?['Summarize this student’s strengths and areas needing support.','How is this student doing with combining like terms?','How has this student’s correctness and affect changed over the recorded period?'][i]:question)} disabled={busy}><div><Icon size={17}/><ArrowUpRight size={15}/></div><small>{label}</small><p>{active?['What should we work on next?','How are like terms going?','What has changed over time?'][i]:text}</p></button>)}</div>
          </div>:<div className="messages" aria-live="polite">{messages.filter(m=>m.content.length>0).map((m,i)=><article className={'message '+m.role} key={i}>{m.role==='assistant'?<div className="assistant-icon"><Layers3 size={17}/></div>:<div className="avatar teacher">T</div>}<div className="message-content"><div className="message-label">{m.role==='assistant'?'Forma':'You'}{m.role==='assistant'&&<span>{m.metrics?(m.metrics.cached?'From recent results':`Answered in ${(m.metrics.total_ms/1000).toFixed(1)}s`):'Grounded in learning records'}</span>}</div><div className="markdown"><Markdown disallowedElements={['img']}>{m.content}</Markdown></div>{m.sources&&<details className="sources"><summary><Database size={13}/>{m.sources.length} retrieved sources <ChevronDown size={13}/></summary><div className="source-list">{m.retrieval&&<p className="retrieval-query"><strong>Search question</strong> {m.retrieval.query}<small>Relevant records found in {m.retrieval.retrieval_ms.toFixed(0)} ms</small></p>}{m.sources.map(s=><button key={s.id} onClick={()=>setSource(s)}><span>{s.id}</span>{s.title}{s.cited&&<Check size={12}/>}<ArrowUpRight size={13}/></button>)}</div></details>}{m.truncated&&<p className="muted">Response reached its length limit. Ask a focused follow-up for more detail.</p>}</div></article>)}{busy&&!messages.at(-1)?.content?<article className="message assistant"><div className="assistant-icon"><Layers3 size={17}/></div><div className="thinking"><span/><span/><span/><p>{progress}</p></div></article>:null}<div ref={end}/></div>}
        </div>
        <div className="composer-area" hidden={builderOpen}>{appliedFilters&&<div className="active-insight"><SlidersHorizontal size={13}/><span>{overview.skills.find(s=>s.id===appliedFilters.skill_id)?.name||'All skills'} · {appliedFilters.accuracy_min}–{appliedFilters.accuracy_max}% · {appliedFilters.intent}</span><button type="button" disabled={busy} onClick={()=>{setAppliedFilters(null);setMessages([]);setError('')}}>Clear filters</button></div>}{error&&<div className="error" role="alert">{error}<button disabled={busy} onClick={()=>{const last=messages.filter(m=>m.role==='user').at(-1);if(last)send(last.content,undefined,true)}}>Retry</button></div>}<form className="composer" onSubmit={(e:FormEvent)=>{e.preventDefault();send()}}><textarea ref={input} aria-label="Ask about your students" placeholder={active?`Ask about ${active.name.split(' ')[0]}’s learning…`:'What would you like to understand about your students?'} value={query} onChange={e=>setQuery(e.target.value)} rows={2} maxLength={4000} disabled={busy} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();send()}}}/><div className="composer-bottom"><span><Database size={13}/> {active?active.name:'Classroom records'}<span className="composer-divider">/</span><span className="model-label">{health?.model||'Local model'}</span></span><button type="submit" disabled={busy||!query.trim()} aria-label="Send message">{busy?<LoaderCircle className="spin" size={17}/>:<ArrowUp size={19}/>}</button></div></form><p className="composer-note"><ShieldCheck size={12}/> Stays on your device <span>·</span> AI can make mistakes. Explore the sources.</p></div>
      </main>:view==='students'?<main className="data-page"><div className="page-heading"><div className="eyebrow">YOUR CLASSROOM</div><h1>{detail?detail.name:'Small details. Real insight.'}</h1><p>{detail?`${detail.id} · Grade ${detail.grade} · Fictional student`:`${overview.students.length} fictional learners. A different learning journey for each.`}</p></div>{detail?<><button className="back-button" onClick={()=>changeStudent('')}><ChevronLeft size={15}/> All students</button><div className="stats-row">{[['Problems',detail.summary.problems],['Correctness',detail.summary.accuracy+'%'],['Attempts',detail.summary.attempts],['Hints taken',detail.summary.hints]].map(([label,value])=><div key={label}><small>{label}</small><strong>{value}</strong></div>)}</div><div className="time-summary" aria-label="Time on problems"><div><small>Total time on problems</small><strong>{duration(detail.summary.total_time_seconds)}</strong></div><div><small>Average time per problem</small><strong>{detail.summary.avg_time_seconds.toFixed(1)} s</strong></div><p>Synthetic elapsed time, including all attempts and hints.</p></div><div className="affect-panel"><h2>The learning experience <span>Average synthetic affect · 0–1</span></h2><div>{affects.map(a=><div className={'affect '+a} key={a}><label>{a}<b>{detail.summary[a].toFixed(2)}</b></label><div className="bar"><i style={{width:detail.summary[a]*100+'%'}}/></div></div>)}</div></div><div className="section-heading"><h2>Problem history <span>{detail.history.length} sessions</span></h2><button className="text-button" onClick={()=>{setView('chat');send('Summarize this student’s learning and suggest a next step.')}} disabled={busy}>Ask Forma <ArrowUpRight size={15}/></button></div><div className="table-scroll"><table><thead><tr><th>Problem / skills</th><th>Outcome</th><th>Attempts</th><th>Hints</th><th>Time taken</th><th>Affect · C / D / C / F</th></tr></thead><tbody>{detail.history.map(r=><tr key={r.id}><td><strong>{r.prompt}</strong><small>{r.skills.join(' · ')}</small><small>{r.id} · {r.occurred_at.slice(0,10)}</small></td><td><span className={'outcome '+(r.correct?'correct':'')}>{r.correct?'Correct':'Incorrect'}</span></td><td>{r.attempts}</td><td>{r.hints}</td><td className="session-duration" title={`${r.time_taken_seconds} seconds`}>{duration(r.time_taken_seconds)}</td><td><div className="affect-values" title="Confusion / Determination / Confidence / Frustration">{affects.map(a=><span key={a} title={a}>{r[a].toFixed(2)}</span>)}</div></td></tr>)}</tbody></table></div><p className="muted">Affect columns: confusion, determination, confidence, frustration. Each score is independently recorded from 0 to 1.</p></>:<><div className="list-toolbar"><span>{overview.students.length} students <span className="muted">· {overview.date_range}</span></span><label className="search"><Search size={15}/><input aria-label="Search students" placeholder="Find a student…" value={search} onChange={e=>setSearch(e.target.value)}/></label></div><div className="student-grid">{overview.students.filter(s=>(s.name+' '+s.id).toLowerCase().includes(search.toLowerCase())).map((s,i)=><button className="student-card" key={s.id} disabled={busy} onClick={()=>changeStudent(s.id)}><div><span className={'avatar color-'+i%4}>{initials(s.name)}</span><ArrowUpRight size={15}/></div><h3>{s.name}</h3><small>{s.id} · Grade {s.grade}</small><div className="student-stats"><span><b>{s.accuracy}%</b> correctness</span><span><b>{s.problems}</b> problems</span></div></button>)}</div>{!overview.students.some(s=>(s.name+' '+s.id).toLowerCase().includes(search.toLowerCase()))&&<p className="empty-state">No students match “{search}”.</p>}</>}</main>:<main className="data-page"><div className="page-heading"><div className="eyebrow">BUILDING BLOCKS</div><h1>A foundation for what’s next.</h1><p>12 pre-algebra and early algebra skills, grounded in classroom practice.</p></div><div className="skill-grid">{overview.skills.map((s,i)=><button className="skill-card" disabled={busy} key={s.id} onClick={()=>{changeStudent('');setBuilderOpen(false);setView('chat');setQuery(`How is the class doing with ${s.name.toLowerCase()}?`)}}><span className="skill-number">{String(i+1).padStart(2,'0')}</span><ArrowUpRight size={16}/><h3>{s.name}</h3><p>{s.problems} linked problem sessions</p><div className="skill-progress"><div className="bar"><i style={{width:s.accuracy+'%'}}/></div><span>{s.accuracy}%</span></div><small>Class correctness</small></button>)}</div><p className="muted">A problem can involve more than one skill, so skill totals may overlap.</p></main>}
      </>}
      <footer><span><span className="tiny-dot"/> MADE FOR THE MOMENTS BETWEEN LESSONS</span><button onClick={()=>setAbout(true)}>About this workspace <ArrowUpRight size={12}/></button></footer>
    </div>
    {(source||about)&&<div className="modal-backdrop" onClick={()=>{setSource(null);setAbout(false)}}><section ref={dialog} role="dialog" aria-modal="true" aria-label={source?'Retrieved evidence':'About Forma'} className="modal" onClick={e=>e.stopPropagation()} onKeyDown={e=>{if(e.key==='Escape'){setSource(null);setAbout(false)}}}><button autoFocus className="modal-close" aria-label="Close dialog" onClick={()=>{setSource(null);setAbout(false)}}><X size={20}/></button>{source?<><div className="eyebrow">RETRIEVED EVIDENCE · {source.id}</div><h2>{source.title}</h2><p className="muted">{source.cited?'Cited in this answer.':'Provided to the model as context.'} Values below come directly from the database.</p><pre>{JSON.stringify(source.data,null,2)}</pre></>:<><div className="welcome-emblem"><Layers3 size={25}/></div><h2>A clearer picture of learning.</h2><p>Forma is a local learning assistant built around {overview?.students.length ?? 0} fictional students, {overview?.problem_count ?? 0} problems, {overview?.skills.length ?? 0} skills, and {overview?.summary.problems.toLocaleString() ?? 0} problem sessions.</p><p>Answers use retrieved Chroma records and a model running on this device. Correctness means the final outcome of a problem session. Time taken is synthetic elapsed time across that session’s attempts and hints. Affect scores are synthetic observations on a 0–1 scale, not psychological assessments.</p><p>Dataset: {overview?.date_range ?? "Loading…"}. Conversations are held in memory and cleared on reload or when switching students.</p><div className="about-tech"><ShieldCheck size={17}/> Local inference · No cloud API</div></>}</section></div>}
  </div>
}
export default App
