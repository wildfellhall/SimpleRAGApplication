import { useEffect, useState } from 'react'
import { ArrowUpRight, BookOpen, LayoutList, LoaderCircle, RotateCcw, Sparkles, Users } from 'lucide-react'

export type InsightFilters = {skill_id:string|null; accuracy_min:number; accuracy_max:number; intent:'recommendations'|'summary'|'overview'}
type Preview = {question:string; filters:InsightFilters; students:{id:string;name:string;accuracy:number}[]; matching_count:number; problem_sessions:number; accuracy_basis:string}
type Props = {
  students:{id:string;name:string}[]; skills:{id:string;name:string}[];
  studentId:string; onStudentChange:(id:string)=>void;
  initialFilters:InsightFilters|null; busy:boolean;
  onGenerate:(question:string,filters:InsightFilters)=>void;
}
const modes=[
  {value:'recommendations' as const,label:'Recommendations',description:'Find the next teaching step.',icon:Sparkles},
  {value:'summary' as const,label:'Summary',description:'Understand the learning story.',icon:BookOpen},
  {value:'overview' as const,label:'Overview',description:'See the group at a glance.',icon:LayoutList},
]

export default function InsightBuilder({students,skills,studentId,onStudentChange,initialFilters,busy,onGenerate}:Props){
  const [skill,setSkill]=useState(initialFilters?.skill_id||'')
  const [minimum,setMinimum]=useState(initialFilters?.accuracy_min??0)
  const [maximum,setMaximum]=useState(initialFilters?.accuracy_max??100)
  const [intent,setIntent]=useState<InsightFilters['intent']>(initialFilters?.intent||'recommendations')
  const [focus,setFocus]=useState('')
  const [result,setResult]=useState<{key:string;data:Preview}|null>(null)
  const [failure,setFailure]=useState<{key:string;message:string}|null>(null)
  const [retry,setRetry]=useState(0)
  const key=JSON.stringify({student_id:studentId||null,skill_id:skill||null,accuracy_min:minimum,accuracy_max:maximum,intent,focus})
  const preview=result?.key===key?result.data:null
  const error=failure?.key===key?failure.message:''
  useEffect(()=>{
    const controller=new AbortController()
    setFailure(null)
    const timer=setTimeout(async()=>{
      try{
        const response=await fetch('/api/insights/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:key,signal:controller.signal})
        const data=await response.json()
        if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Could not preview these filters. Please try again.')
        if(!controller.signal.aborted)setResult({key,data})
      }catch(e){if(!controller.signal.aborted)setFailure({key,message:e instanceof Error?e.message:'Could not load the preview.'})}
    },180)
    return()=>{clearTimeout(timer);controller.abort()}
  },[key,retry])
  function reset(){onStudentChange('');setSkill('');setMinimum(0);setMaximum(100);setIntent('recommendations');setFocus('')}
  return <form className="insight-builder" onSubmit={e=>{e.preventDefault();if(preview&&preview.matching_count&&!busy)onGenerate(preview.question,preview.filters)}}>
    <div className="insight-heading"><div><div className="eyebrow">A MORE FOCUSED CONVERSATION</div><h1>Build an insight<span>.</span></h1><p>Choose who to focus on and what you’d like to understand.</p></div><button type="button" className="builder-reset" onClick={reset} disabled={busy}><RotateCcw size={13}/> Reset</button></div>
    <div className="insight-card">
      <div className="insight-selects">
        <label>Student<select aria-label="Insight student" value={studentId} disabled={busy} onChange={e=>onStudentChange(e.target.value)}><option value="">All students</option>{students.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select></label>
        <label>Skill<select aria-label="Insight skill" value={skill} disabled={busy} onChange={e=>setSkill(e.target.value)}><option value="">All skills</option>{skills.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select></label>
      </div>
      <fieldset className="accuracy-controls" disabled={busy}><legend>Accuracy range <span>{minimum}–{maximum}%</span></legend>
        <div className="range-inputs">
          <label htmlFor="accuracy-min">Minimum <output htmlFor="accuracy-min">{minimum}%</output><input id="accuracy-min" aria-label="Minimum accuracy" type="range" min={0} max={100} step={1} value={minimum} onChange={e=>setMinimum(Math.min(Number(e.target.value),maximum))}/></label>
          <label htmlFor="accuracy-max">Maximum <output htmlFor="accuracy-max">{maximum}%</output><input id="accuracy-max" aria-label="Maximum accuracy" type="range" min={0} max={100} step={1} value={maximum} onChange={e=>setMaximum(Math.max(Number(e.target.value),minimum))}/></label>
        </div>
        <div className="range-presets">{[[0,100,'Any accuracy'],[0,60,'0–60%'],[60,80,'60–80%'],[80,100,'80–100%']].map(([low,high,label])=><button type="button" key={label} aria-pressed={minimum===low&&maximum===high} onClick={()=>{setMinimum(Number(low));setMaximum(Number(high))}}>{label}</button>)}</div>
        <p className="control-help">Final correctness {skill?'within the selected skill':'across all skills'}. Both ends of the range are included.</p>
      </fieldset>
      <fieldset className="intent-controls" disabled={busy}><legend>What would you like?</legend><div className="intent-options">{modes.map(({value,label,description,icon:Icon})=><label className={intent===value?'chosen':''} key={value}><span><Icon size={16}/><input type="radio" name="insight-intent" value={value} checked={intent===value} onChange={()=>setIntent(value)}/></span><strong>{label}</strong><small>{description}</small></label>)}</div></fieldset>
      <label className="insight-focus">Additional focus <span>optional</span><input aria-label="Additional focus" value={focus} onChange={e=>setFocus(e.target.value)} disabled={busy} maxLength={300} placeholder="For example, focus on hints and confidence"/></label>
    </div>
    <div className="insight-preview" aria-live="polite">
      {error?<div className="error" role="alert">{error}<button type="button" onClick={()=>setRetry(n=>n+1)}>Retry preview</button></div>:!preview?<p className="preview-loading"><LoaderCircle size={14} className="spin"/> Checking the learning records…</p>:<>
        <div className="preview-count"><Users size={15}/><strong>{preview.matching_count} {preview.matching_count===1?'student matches':'students match'}</strong><span>· {preview.problem_sessions.toLocaleString()} problem sessions</span></div>
        {preview.matching_count?<><div className="matching-students">{preview.students.slice(0,4).map(s=><span key={s.id}>{s.name}<b>{s.accuracy}%</b></span>)}{preview.students.length>4&&<span>+{preview.students.length-4} more</span>}</div><div className="question-preview"><small>YOUR QUESTION</small><p>{preview.question}</p></div></>:<p className="no-matches">No students match these settings. Widen the accuracy range or choose another student or skill.</p>}
      </>}
    </div>
    <div className="insight-actions"><span><BookOpen size={13}/> Grounded in matching learning records</span><button type="submit" disabled={busy||!preview?.matching_count}>Generate insight <ArrowUpRight size={16}/></button></div>
  </form>
}
