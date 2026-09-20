import { spawn } from 'node:child_process'
import { setTimeout as delay } from 'node:timers/promises'
const children=[]
let closing=false
function launch(cmd,args){const child=spawn(cmd,args,{stdio:'inherit',detached:process.platform!=='win32'});children.push(child);child.on('error',e=>{console.error(e.message);stop(1)});child.on('exit',code=>{if(!closing)stop(code||0)});return child}
function stop(code=0){if(closing)return;closing=true;children.forEach(p=>{try{if(process.platform==='win32')p.kill('SIGTERM');else process.kill(-p.pid,'SIGTERM')}catch{}});setTimeout(()=>process.exit(code),300)}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop())
async function healthy(){try {return (await fetch('http://127.0.0.1:8091/health')).ok}catch{return false}}
if(!await healthy()){
  launch('python3',['scripts/start_model.py'])
  let ready=false
  for(let i=0;i<120&&!closing;i++){if(await healthy()){ready=true;break}await delay(1000)}
  if(!ready){console.error('Local model did not become ready.');stop(1)}
}
if(!closing){launch('.venv/bin/uvicorn',['backend.main:app','--host','127.0.0.1','--port','8008']);launch('npm',['--prefix','frontend','run','dev'])}
