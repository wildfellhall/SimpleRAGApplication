export type StreamEvent = {event:string; data:Record<string,unknown>}

/** Parse SSE across arbitrary UTF-8/network chunk boundaries. */
export async function streamChat(body:unknown, onEvent:(event:StreamEvent)=>void) {
  const response=await fetch('/api/chat/stream',{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),
  })
  if(!response.ok){
    const error=await response.json().catch(()=>null)
    throw new Error(typeof error?.detail==='string'?error.detail:'Unable to start a response. Please try again.')
  }
  if(!response.body)throw new Error('Your browser could not open the response stream.')
  const reader=response.body.getReader()
  const decoder=new TextDecoder()
  let pending='', completed=false
  const dispatch=(frame:string)=>{
    let event='message'
    const data:string[]=[]
    for(const line of frame.split('\n')){
      if(line.startsWith('event:'))event=line.slice(6).trim()
      if(line.startsWith('data:'))data.push(line.slice(5).trimStart())
    }
    if(!data.length)return
    const parsed=JSON.parse(data.join('\n'))
    if(event==='error')throw new Error(parsed.detail||'The response was interrupted. Please try again.')
    if(event==='done'){
      if(typeof parsed.answer!=='string'||!parsed.answer.trim())throw new Error('The model did not return a complete answer. Please retry.')
      if(/<\/?(?:think|analysis|reasoning)\b|<\|channel\|>/i.test(parsed.answer))throw new Error('An invalid model response was blocked. Please retry.')
      completed=true
    }
    onEvent({event,data:parsed})
  }
  try{
    while(true){
      const {value,done}=await reader.read()
      pending+=decoder.decode(value,{stream:!done}).replace(/\r\n/g,'\n')
      let separator:number
      while((separator=pending.indexOf('\n\n'))!==-1){
        dispatch(pending.slice(0,separator));pending=pending.slice(separator+2)
      }
      if(done){if(pending.trim())dispatch(pending);break}
    }
    if(!completed)throw new Error('The response was interrupted. Please try again.')
  }finally{
    await reader.cancel().catch(()=>{})
    reader.releaseLock()
  }
}
