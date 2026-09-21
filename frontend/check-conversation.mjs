import { chromium, expect } from '@playwright/test'
const browser=await chromium.launch({channel:'chrome',headless:true})
try{
  const context=await browser.newContext({viewport:{width:1440,height:1000}})
  const page=await context.newPage()
  const errors=[],requests=[]
  page.on('pageerror',error=>errors.push(error.message))
  let mode='normal'
  await context.route('**/api/chat/stream',route=>{
    const body=route.request().postDataJSON();requests.push(body)
    const answer=mode==='empty'?'':mode==='thinking'?'<think>PRIVATE THOUGHT</think>':`Activity ${requests.length}: use paper strips for fraction practice, keeping the original ten-minute limit and no-calculator constraint. Model one example, ask the student to explain a second example, and end with a short independent check.`
    const result={answer,sources:[],model:'Test local model',retrieval:{query:body.question,student_ids:['STU-1020'],skill_ids:['S06'],retrieval_ms:1,matches:[]},metrics:{cached:false,total_ms:10,first_token_ms:5}}
    const frames=[['status',{message:'Preparing a complete answer…'}],['delta',{text:answer}],['done',result]]
    return route.fulfill({contentType:'text/event-stream',body:frames.map(([event,data])=>`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('')})
  })
  await page.goto('http://127.0.0.1:5178')
  const first='Help Julian with fractions. Ten minutes, no calculators. '+('Please use paper strips and keep practice concrete. '.repeat(15))
  for(const question of [first,'Make an activity.','Give two problems.','Adapt the second problem.','What did I originally ask for?','Revise the activity again.']){
    await page.getByRole('textbox',{name:'Ask about your students'}).fill(question)
    await page.getByRole('button',{name:'Send message'}).click()
    await expect(page.getByRole('button',{name:'New conversation'})).toBeEnabled()
  }
  expect(requests.at(-1).history).toHaveLength(10)
  expect(requests.at(-1).history[0].content).toBe(first.trim())
  expect(requests.at(-1).history[1].scope).toEqual({student_ids:['STU-1020'],skill_ids:['S06']})
  await expect(page.locator('.message.assistant .markdown')).toHaveCount(6)
  for(const failureMode of ['empty','thinking']){
    mode=failureMode
    await page.getByRole('textbox',{name:'Ask about your students'}).fill('Another useful activity.')
    await page.getByRole('button',{name:'Send message'}).click()
    await expect(page.getByRole('alert')).toBeVisible()
    await expect(page.locator('.markdown').filter({hasText:'PRIVATE THOUGHT'})).toHaveCount(0)
    expect(requests.at(-1).history.every(m=>m.content.trim())).toBe(true)
    const failedQuestionCount=await page.locator('.message.user').count()
    mode='normal'
    await page.getByRole('button',{name:'Retry',exact:true}).click()
    await expect(page.getByRole('button',{name:'New conversation'})).toBeEnabled()
    await expect(page.getByRole('alert')).toHaveCount(0)
    expect(await page.locator('.message.user').count()).toBe(failedQuestionCount)
    expect(requests.at(-1).history.every(m=>m.content!=='Another useful activity.'||m.role==='assistant'||failureMode==='thinking')).toBe(true)
  }
  await page.getByRole('button',{name:'New conversation'}).click()
  await expect(page.locator('.message')).toHaveCount(0)
  await page.getByRole('textbox',{name:'Ask about your students'}).fill('Start fresh with Maya.')
  await page.getByRole('button',{name:'Send message'}).click()
  await expect(page.getByRole('button',{name:'New conversation'})).toBeEnabled()
  expect(requests.at(-1).history).toEqual([])
  expect(errors).toEqual([])
  console.log('Full-history transport, retrieval scope, hidden invalid output, retry without duplicate turns, and conversation reset passed.')
}finally{await browser.close()}
