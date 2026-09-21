"""Bounded conversation context and validation of user-visible model answers."""
import re

TEMPLATE_OPTIONS = {'enable_thinking': False, 'preserve_thinking': False}
MAX_OUTPUT_TOKENS = 1200


def clean_answer(raw):
    """Run before *any* model text is sent to the browser or saved as history."""
    text = raw or ''
    # A closing marker without an opener means the parser emitted a reasoning tail.
    for tag in ('think', 'analysis', 'reasoning'):
        text = re.sub(rf'<{tag}\b[^>]*>.*?</{tag}\s*>', '', text, flags=re.S|re.I)
        text = re.sub(rf'<{tag}\b[^>]*>.*$', '', text, flags=re.S|re.I)
        if re.search(rf'</{tag}\s*>', text, re.I):
            text = re.split(rf'</{tag}\s*>', text, flags=re.I)[-1]
    # Some model templates emit explicit channel delimiters instead of XML tags.
    if '<|channel|>final' in text:
        text = text.split('<|channel|>final')[-1].removeprefix('<|message|>')
    elif '<|channel|>analysis' in text:
        return ''
    text = re.sub(r'<\|(?:im_start|im_end|endoftext|message|channel|end|return)\|>', '', text).strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:markdown|md)?\s*\n?', '', text)[:-3].strip()
    return text


def answer_issue(text, question, has_history, finish):
    if not text.strip():return 'empty answer'
    if finish != 'stop':return 'unfinished answer'
    # These patterns describe composing an answer, rather than explaining mathematics.
    if re.search(r'(?im)^\s*(?:#{1,4}\s*)?(?:analysis|thinking|reasoning|internal (?:analysis|reasoning))\s*(?::|$)',text):
        return 'internal reasoning'
    if re.search(r'(?i)\b(?:we (?:need|should|must) (?:to )?(?:answer|respond|provide|mention|cite)|the user (?:asks|wants|is asking|requested)|I (?:need|should|must) (?:to )?(?:respond|craft|compose)|let me (?:think|reason)|I (?:will|shall) now (?:answer|respond))\b',text):
        return 'internal reasoning'
    if sum(phrase in text.lower() for phrase in ['use only supplied fictional records','cite each factual paragraph','never invent facts or source ids','current evidence overrides earlier answers','answer directly in markdown'])>=2:
        return 'instruction echo instead of a teacher-facing answer'
    if has_history and re.search(r"(?i)(?:I (?:do not|don't|cannot|can't) (?:have|access|remember|see)|no (?:access to |prior )?)(?:[^.!?\n]{0,70})(?:conversation|previous messages|earlier messages|chat history|previous context)",text):
        return 'incorrect claim that conversation history is unavailable'
    # Keep naturally brief greetings and direct factual replies; recover terse non-answers.
    brief = bool(re.search(r'(?i)^(?:what (?:is|was|are)|how many|how much|what percent)\b',question.strip()) or re.search(r'(?i)\b(?:brief|short|one sentence|just the|only the|exact (?:number|value|correctness|accuracy|average|total|time))\b',question))
    minimal = bool(re.search(r'(?i)\b(?:one word|yes or no|number only)\b',question))
    greeting = bool(re.fullmatch(r'(?i)\s*(?:hi|hello|hey|thanks|thank you|ok|okay)[!. ]*',question))
    if greeting or minimal:return None
    if re.fullmatch(r'(?i)\W*(?:yes|no|sure|okay|ok|certainly|done|here you go)[.! ]*',text):return 'unhelpfully short answer'
    words = re.findall(r'\b\w+\b',text)
    if len(words)<(4 if brief else 24):return 'unhelpfully short answer'
    return None


def clip(text, limit):
    if len(text)<=limit:return text
    half=(limit-30)//2
    return text[:half]+'\n[excerpt shortened]\n'+text[-half:]


def history_context(history, question, budget=14000):
    """Keep complete recent turns plus explicitly labelled excerpts of earlier turns.

    Excerpts are deterministic, never invented model summaries. The original history
    stays with the client and remains available to retrieval on subsequent turns.
    """
    cleaned=[]
    for m in history or []:
        text=clean_answer(m['content']) if m['role']=='assistant' else m['content'].strip()
        if m['role']=='assistant' and answer_issue(text,'brief',False,'stop') in ('internal reasoning','instruction echo instead of a teacher-facing answer'):continue
        if text:cleaned.append({'role':m['role'],'content':text})
    recent=[]
    used=0
    recent_budget=int(budget*.72)
    for m in reversed(cleaned):
        size=len(m['content'])+30
        if used+size>recent_budget and recent:break
        recent.append({**m,'content':clip(m['content'],recent_budget)})
        used+=min(size,recent_budget)
    recent.reverse()
    older=cleaned[:len(cleaned)-len(recent)]
    if not older:return recent, '', False
    terms=set(re.findall(r'\w{3,}',question.lower()))
    def rank(item):
        i,m=item
        overlap=len(terms & set(re.findall(r'\w{3,}',m['content'].lower())))
        # Preserve the opening teacher instructions and prioritize relevant teacher turns.
        return (4 if i==0 else 0)+(2 if m['role']=='user' else 0)+overlap+i/max(1,len(older))
    remaining=budget-used
    excerpts=[]
    for i,m in sorted(enumerate(older),key=rank,reverse=True):
        excerpt=f"Earlier {m['role']} message {i+1}: {clip(m['content'],min(600, max(180,remaining-60)))}"
        if len(excerpt)>remaining:continue
        excerpts.append((i,excerpt));remaining-=len(excerpt)+2
        if remaining<200:break
    notes='\n\n'.join(text for _,text in sorted(excerpts))
    return recent, notes, True
