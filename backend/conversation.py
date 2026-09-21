"""Bounded conversation context and validation of user-visible model answers."""
import re
from difflib import SequenceMatcher

# History is sanitized below and contains no reasoning_content. For this Qwen
# template, preserve_thinking keeps the empty structural <think></think> prefix
# on historical assistant turns, matching the prefix used for the next answer.
TEMPLATE_OPTIONS = {'enable_thinking': False, 'preserve_thinking': True}
MAX_OUTPUT_TOKENS = 1200
# Qwen3.8's published non-thinking settings; very low temperatures can loop.
SAMPLING_OPTIONS = {'temperature': .7, 'top_p': .8, 'top_k': 20, 'min_p': 0.0,
                    'presence_penalty': 1.5, 'repeat_penalty': 1.0}


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


def answer_words(text):
    """Ignore presentation/citations, retaining numbers and math operators."""
    text=re.sub(r'\[(?:STU-\d+|R\d+|S\d+|CLASS|ROSTER|FILTERED)\]', '', clean_answer(text))
    return re.findall(r'\w+|[+−×÷=<>/]',text.casefold())


def repetition_issue(text, question, history):
    """Catch copied follow-up answers, not shared vocabulary or repeated facts.

    This is a lexical guard, not a semantic relevance judge. Check recent answers
    because a model can also loop back to the answer before its most recent one.
    """
    current=answer_words(text)
    # A repeated factual value or greeting can legitimately answer a new question.
    if len(current)<30:return None
    q=question.casefold().replace('’',"'")
    repeat_requested=bool(re.search(r'\b(?:repeat|restate|reproduce|quote|resend)\b|\b(?:say|show|give) (?:me )?(?:that|it|the same (?:answer|response)) again\b',q))
    modification=bool(re.search(r"\b(?:but|instead|change|revise|add|different|new|shorter|simpler|easier|harder|correct|replace|update|don't|not|never|avoid|without|stop)\b",q))
    if repeat_requested and not modification:return None
    last_question=''
    pairs=[]
    for message in history or []:
        if message['role']=='user':last_question=message['content']
        elif message['role']=='assistant':pairs.append((last_question,message['content']))
    for previous_question,answer in pairs[-3:]:
        if answer_words(question)==answer_words(previous_question):continue
        previous=answer_words(answer)
        if len(previous)<30:continue
        if current==previous:return 'repeated earlier answer instead of addressing the latest request'
        # A shorter summary or a concrete correction can reuse much of its source.
        if re.search(r'\b(?:summari[sz]e|shorten|shorter|condense|brief|concise)\b',q) and len(current)<len(previous)*.8:continue
        if re.search(r'\b(?:change|revise|correct|replace|update|make|adapt)\b',q):
            numbers=lambda words:[w for w in words if re.search(r'\d',w)]
            if numbers(current)!=numbers(previous):continue
            # Small requested edits (e.g. ten minutes -> five minutes, strips -> tiles)
            # can be valid even when the rest of the response is intentionally reused.
            edit_words=set(answer_words(question))-set('change revise correct replace update make adapt the a an it that this to into with for instead now please easier harder shorter simpler different new'.split())
            if (set(current)-set(previous)) & edit_words:continue
        # Allow substantive additions while catching minor edits and copied sections.
        similarity=SequenceMatcher(None,previous,current,autojunk=False).ratio()
        def shingles(words):return {tuple(words[i:i+5]) for i in range(len(words)-4)}
        old,new=shingles(previous),shingles(current)
        if similarity>=.92 or (len(old & new)/len(new)>=.9 and len(old & new)/len(old)>=.7):
            return 'repeated earlier answer instead of addressing the latest request'
    return None


def retry_instruction(issue):
    if issue=='repeated earlier answer instead of addressing the latest request':
        return ('The discarded draft repeated an earlier answer. Fulfill the final user message: '
                'produce the requested new material, revision, or explanation. Use earlier answers only '
                'to resolve references. Do not repeat the student overview or the previous plan unless '
                'the latest request explicitly asks for it. Start with the requested result.')
    return ('The previous generation did not produce a usable final answer. Answer the final user message '
            'directly using the supplied conversation and records. Provide a complete teacher-facing '
            'response, without internal reasoning. Keep it under 350 words.')


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
