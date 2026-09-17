"""Context-aware grading for new Phase 13 mathematics protocols.

Historical graders and frozen runs are unchanged. This backend preserves radix,
choice, tuple order and interval boundaries instead of treating every answer as
an untyped numerical string. Multi-answer matching requires full multiplicity.
"""
from __future__ import annotations

from functools import lru_cache
import re

from .math_grading import extract_boxed


def clean_math(value):
    value=str(value).strip().replace(r'\$','').replace('$','').replace('\u2212','-')
    value=value.replace(r'\left','').replace(r'\right','').replace(r'\dfrac',r'\frac')
    value=re.sub(r'\\[,!;:]','',value)
    value=re.sub(r'(?<!\w)([+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE]([+-]?\d+)(?!\w)',
                 lambda m:m.group(1)+r'\cdot10^{'+m.group(2)+'}',value)
    return value.strip().rstrip('.,;').strip()


def split_top_level(value, separators=',;'):
    """Split only outside TeX groups, tuples and intervals."""
    parts=[];start=0;depth=0
    for i,char in enumerate(value):
        escaped=i>0 and (len(value[:i])-len(value[:i].rstrip('\\'))) % 2 == 1
        # Escaped set braces still delimit a mathematical group.
        if char in '([{':depth+=1
        elif char in ')]}':depth-=1
        elif char in separators and depth==0 and not escaped:
            parts.append(value[start:i].strip());start=i+1
        if depth<0:raise ValueError('Unbalanced answer delimiters')
    if depth:raise ValueError('Unbalanced answer delimiters')
    parts.append(value[start:].strip())
    if any(not p for p in parts):raise ValueError('Empty answer component')
    return parts


def _unbrace_set(value):
    if value.startswith(r'\{') and value.endswith(r'\}'):
        return value[2:-2].strip()
    return value


def _wrapped(value):
    if not value or value[0] not in '([' or value[-1] not in ')]':return False
    depth=0
    for i,char in enumerate(value):
        if char in '([{':depth+=1
        elif char in ')]}':depth-=1
        if depth==0:return i==len(value)-1
    return False


def _parts(value, multiple):
    value=clean_math(value)
    if not multiple:return [value]
    parts=split_top_level(_unbrace_set(value))
    expanded=[]
    for part in parts:
        if r'\pm' in part:
            expanded.extend([part.replace(r'\pm','+'),part.replace(r'\pm','-')])
        else:expanded.append(part)
    return expanded


def compile_answer_spec(row, config):
    gold=row['answer']
    if gold is None:raise ValueError('Missing reference answer')
    if isinstance(gold,list):gold=','.join(gold)
    gold=clean_math(gold)
    if not gold:raise ValueError('Empty reference answer')
    dataset=row['dataset'];question=row.get('question','')
    kind=row.get('answer_kind','symbolic');multiple=False;base=None
    tolerance=0.;relative_tolerance=0.
    if dataset=='aqua_rat':kind='choice'
    elif dataset=='olympiadbench':
        kind={'Numerical':'symbolic','Expression':'symbolic','Tuple':'tuple','Interval':'interval'}[row['answer_type']]
        multiple=bool(row['is_multiple_answer'])
        raw_tolerance=row.get('error_tolerance')
        tolerance=float(raw_tolerance) if raw_tolerance is not None else config['olympiad_absolute_tolerance']
    elif dataset=='gsm8k_hard':
        tolerance=config['gsmhard_absolute_tolerance'];relative_tolerance=config['gsmhard_relative_tolerance']
    if kind=='symbolic' and re.fullmatch(r'(?:\\text\{)?\(?[A-F]\)?\}?',gold) and re.search(r'(?i)(?:letter|option|choice)',question):
        kind='choice'
    radix=re.fullmatch(r'([+-]?[0-9A-Za-z]+)_\{?(\d+)\}?',gold)
    if kind=='radix' or (radix and re.search(r'(?i)base|binary|ternary|octal|hexadecimal',question)):
        kind='radix';base=int(row.get('gold_review',{}).get('base') or radix.group(2))
    ordered_multiple=False
    if kind=='symbolic' and dataset!='olympiadbench':
        top=split_top_level(gold)
        if len(top)>1 and all(_wrapped(part) for part in top):
            kind='tuple';multiple=True
            ordered_multiple=bool(re.search(r'(?i)increasing|decreasing|in order',question))
        elif r'\cup' in gold:kind='interval'
        elif _wrapped(gold):
            count=len(split_top_level(gold[1:-1]))
            interval_hint=(gold.startswith('[') or gold.endswith(']') or r'\infty' in gold or
                           re.search(r'(?i)interval|domain|range|solution set',question))
            if count==2 and interval_hint:kind='interval'
            elif count>1:kind='tuple'
        elif re.fullmatch(r'\d{1,2}:\d{2}',gold) and re.search(r'(?i)time|AB:CD',question):kind='clock'
    components=_parts(gold,multiple)
    # OlympiadBench's Numerical label also covers symbolic values and labelled
    # quantities; preserve its metadata while removing explicit quantity labels.
    strip_assignment=(dataset=='olympiadbench' and row['answer_type'] in ('Numerical','Expression'))
    return {'kind':kind,'multiple':multiple,'gold_components':components,'base':base,
        'absolute_tolerance':tolerance,'relative_tolerance':relative_tolerance,
        'integer_reference_exact':dataset=='gsm8k_hard','strip_assignment':strip_assignment,
        'ordered_multiple':ordered_multiple,'rounded_answer_requested':bool(re.search(r'(?i)round|nearest',question)),
        'timeout_seconds':config['timeout_seconds'],'max_expression_characters':config['max_expression_characters']}


def _choice(text):
    text=re.sub(r'\\(?:text|mathrm)\{([^{}]+)\}',r'\1',clean_math(text))
    match=re.fullmatch(r'(?i)(?:option\s+)?[\[(]?\s*([A-F])\s*[\])]?',text)
    if not match:raise ValueError('Expected one choice letter')
    return match.group(1).upper()


def _radix(text,base):
    if not 2<=base<=36:raise ValueError('Unsupported numeral base')
    match=re.fullmatch(r'([+-]?[0-9A-Za-z]+)(?:_\{?(\d+)\}?)?',clean_math(text))
    if not match or (match.group(2) and int(match.group(2))!=base):raise ValueError('Invalid numeral base')
    return int(match.group(1),base)


@lru_cache(maxsize=20000)
def _symbolic(text,timeout_seconds):
    from math_verify import parse,LatexExtractionConfig
    from math_verify.utils import timeout
    from math_verify.parser import parse_latex_cached
    text=text.replace('{,}',',')
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',text):text=text.replace(',','')
    # Generic parsers silently discard a numerical base suffix. Radix questions
    # must reach _radix instead of passing this lossy conversion.
    if re.search(r'\d+_\{?\d+\}?',text):raise ValueError('Numeral base requires a typed grader')
    result=parse(r'\boxed{'+text+'}',extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
        fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=timeout_seconds,raise_on_error=True)
    if result:return result[0]
    # Parsing an already isolated expression directly avoids the extraction
    # regex missing expressions such as sin^2 t. No Python eval is used here.
    value=timeout(timeout_seconds)(parse_latex_cached)(text)
    if value is None:raise ValueError('Unparsed expression')
    return value


def _numeric_equal(gold,pred,spec):
    from math_verify.utils import timeout
    import sympy as sp
    @timeout(spec['timeout_seconds'])
    def compare():
        if not isinstance(gold,sp.Expr) or not isinstance(pred,sp.Expr):return None
        if gold.free_symbols or pred.free_symbols:return None
        if spec['integer_reference_exact'] and gold.is_finite and gold.is_real:
            if gold.is_integer or sp.simplify(gold-sp.floor(gold))==0:
                return bool(sp.simplify(gold-pred)==0)
        if not (spec['absolute_tolerance'] or spec['relative_tolerance']):return None
        g,p=sp.N(gold,40),sp.N(pred,40)
        if not g.is_finite or not p.is_finite or not g.is_real or not p.is_real:return None
        allowed=max(sp.Float(spec['absolute_tolerance']),sp.Float(spec['relative_tolerance'])*abs(g))
        return bool(abs(g-p)<=allowed)
    return compare()


def _interval_set(text,spec):
    import sympy as sp
    from math_verify.utils import timeout
    text=re.sub(r'^[A-Za-z]\s*(?:\\in\s*|=\s*)?(?=[(\[])','',text)
    pieces=[]
    for part in text.split(r'\cup'):
        part=part.strip()
        if part.startswith(r'\{') and part.endswith(r'\}'):
            values=[_symbolic(v,spec['timeout_seconds']) for v in split_top_level(part[2:-2])]
            pieces.append(timeout(spec['timeout_seconds'])(sp.FiniteSet)(*values))
        else:
            if not _wrapped(part):raise ValueError('Expected interval or finite set')
            ends=split_top_level(part[1:-1])
            if len(ends)!=2:raise ValueError('Interval requires two endpoints')
            left,right=[_symbolic(v,spec['timeout_seconds']) for v in ends]
            pieces.append(timeout(spec['timeout_seconds'])(sp.Interval)(left,right,
                left_open=part[0]=='(',right_open=part[-1]==')'))
    return timeout(spec['timeout_seconds'])(sp.Union)(*pieces)


def _equal(gold,pred,spec,kind=None):
    from math_verify import verify
    kind=kind or spec['kind'];gold,pred=clean_math(gold),clean_math(pred)
    if max(len(gold),len(pred))>spec['max_expression_characters']:raise ValueError('Answer exceeds parsing budget')
    if spec['strip_assignment']:
        if gold.count('=')==1:gold=gold.split('=',1)[1].strip()
        if pred.count('=')==1:pred=pred.split('=',1)[1].strip()
    if spec['rounded_answer_requested']:
        if r'\approx' in gold:gold=gold.split(r'\approx')[-1].strip()
        if r'\approx' in pred:pred=pred.split(r'\approx')[-1].strip()
    if kind=='choice':return _choice(gold)==_choice(pred)
    if kind=='radix':return _radix(gold,spec['base'])==_radix(pred,spec['base'])
    if kind=='clock':
        def clock_value(value):
            if not re.fullmatch(r'\d{1,2}:\d{2}',value):raise ValueError('Expected hours:minutes')
            h,m=map(int,value.split(':'))
            if not 0<=h<=23 or not 0<=m<=59:raise ValueError('Invalid clock time')
            return h,m
        return clock_value(gold)==clock_value(pred)
    if kind=='interval':
        a,b=_interval_set(gold,spec),_interval_set(pred,spec)
        return bool(verify(a,b,strict=True,timeout_seconds=spec['timeout_seconds'],raise_on_error=True))
    if kind=='tuple':
        if not(gold.startswith('(') and gold.endswith(')') and pred.startswith('(') and pred.endswith(')')):return False
        left,right=split_top_level(gold[1:-1]),split_top_level(pred[1:-1])
        return len(left)==len(right) and all(_equal(a,b,spec,'symbolic') for a,b in zip(left,right))
    a,b=_symbolic(gold,spec['timeout_seconds']),_symbolic(pred,spec['timeout_seconds'])
    numeric=_numeric_equal(a,b,spec)
    if numeric is not None:return numeric
    return bool(verify(a,b,strict=True,float_rounding=spec.get('float_rounding',12),
                       timeout_seconds=spec['timeout_seconds'],raise_on_error=True))


def _perfect_matching(left,right,equal):
    if len(left)!=len(right):return False
    # Bipartite matching, so overlapping tolerance intervals cannot make a
    # greedy early choice discard an otherwise valid one-to-one assignment.
    edges=[[j for j,b in enumerate(right) if equal(a,b)] for a in left]
    assigned={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in assigned or augment(assigned[j],seen):assigned[j]=i;return True
        return False
    return all(augment(i,set()) for i in range(len(left)))


def grade_typed_response(prediction,row,config):
    from math_verify.errors import TimeoutException
    answer=extract_boxed(prediction)
    if answer is None:return {'is_correct':False,'status':'missing_final_box','predicted_answer':None}
    try:
        spec=compile_answer_spec(row,config);parts=_parts(answer,spec['multiple'])
        if spec['ordered_multiple']:
            correct=len(spec['gold_components'])==len(parts) and all(_equal(a,b,spec) for a,b in zip(spec['gold_components'],parts))
        else:correct=_perfect_matching(spec['gold_components'],parts,lambda a,b:_equal(a,b,spec))
        return {'is_correct':correct,'status':'graded','predicted_answer':answer,
                'answer_kind':spec['kind'],'component_count':len(parts)}
    except (Exception,TimeoutException) as error:
        return {'is_correct':False,'status':'grading_error','predicted_answer':answer,
                'error_type':type(error).__name__,'error':str(error)}
