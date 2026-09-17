"""Explicit reviewed answer definitions without changing frozen v2 scoring.

This module extends the old balanced-box logic with outer-box spans and unary
sign context. It intentionally lives separately: historical audit markers bind
the original math_grading.py and typed_math_grading.py bytes.
"""
import re

from .baseline_data_preflight import normalize_question
from . import typed_math_grading as v2


KINDS = {'symbolic', 'tuple', 'vector', 'interval', 'degree', 'percentage', 'radix', 'clock', 'choice', 'periodic', 'text'}


def outer_box_spans(text):
    """Balanced outer boxes; nested boxes are part of the outer expression."""
    result = []
    end = 0
    for match in re.finditer(r'\\(?:boxed|fbox)\b\s*', text):
        if match.start() < end: continue
        if text[match.end():match.end()+1] != '{':
            # TeX permits a single unbraced token. The reviewed source cases use
            # one decimal digit; do not reinterpret an ambiguous bare expression.
            bare = re.match(r'[0-9](?![0-9A-Za-z]|\.\d)', text[match.end():])
            if bare:
                end = match.end()+1
                result.append({'start': match.start(), 'end': end, 'value': bare[0]})
                continue
            result.append({'start': match.start(), 'end': len(text), 'value': None})
            break
        depth = 1
        for i in range(match.end()+1, len(text)):
            cursor = i-1
            while cursor >= 0 and text[cursor] == '\\': cursor -= 1
            if (i-1-cursor) % 2: continue
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            if depth == 0:
                end = i+1
                result.append({'start': match.start(), 'end': end, 'value': text[match.end()+1:i].strip()})
                break
        else:
            # An incomplete final box must not fall back to an earlier answer.
            result.append({'start': match.start(), 'end': len(text), 'value': None})
            break
    return result


def final_value(text, kind):
    spans = outer_box_spans(text)
    if not spans or not spans[-1]['value']: raise ValueError('Missing complete final box')
    span = spans[-1]
    value = span['value']
    if re.search(r'\\(?:boxed|fbox)\b', value):
        raise ValueError('Nested final box needs explicit interpretation')
    prefix = text[:span['start']].rstrip()
    if prefix.endswith('-'):
        before = prefix[:-1].rstrip()
        unary = (not before or before[-1:] in '=([:$' or
                 re.search(r'(?:\\\[|\\\(|\b(?:is|equals|answer))$', before, re.I))
        if unary:
            # The leading-minus reference case is inside a display-math block.
            # A standalone Markdown bullet with a space is ambiguous.
            if not before and text[:span['start']].endswith('- '):
                raise ValueError('Ambiguous bullet versus unary minus')
            value = '-('+value+')'
    tail = text[span['end']:]
    if kind == 'radix':
        suffix = re.match(r'\s*_\s*(?:\{\s*(\d+)\s*\}|(\d+))', tail)
        if suffix: value += '_{'+(suffix.group(1) or suffix.group(2))+'}'
    if kind == 'clock':
        suffix = re.match(r'\s*(?:\\[,!;]\s*)*(?:\\(?:text|mathrm)\{\s*)?([ap])\.?\s*m\.?\s*\}?', tail, re.I)
        if suffix: value += ' '+suffix.group(1)+'m'
    return value


def plain_wrappers(text):
    for _ in range(4):
        changed = re.sub(r'\\(?:text|mbox|mathrm)\{([^{}]*)\}', r'\1', text)
        if changed == text: break
        text = changed
    return text


def final_values(text, spec):
    """Read the last contiguous answer list without collecting intermediate work.

    Separate final boxes joined by commas/and/or are one answer list. Equations,
    new sentences and explanatory prose end a list. Only reviewed target labels
    may appear between a connector and its following box.
    """
    spans = outer_box_spans(text)
    if not spans: raise ValueError('Missing complete final box')
    first = len(spans)-1
    while first > 0:
        previous, current = spans[first-1], spans[first]
        between = plain_wrappers(text[previous['end']:current['start']])
        between = re.sub(r'\\[\[\]()]|\$|\\[,!;:]|\\(?:quad|qquad)\b', ' ', between).strip()
        for lhs in spec.get('assignment_lhs', []):
            between = re.sub(re.escape(lhs)+r'\s*=\s*$', '', between).strip()
        if not re.fullmatch(r'(?:\s|[,;]|\band\b|\bor\b)*', between, re.I): break
        if previous['value'] is None: raise ValueError('Incomplete box in final answer list')
        first -= 1
    values = []
    for i in range(first, len(spans)):
        stop = spans[i+1]['start'] if i+1 < len(spans) else len(text)
        values.append(final_value(text[:stop], spec['kind']))
    return values


def normalize_expression(text):
    text = v2.clean_math(text)
    # TeX's unbraced arguments are single tokens: \frac\pi 2 is pi/2,
    # and \phantom -1 hides only the minus, leaving a visible positive 1.
    atom = r'(\{[^{}]*\}|\\[A-Za-z]+|[^\\\s{}])'
    text = re.sub(r'\\(?:hphantom|vphantom|phantom)\s*'+atom, '', text)
    def fraction(match):
        def brace(value):
            return value if value.startswith('{') else '{'+value+'}'
        return r'\frac'+brace(match[1])+brace(match[2])
    text = re.sub(r'\\(?:frac|tfrac)\s*'+atom+r'\s*'+atom, fraction, text)
    def grouped(value):
        if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', value.strip()):
            return value.replace(',', '')
        return value
    # Restrict grouped-digit cleanup to fraction arguments. A tuple (1,234)
    # and a solution list 1,234 must retain their two components.
    text = re.sub(r'(\\(?:frac|tfrac)\s*\{)([^{}]*)(\}\s*\{)([^{}]*)(\})',
                  lambda m: m[1]+grouped(m[2])+m[3]+grouped(m[4])+m[5], text)
    return text.strip()


def strip_registered_unit(value, spec):
    """Accept omitted units or a registered spelling, never arbitrary unit loss."""
    if r'\$' in value and spec.get('currency_symbol') != '$':
        raise ValueError('Dollar notation contradicts the registered quantity unit')
    value = plain_wrappers(normalize_expression(value)).replace('\\ ', ' ').strip().rstrip('.').strip()
    if spec.get('rounded_answer_requested'):
        value = re.sub(r'^\\approx\s*', '', value)
    aliases = sorted(spec['unit_aliases'], key=len, reverse=True)
    for alias in aliases:
        if value.casefold().endswith(alias.casefold()):
            value = value[:-len(alias)].strip()
            break
    permitted = {'frac','tfrac','sqrt','pi','cdot','times','div','log','ln','sin','cos','tan','exp','overline'}
    commands = re.findall(r'\\([A-Za-z]+)', value)
    if any(command not in permitted for command in commands):
        raise ValueError('Unregistered command in a quantity answer')
    if re.search(r'[A-Za-z]', re.sub(r'\\[A-Za-z]+', '', value)):
        raise ValueError('Unregistered unit or prose in a quantity answer')
    if not value: raise ValueError('Quantity answer has no numeric value')
    return value


def parts(value, spec):
    if spec.get('unit_aliases'):
        value = strip_registered_unit(value, spec)
    value = normalize_expression(value)
    if spec['kind'] == 'choice' and spec['mode'] == 'single':
        # A comma may separate a letter from its label; parse the complete label
        # before deciding whether it contradicts the chosen option.
        return [value]
    if spec['kind'] in {'text', 'choice'}:
        value = plain_wrappers(value)
    # Connectors explicitly separate complete alternatives/solutions; their
    # presence must not let a parser silently discard a second answer.
    value = re.sub(r'\\(?:text|mbox|mathrm)\{\s*(?:,\s*)?(?:and|or)\s*\}', ',', value, flags=re.I)
    value = re.sub(r'\s+(?:and|or)\s+', ',', value, flags=re.I)
    value = re.sub(r'\\(?:quad|qquad)\b', ' ', value)
    value = re.sub(r',\s*,', ',', value)
    if spec['mode'] == 'single' and spec['kind'] == 'symbolic':
        # The reviewed scalar type disambiguates a grouped numeral from a list.
        # Never remove commas from an all-answer target or from tuple coordinates.
        grouped = re.sub(r',\s+', ',', value.replace('{,}', ','))
        if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', grouped):
            return [grouped.replace(',', '')]
    if spec['kind'] == 'vector':
        # Rows of a matrix are not top-level answer separators.
        value = value.strip()
        if value.startswith(r'\begin') or value.startswith(r'-\begin') or value.startswith('('):
            return [value]
    if spec['kind'] != 'tuple' and spec['mode'] == 'all':
        if value.startswith(r'\{') and value.endswith(r'\}'): value = value[2:-2]
        elif value.startswith('(') and value.endswith(')') and v2._wrapped(value): value = value[1:-1]
    pieces = v2.split_top_level(value)
    expanded = []
    for piece in pieces:
        if r'\pm' in piece:
            if piece.count(r'\pm') != 1 or r'\mp' in piece:
                raise ValueError('Ambiguous coupled plus/minus expressions')
            expanded.extend([piece.replace(r'\pm', '+'), piece.replace(r'\pm', '-')])
        else: expanded.append(piece)
    return [p.strip() for p in expanded]


def base_spec(config):
    return v2.compile_answer_spec({'dataset': 'math_train', 'question': '', 'answer': '0'}, config)


def clock_seconds(value, default_meridiem=None):
    value = plain_wrappers(normalize_expression(value)).strip()
    match = re.fullmatch(r'(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*(\d{2}))?(?:\s*([ap])\.?\s*m\.?)?', value, re.I)
    if not match: raise ValueError('Expected a complete clock time')
    h, m = int(match[1]), int(match[2])
    seconds = int(match[3] or 0)
    if not 0 <= m < 60: raise ValueError('Invalid clock minutes')
    if not 0 <= seconds < 60: raise ValueError('Invalid clock seconds')
    # Some reviewed questions explicitly request a time without am/pm. Only
    # those definitions supply the context needed to interpret a twelve-hour
    # answer; an explicit contradictory meridiem is never overridden.
    meridiem = match[4] or (default_meridiem[:1] if default_meridiem and 1 <= h <= 12 else None)
    if meridiem:
        if not 1 <= h <= 12: raise ValueError('Invalid twelve-hour clock hour')
        h = h % 12 + (12 if meridiem.lower() == 'p' else 0)
    elif not 0 <= h < 24: raise ValueError('Invalid twenty-four-hour clock hour')
    return h*3600+m*60+seconds


def vector_components(value):
    value = normalize_expression(value)
    sign = 1
    if value.startswith('-'):
        sign = -1
        value = value[1:].strip()
    matrix = re.fullmatch(r'\\begin\{([pb]?matrix)\}(.*?)\\end\{\1\}', value, re.S)
    if matrix:
        body = matrix[2].strip()
        if r'\\' in body and '&' in body: raise ValueError('A matrix with several rows and columns is not a vector')
        values = re.split(r'\\\\|&', body)
        if not values[-1].strip(): values.pop()
    elif value.startswith('(') and value.endswith(')') and v2._wrapped(value):
        values = v2.split_top_level(value[1:-1])
    else: raise ValueError('Expected a coordinate tuple or one-dimensional matrix')
    if not values or any(not v.strip() for v in values): raise ValueError('Empty vector coordinate')
    return [v.strip() if sign == 1 else '-('+v.strip()+')' for v in values]


def percentage_value(value, comparison):
    value = plain_wrappers(normalize_expression(value))
    value = re.sub(r'(?i)\s*(?:percentage\s+points?|percent(?:age)?|pp)\s*$', r'\\%', value)
    if '=' in value:
        terms = v2.split_top_level(value, '=')
        if len(terms) != 2 or not re.search(r'\\?%$', terms[1]):
            raise ValueError('Unsupported percentage equivalence chain')
        percent = re.sub(r'\\?%$', '', terms[1]).strip()
        if not v2._equal(terms[0], r'\frac{'+percent+'}{100}', comparison):
            raise ValueError('Contradictory percentage equivalence chain')
        return percent
    return re.sub(r'\\?%$', '', value).strip()


def compare(gold, prediction, spec, config):
    kind = spec['kind']
    comparison = base_spec(config)
    if spec.get('unit_aliases'):
        gold, prediction = strip_registered_unit(gold, spec), strip_registered_unit(prediction, spec)
    gold, prediction = normalize_expression(gold), normalize_expression(prediction)
    if spec.get('rounded_answer_requested'):
        comparison['rounded_answer_requested'] = True
        gold, prediction = [re.sub(r'^\\approx\s*', '', value) for value in (gold, prediction)]
    for lhs in spec.get('assignment_lhs', []):
        match = re.match(r'^'+re.escape(lhs)+r'\s*=\s*', prediction)
        if match:
            prediction = prediction[match.end():]
            break
    if kind == 'clock':
        return clock_seconds(gold, spec.get('clock_meridiem')) == clock_seconds(prediction, spec.get('clock_meridiem'))
    if kind == 'radix': return v2._radix(plain_wrappers(gold), spec['base']) == v2._radix(plain_wrappers(prediction), spec['base'])
    if kind == 'vector':
        left, right = vector_components(gold), vector_components(prediction)
        return len(left) == len(right) and all(v2._equal(a, b, comparison) for a, b in zip(left, right))
    if kind == 'percentage':
        return v2._equal(percentage_value(gold, comparison), percentage_value(prediction, comparison), comparison)
    if kind == 'degree':
        def degrees(value):
            return re.sub(r'(?:\^\s*\{?\s*\\circ\s*\}?|\\text\{\s*degrees?\s*\}|\s+degrees?)$', '', value, flags=re.I).strip()
        return v2._equal(degrees(gold), degrees(prediction), comparison)
    if kind == 'text':
        predicted_text = normalize_question(plain_wrappers(prediction))
        predicted_text = spec.get('literal_aliases', {}).get(predicted_text, predicted_text)
        return normalize_question(plain_wrappers(gold)) == normalize_question(predicted_text)
    if kind == 'choice':
        value = plain_wrappers(prediction).strip()
        value = spec.get('literal_aliases', {}).get(normalize_question(value), value)
        match = re.fullmatch(r'(?:[Oo]ption\s+)?[\[(]?([A-Za-z])[\])]?(?:(?:\s*[:,.)-]\s*|\s+)(.*))?', value)
        if not match: raise ValueError('Expected one choice letter with an optional registered label')
        letter = match[1].upper()
        if match[2]:
            label = spec.get('choice_options', {}).get(letter)
            if isinstance(label, dict):
                same_label = label['kind'] in {'symbolic','tuple','interval'} and v2._equal(
                    label['answer'], match[2], comparison, kind=label['kind'])
            else:
                labels = ([] if label is None else [label])+spec.get('choice_label_aliases', {}).get(letter, [])
                same_label = any(normalize_question(plain_wrappers(match[2])) == normalize_question(plain_wrappers(x)) for x in labels)
            if not same_label:
                raise ValueError('Choice letter and appended label are inconsistent or unregistered')
        return letter == gold.upper()
    if kind == 'periodic':
        import sympy as sp
        from math_verify.utils import timeout
        @timeout(config['timeout_seconds'])
        def same_phase():
            period = v2._symbolic(spec['period'], config['timeout_seconds'])
            delta = v2._symbolic(prediction, config['timeout_seconds'])-v2._symbolic(gold, config['timeout_seconds'])
            return sp.simplify(delta/period).is_integer is True
        return same_phase()
    # Number words are accepted only when explicitly enumerated by the reviewed
    # definition; no guessing of numerical quantities in arbitrary prose.
    prediction = spec.get('literal_aliases', {}).get(normalize_question(plain_wrappers(prediction)), prediction)
    return v2._equal(gold, prediction, comparison, kind=kind if kind in {'tuple', 'interval'} else 'symbolic')


def validate_definition(spec):
    if spec['mode'] not in {'single', 'all', 'any'} or spec['kind'] not in KINDS:
        raise ValueError('Unknown reviewed answer mode or kind')
    answers = spec['answers']
    if not answers or not isinstance(answers, list) or any(not isinstance(x, str) or not x.strip() for x in answers):
        raise ValueError('Missing complete canonical answers')
    if spec['mode'] == 'single' and len(answers) != 1: raise ValueError('A single target requires one canonical answer')
    if spec['mode'] != 'all' and spec.get('ordered'): raise ValueError('Ordering only applies to all-answer targets')
    if spec['kind'] == 'radix' and not 2 <= spec.get('base', 0) <= 36: raise ValueError('Missing valid output base')
    if spec['kind'] == 'periodic' and not spec.get('period'): raise ValueError('Missing equivalence period')
    if spec['kind'] == 'choice' and any(not re.fullmatch(r'[A-Z]', x) for x in answers):
        raise ValueError('Canonical choices require one uppercase letter per component')
    if 'clock_meridiem' in spec and (spec['kind'] != 'clock' or spec['clock_meridiem'] not in {'am', 'pm'}):
        raise ValueError('Clock meridiem requires explicitly reviewed am/pm context')
    if 'unit_aliases' in spec and (spec['kind'] != 'symbolic' or not spec['unit_aliases'] or
                                 any(not isinstance(x, str) or not x for x in spec['unit_aliases'])):
        raise ValueError('Quantity units require explicit nonempty symbolic aliases')


def grade_reviewed_response(prediction, row, config):
    """Require an explicit reviewed definition; there is no silent v2 fallback."""
    from math_verify.errors import TimeoutException
    spec = row['reviewed_answer']
    validate_definition(spec)
    try:
        values = final_values(prediction, spec)
        if sum(map(len, values)) > config['max_expression_characters']: raise ValueError('Answer exceeds parsing budget')
        predicted = [part for value in values for part in parts(value, spec)]
        if spec['mode'] == 'single':
            # A final list of equivalent expressions is still one scalar value;
            # every expression must agree, so a conflicting alternative fails.
            correct = bool(predicted) and all(compare(spec['answers'][0], p, spec, config) for p in predicted)
        elif spec['mode'] == 'any':
            correct = len(predicted) == 1 and any(compare(a, predicted[0], spec, config) for a in spec['answers'])
        elif spec.get('ordered'):
            correct = len(predicted) == len(spec['answers']) and all(compare(a, b, spec, config) for a, b in zip(spec['answers'], predicted))
        else:
            correct = v2._perfect_matching(spec['answers'], predicted, lambda a, b: compare(a, b, spec, config))
        return {'is_correct': bool(correct), 'status': 'graded_reviewed', 'predicted_answer': '; '.join(values),
                'mode': spec['mode'], 'answer_kind': spec['kind'], 'component_count': len(predicted),
                'final_box_count': len(values)}
    except (Exception, TimeoutException) as error:
        return {'is_correct': False, 'status': 'reviewed_grading_error',
                'error_type': type(error).__name__, 'error': str(error)}
