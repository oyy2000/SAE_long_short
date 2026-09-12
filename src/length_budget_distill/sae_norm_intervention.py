"""Measured interventions with explicit per-response timing and norm control.

Reuses the SAE checkpoint conventions and TopK readback of MeasuredSAEController.
Generation starts each condition with an independent cache and identical per-item
random uniforms, avoiding cache mutation and padding-dependent RNG coupling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .sae_paired_intervention import MeasuredSAEController
from .sae_local_data import ROOT, paths, jsonl, evidence, verify
from .experiment_io import read_json, write_json_exclusive
from .verifiers import extract_final_answer, verify_answer


class NormMatchedController(MeasuredSAEController):
    def __init__(self, *, short_ids, long_ids, random_ids, **kwargs):
        self.short_ids_list = list(short_ids)
        self.long_ids_list = list(long_ids)
        self.random_ids_list = list(random_ids)
        super().__init__(short_feature_ids=short_ids, long_feature_ids=long_ids,
                         random_feature_ids=random_ids, measured_feature_ids=list(short_ids) + list(long_ids),
                         **kwargs)
        from safetensors.torch import load_file
        self.decoder = load_file(str(kwargs['checkpoint_path']))['decoder_weight'].to(
            device=kwargs['device'], dtype=kwargs['dtype'])
        self.spec = None
        self.live = None
        self.step = 0

    def begin(self, spec, batch_size):
        torch = self.torch
        if not 0 <= spec['rho'] <= self.maximum_delta_fraction:
            raise ValueError('Requested norm exceeds the registered intervention range')
        self.spec = spec
        self.live = torch.ones(batch_size, dtype=torch.bool, device=self.decoder.device)
        self.step = 0
        self.per_row = torch.zeros(batch_size, 10, dtype=torch.float64, device=self.decoder.device)
        self.max_fraction = torch.zeros(batch_size, device=self.decoder.device)
        count = spec['count']
        self.selected_short = self.short_ids_list[:count]
        self.selected_long = self.long_ids_list[:count]
        self.selected_random = self.random_ids_list[:count]
        def direction(ids):
            if not ids:
                raise ValueError('No candidate directions available')
            d = self.decoder[ids].float().sum(0)
            return d / d.norm().clamp_min(1e-12)
        self.direction = direction(self.selected_short)
        if spec['mode'] == 'random':
            self.direction = direction(self.selected_random)
        elif spec['mode'] == 'joint':
            d = direction(self.selected_short) - direction(self.selected_long)
            self.direction = d / d.norm().clamp_min(1e-12)
        self._handle = self.layer_module.register_forward_hook(self._hook)

    def end(self):
        self._handle.remove()
        self._handle = None
        rows = []
        for numbers, maximum in zip(self.per_row.cpu().tolist(), self.max_fraction.cpu().tolist()):
            n, changed, fraction, short_before, short_after, long_before, long_after, target_changed, gated, norm = numbers
            rows.append({'eligible_forward_positions': int(n), 'modified_positions': int(changed),
                         'mean_delta_to_hidden_norm_fraction': fraction/max(n,1),
                         'max_delta_to_hidden_norm_fraction': maximum,
                         'mean_short_activation_before': short_before/max(n,1),
                         'mean_short_activation_after': short_after/max(n,1),
                         'mean_long_activation_before': long_before/max(n,1),
                         'mean_long_activation_after': long_after/max(n,1),
                         'target_code_changed_fraction': target_changed/max(n,1),
                         'gated_long_positions': int(gated), 'mean_hidden_norm': norm/max(n,1),
                         'scope': 'per-response live positions only, including position predicting EOS'})
        return rows

    def _hook(self, module, inputs, output):
        del module, inputs
        torch = self.torch
        if self.step < self.spec['start'] or self.spec['rho'] == 0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        flat = hidden[:, -1, :]
        before = self.feature_values(flat).float()
        short_count = len(self.short_ids_list)
        delta_direction = self.direction.expand_as(flat)
        gated = torch.zeros(len(flat), dtype=torch.bool, device=flat.device)
        if self.spec['mode'] == 'long':
            long_values = before[:, short_count:short_count + len(self.selected_long)]
            raw = -(long_values @ self.decoder[self.selected_long].float())
            gated = raw.norm(dim=-1) > 0
            delta_direction = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        requested = self.spec['rho'] * flat.float().norm(dim=-1, keepdim=True) * delta_direction
        requested *= self.live.unsqueeze(1)
        changed = hidden.clone()
        changed[:, -1, :] = (flat.float() + requested).to(flat.dtype)
        actual = changed[:, -1, :].float() - flat.float()
        fractions = actual.norm(dim=-1) / flat.float().norm(dim=-1).clamp_min(1e-12)
        after = self.feature_values(changed[:, -1, :]).float()
        live = self.live.double()
        self.per_row[:, 0] += live
        self.per_row[:, 1] += (fractions > 0).double() * live
        self.per_row[:, 2] += fractions.double() * live
        self.per_row[:, 3] += before[:, :len(self.selected_short)].sum(1).double() * live
        self.per_row[:, 4] += after[:, :len(self.selected_short)].sum(1).double() * live
        self.per_row[:, 5] += before[:, short_count:short_count+len(self.selected_long)].sum(1).double() * live
        self.per_row[:, 6] += after[:, short_count:short_count+len(self.selected_long)].sum(1).double() * live
        self.per_row[:, 7] += ((after-before).abs().sum(1) > 0).double() * live
        self.per_row[:, 8] += gated.double() * live
        self.per_row[:, 9] += flat.float().norm(dim=-1).double() * live
        self.max_fraction = torch.maximum(self.max_fraction, fractions * self.live)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed


def sample_from_uniform(logits, uniforms, temperature, top_p):
    """Inverse-CDF nucleus sampling driven by item-specific fixed uniforms."""
    import torch
    if temperature == 0:
        return logits.argmax(-1)
    sorted_logits, order = torch.sort(logits.float()/temperature, descending=True, dim=-1)
    probs = sorted_logits.softmax(-1)
    cumulative = probs.cumsum(-1)
    remove = cumulative - probs >= top_p
    probs = probs.masked_fill(remove, 0)
    probs /= probs.sum(-1, keepdim=True)
    slot = torch.searchsorted(probs.cumsum(-1).contiguous(), uniforms.unsqueeze(1).contiguous())
    return order.gather(1, slot.clamp_max(order.shape[1]-1)).squeeze(1)


def generate_condition(model, tokenizer, controller, spec, rows, settings):
    import torch
    rendered = [tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],
                 tokenize=False, add_generation_prompt=True) for r in rows]
    encoded = tokenizer(rendered, padding=True, return_tensors='pt')
    ids = encoded['input_ids'].to(model.device)
    mask = encoded['attention_mask'].to(model.device)
    eos = tokenizer.eos_token_id
    seeds = [int.from_bytes(hashlib.sha256(f"{settings['seed']}:{r['problem_id']}".encode()).digest()[:4], 'little') for r in rows]
    uniforms = torch.tensor(np.stack([np.random.default_rng(s).random(settings['max_new_tokens']) for s in seeds]),
                            dtype=torch.float32, device=model.device)
    cache = None
    generated = []
    controller.begin(spec, len(rows))
    try:
        with torch.inference_mode():
            for step in range(settings['max_new_tokens']):
                controller.step = step
                positions = mask.long().cumsum(-1)-1
                positions.masked_fill_(mask == 0, 1)
                if cache is not None:
                    positions = positions[:, -1:]
                output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                     past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = model.lm_head(output.last_hidden_state[:, -1, :])
                token = sample_from_uniform(logits, uniforms[:, step], settings['temperature'], settings['top_p'])
                token = torch.where(controller.live, token, torch.full_like(token, eos))
                generated.append(token)
                controller.live &= token != eos
                if not controller.live.any():
                    break
                ids = token.unsqueeze(1)
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
        diagnostics = controller.end()
    finally:
        if controller._handle is not None:
            controller.end()
    token_rows = torch.stack(generated, dim=1).cpu().tolist()
    output_rows = []
    for row, tokens, diag, seed in zip(rows, token_rows, diagnostics, seeds):
        ended = eos in tokens
        if ended:
            tokens = tokens[:tokens.index(eos)+1]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        prediction = extract_final_answer(text)
        gold = extract_final_answer(row['gold_answer'])
        if gold is None:
            raise ValueError('Missing source gold answer')
        output_rows.append({'problem_id': row['problem_id'], 'question_split': row['question_split'],
                            'condition': spec['name'], 'spec': spec, 'response': text, 'token_ids': tokens,
                            'output_token_count': len(tokens), 'hit_max_new_tokens': not ended,
                            'predicted_answer': prediction, 'gold_answer': gold,
                            'is_correct': verify_answer(prediction, gold), 'seed': seed,
                            'diagnostics': diag})
    return output_rows


def registered_specs():
    specs = [{'name':'no_steering', 'dictionary':'full_trace_balanced','mode':'short','count':4,'rho':0.,'start':0}]
    def add(dictionary, mode, count, strengths, start=0):
        for rho in strengths:
            name=f'{dictionary}__{mode}{count}__rho{rho:g}__start{start}'
            specs.append(dict(name=name,dictionary=dictionary,mode=mode,count=count,rho=rho,start=start))
    add('full_trace_balanced','short',4,[.013,.05,.15,.30])
    for mode in ('short','joint'):
        add('full_trace_balanced',mode,16,[.05,.15,.30])
    add('full_trace_balanced','long',16,[.15,.30])
    add('full_trace_balanced','random',16,[.013,.05,.15,.30])
    add('full_trace_balanced','random',4,[.15])
    add('prefix64_trace_balanced','short',16,[.05,.15,.30])
    add('prefix64_trace_balanced','random',16,[.05,.15,.30])
    add('full_trace_balanced','short',4,[.013,.15],64)
    add('full_trace_balanced','random',4,[.15],64)
    return specs


def freeze_generation(config):
    root, _ = paths(config)
    specs = registered_specs()
    sources = {}
    for condition in config['sampling']['conditions']:
        filename = root/'feature_screen'/condition/'SCREEN_COMPLETE.json'
        screen = read_json(filename)
        verify(screen['checkpoint'])
        if len([f for f in screen['features'] if f['direction']=='short']) < 16 or len([f for f in screen['features'] if f['direction']=='long']) < 16:
            raise ValueError('Insufficient dev candidates for registered breadth comparison')
        sources[condition] = evidence(filename)
    corpus = jsonl(root/'corpus.jsonl')
    cohorts = {}
    for split in ('dev','test'):
        rows = [r for r in corpus if r['question_split']==split and r['analysis_length_label']=='short']
        rows.sort(key=lambda r: hashlib.sha256(f"61073:{r['problem_id']}".encode()).hexdigest())
        cohorts[split] = [r['problem_id'] for r in rows[:config['generation'][f'{split}_questions']]]
    write_json_exclusive(root/'protocol/generation_protocol.json', {
        'specs':specs,'cohorts':cohorts,'sources':sources,'config_hash':evidence(root/'protocol/frozen_protocol.json'),
        'source_code':evidence(Path(__file__)),'scope':'exploratory; fixed dev ranking with confirmation flags reported',
        'note':'Random16 is the common same-norm control; random4 at rho=.15 separately controls dictionary count.'})


def generate_shard(config, split, shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root, _ = paths(config)
    protocol = read_json(root/'protocol/generation_protocol.json')
    verify(protocol['source_code'])
    corpus = {r['problem_id']:r for r in jsonl(root/'corpus.jsonl') if r['analysis_length_label']=='short'}
    all_ids = protocol['cohorts'][split]
    rows = [corpus[pid] for i,pid in enumerate(all_ids) if i%config['generation']['shards']==shard]
    directory = root/'generation'/split/f'shard_{shard}'
    directory.mkdir(parents=True,exist_ok=False)
    teacher=config['teacher']['snapshot_path']
    tokenizer=AutoTokenizer.from_pretrained(teacher,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id=tokenizer.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(teacher,local_files_only=True,torch_dtype=torch.bfloat16,
                                              attn_implementation='sdpa').cuda().eval()
    controllers={}
    for dictionary in sorted({s['dictionary'] for s in protocol['specs']}):
        verify(protocol['sources'][dictionary])
        screen=read_json(protocol['sources'][dictionary]['path'])
        verify(screen['checkpoint'])
        features=screen['features']
        short=[r['feature_id'] for r in features if r['direction']=='short']
        long=[r['feature_id'] for r in features if r['direction']=='long']
        pool=np.array(sorted(set(range(28672))-set(short+long)))
        random=np.random.default_rng(config['generation']['random_seed']).choice(pool,16,replace=False).tolist()
        controllers[dictionary]=NormMatchedController(short_ids=short,long_ids=long,random_ids=random,
            torch_module=torch,checkpoint_path=screen['checkpoint']['path'],layer_module=model.model.layers[17],
            k=64,maximum_delta_fraction=.30,device=model.device,dtype=torch.bfloat16)
    output=directory/'predictions.jsonl'
    n=0
    with output.open('x') as handle:
        batch_size=config['generation']['batch_size']
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            baseline=None
            for spec in protocol['specs']:
                generated=generate_condition(model,tokenizer,controllers[spec['dictionary']],spec,batch,config['generation'])
                if spec['name']=='no_steering': baseline=generated
                if spec['start']:
                    for a,b in zip(baseline,generated):
                        if a['token_ids'][:spec['start']]!=b['token_ids'][:spec['start']]:
                            raise RuntimeError('Late intervention changed the natural prefix')
                for row in generated: handle.write(json.dumps(row,ensure_ascii=False)+'\n');n+=1
                handle.flush()
                print(json.dumps({'split':split,'shard':shard,'records':n,'condition':spec['name']}),flush=True)
    write_json_exclusive(directory/'GENERATION_COMPLETE.json',{'status':'complete','records':n,
        'problem_ids':[r['problem_id'] for r in rows],'predictions':evidence(output),
        'protocol':evidence(root/'protocol/generation_protocol.json'),'formal_claim_allowed':False})
