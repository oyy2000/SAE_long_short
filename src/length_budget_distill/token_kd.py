"""Teacher-forced, completion-only forward-KL distillation on aligned token IDs."""
import torch
import torch.nn.functional as F


def completion_kd_loss(student_logits, teacher_logits, labels, *, vocabulary_size,
                       alpha, temperature, chunk_tokens=64, num_items_in_batch=None):
    if not 0 <= alpha <= 1 or temperature <= 0 or chunk_tokens < 1:
        raise ValueError('Invalid KD mixture, temperature, or chunk size')
    if student_logits.shape[:2] != labels.shape:
        raise ValueError('Student logits and labels are not aligned')
    mask = labels[:, 1:] != -100
    if not mask.any():
        raise ValueError('No supervised next-token positions')
    targets = labels[:, 1:][mask]
    if torch.any(targets < 0) or torch.any(targets >= vocabulary_size):
        raise ValueError('Target outside the validated shared vocabulary')
    student = student_logits[:, :-1][mask]
    ce = F.cross_entropy(student.float(), targets)
    kd = ce.new_zeros(())
    discarded = ce.new_zeros(())
    if alpha:
        if teacher_logits is None or teacher_logits.shape[:2] != labels.shape:
            raise ValueError('Aligned teacher logits required')
        if vocabulary_size > min(student.shape[-1], teacher_logits.shape[-1]):
            raise ValueError('Shared vocabulary exceeds a model output')
        teacher = teacher_logits[:, :-1][mask].detach()
        for start in range(0, len(targets), chunk_tokens):
            s = student[start:start+chunk_tokens, :vocabulary_size].float()/temperature
            t_full = teacher[start:start+chunk_tokens].float()/temperature
            t = t_full[:, :vocabulary_size]
            logp = F.log_softmax(t, dim=-1)
            logq = F.log_softmax(s, dim=-1)
            kd = kd + F.kl_div(logq, logp, reduction='sum', log_target=True)
            discarded = discarded + (1-(t.logsumexp(-1)-t_full.logsumexp(-1)).exp()).sum()
        kd = kd * temperature**2 / len(targets)
        discarded = discarded/len(targets)
    loss = (1-alpha)*ce + alpha*kd
    if num_items_in_batch is not None:
        if num_items_in_batch < len(targets):
            raise ValueError('Accumulation denominator smaller than current token count')
        # Transformers 4.48 passes the supervised-token count of the whole
        # accumulation group and skips its legacy per-microbatch division.
        loss = loss * (len(targets) / num_items_in_batch)
    return loss, {'ce':ce.detach(), 'temperature_scaled_kl':kd.detach(),
                  'discarded_teacher_probability':discarded.detach(),
                  'supervised_tokens':len(targets)}


def install_kd_loss(trainer, cfg, vocabulary_size):
    """Keep the pinned Trainer/optimizer/LoRA recipe; change only compute_loss."""
    from transformers import AutoModelForCausalLM
    spec=cfg['kd']; teacher=None
    if not trainer.model_accepts_loss_kwargs:
        raise ValueError('This frozen KD adapter requires Trainer token-count loss kwargs')
    if spec['alpha']:
        # Loading the fixed teacher must not change the student's random stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            teacher=AutoModelForCausalLM.from_pretrained(cfg['teacher']['snapshot_path'],
                torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,
                device_map={'':torch.cuda.current_device()})
        teacher.eval();teacher.requires_grad_(False);teacher.config.use_cache=False
    meter={'microbatches':0,'supervised_tokens':0,'ce_sum':0.,'kl_sum':0.,
           'teacher_discarded_probability_sum':0.,'teacher_frozen':True}
    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        features={k:inputs[k] for k in ('input_ids','attention_mask')}
        output=model(**features,use_cache=False)
        with torch.no_grad():
            teacher_logits=teacher(**features,use_cache=False).logits if teacher is not None else None
        loss,values=completion_kd_loss(output.logits,teacher_logits,inputs['labels'],
            vocabulary_size=vocabulary_size,num_items_in_batch=num_items_in_batch,**spec)
        if not torch.isfinite(loss):raise ValueError('Nonfinite KD loss')
        if model.training:
            if num_items_in_batch is None:
                raise ValueError('Training requires the full accumulation token denominator')
            n=values['supervised_tokens'];meter['microbatches']+=1;meter['supervised_tokens']+=n
            meter['ce_sum']+=float(values['ce'])*n;meter['kl_sum']+=float(values['temperature_scaled_kl'])*n
            meter['teacher_discarded_probability_sum']+=float(values['discarded_teacher_probability'])*n
            if teacher is not None and any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
                raise ValueError('Teacher must remain frozen')
        output['loss']=loss
        return (loss,output) if return_outputs else loss
    trainer.compute_loss=compute_loss
    return meter
