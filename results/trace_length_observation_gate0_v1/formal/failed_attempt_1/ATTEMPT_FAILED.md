# Phase-0 formal training attempt 1

Status: failed preflight and cancelled on 2026-09-04.

The immutable 881-question data build completed, but answer-only training found
correct parent traces ending in `####` or `\\boxed{}` that were not recognized
by the initial `Answer:`-only character-span mask. Jobs 278952, 278954, and
278955 and dependent jobs 278956 and 278957 were cancelled. Eight adapters had
completed before cancellation. They are retained only as failed-attempt
evidence and must not enter Phase-0 analysis.

The corrective action is to support all verifier-recognized final-answer
formats, audit maskability for every selected trace during data construction,
bind the masking source hash into every run marker, rebuild the formal data
manifest, and rerun all 108 adapters in a clean attempt.
