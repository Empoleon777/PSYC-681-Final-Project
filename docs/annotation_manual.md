# Annotation Manual

Target batch size: `1,000-1,500` posts  
Annotators per post: `3`

## Ground Rules

1. Label only what appears in the text.
2. Use uncertain options when confidence is low.
3. Add evidence spans for ideology-relevant decisions.
4. Avoid inferring author intent from profile assumptions.

## Question Set

1. `q01_relevance` (`0/1`)
2. `q02_target`
3. `q03_stance`
4. `q04_frame`
5. `q05_moralization` (`0/1/2`)
6. `q06_identity_signaling` (`0/1/2`)
7. `q07_economic_direction` (`-1/0/1`)
8. `q08_social_direction` (`-1/0/1`)
9. `q09_intensity` (`0/1/2`)
10. `q10_ambiguity` (`0/1/2`)
11. `q11_confidence` (`0/1/2`)
12. `q12_notes`

Evidence field:
- `evidence_spans_json` with `start_char`, `end_char`, `text`, `type`

## Sampling Plan

Stratify by:
1. subreddit
2. topic
3. weak-label quartile (`Q1..Q4` from `q_score`)

Build packets with:
- `python scripts/make_annotation_batch.py`

## Quality Checks

1. Run a short calibration pass first.
2. Keep exactly 3 annotations per post.
3. Revisit posts with high disagreement entropy.