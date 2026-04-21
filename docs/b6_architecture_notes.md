# B6 Draft Notes

Current state of the draft model in [b6_hierarchy_draft.py](/Users/jacobkim/PSYC-681-Final-Project/Models/b6_hierarchy_draft.py):

Implemented:

1. shared transformer encoder (`roberta-base`)
2. discourse heads: relevance, target, stance, frame
3. intermediate representations from target/stance/frame logits
4. projection from pooled encoder + intermediate reps into latent vector `z`
5. ideology heads: economic, social, intensity, ambiguity
6. orthogonality regularization helper for Level-B heads

Still open:

1. full multi-task loss wiring
2. soft-label/disagreement loss
3. training loop integration
4. full evaluation and checkpointing