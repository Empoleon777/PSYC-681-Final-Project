# B6 Draft Notes

Current state of the model in [b6_ideology_model.py](/Users/jacobkim/PSYC-681-Final-Project/models/b6_ideology_model.py):

Implemented:

1. shared transformer encoder (`roberta-base`)
2. discourse heads: relevance, target, stance, frame
3. intermediate representations from target/stance/frame logits
4. projection from pooled encoder + intermediate reps into latent vector `z`
5. ideology heads: economic, social, intensity, ambiguity
6. orthogonality regularization helper for Level-B heads
7. two-stage training runner:
   - weak-label representation pretraining
   - gold fine-tuning with `hard`, `soft`, or `hybrid` supervision
8. disagreement-aware fine-tuning via annotator distributions from `soft_distribution_json`
   - ideology heads use an explicit KL soft-label objective
   - ambiguity-heavy subset metrics are logged during validation
9. training summary/checkpoint outputs for stage-wise comparisons

Training entry point:
- [train_b6_representation.py](/Users/jacobkim/PSYC-681-Final-Project/scripts/train_b6_representation.py)

Still open:

1. full experiment orchestration for all evaluation regimes (topic/community/platform shift)
2. richer calibration analysis (`ECE`, per-task calibration plots)
3. production-scale checkpoint selection and tracking
