# B6 Draft Notes

Current state of the model in [b6_hierarchy.py](models/b6_hierarchy.py):

What is already implemented:

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
10. consistency regularization toggle for weak pretraining (`--consistency-weight`)
11. confidence-based curriculum toggle for weak pretraining (`--enable-curriculum`)
12. calibration + abstention analysis (`calibration_summary.json`)
13. continuous signal/vector-field analysis script:
   - [analyze_b6_signal_field.py](scripts/analyze_b6_signal_field.py)

Training entry point:
- [train_b6_representation.py](scripts/train_b6_representation.py)

What is still open:

1. full experiment orchestration for all evaluation regimes (topic/community/platform shift)
2. production-scale checkpoint selection and tracking
