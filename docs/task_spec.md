# Task Specification

Last updated: `2026-05-04`

This page is the working reference for task labels used across data prep, annotation, and modeling.

Core references:
- [label_schema.md](/Users/jacobkim/PSYC-681-Final-Project/docs/label_schema.md)
- [mitweet_mapping.md](/Users/jacobkim/PSYC-681-Final-Project/docs/mitweet_mapping.md)

## Level A Labels

- `relevance`: `{0,1}`
- `target`: `government_institutions | political_party_or_actor | policy_or_legislation | social_group_or_identity | foreign_actor_or_geopolitics | media_or_information | other`
- `stance`: `support | oppose | mixed_or_unclear`
- `frame`: `economic_freedom | economic_redistribution | law_and_order | civil_rights | national_security | environmental_protection | public_health | identity_and_values | other`
- `moralization`: `{0,1,2}`
- `identity_signaling`: `{0,1,2}`
- `evidence_spans`: token/character spans with type tags

## Level B Labels

- `economic_direction`: `{-1,0,1}`
- `social_direction`: `{-1,0,1}`
- `intensity`: `{0,1,2}`
- `ambiguity`: `{0,1,2}`

## MITweet Reference

- Mapping notes: [mitweet_mapping.md](/Users/jacobkim/PSYC-681-Final-Project/docs/mitweet_mapping.md)
- Scripts:
  - [RoBERTa-base-mitweet.py](/Users/jacobkim/PSYC-681-Final-Project/models/RoBERTa-base-mitweet.py)
  - [tfidf-Task20.py](/Users/jacobkim/PSYC-681-Final-Project/models/tfidf-Task20.py)

## Team Sign-off

- Jacob: signed `2026-05-04`
- Billy: signed `2026-05-04`
