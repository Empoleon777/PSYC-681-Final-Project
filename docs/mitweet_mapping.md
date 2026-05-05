# MITweet Mapping

Last updated: `2026-05-04`

This document explains how MITweet columns are translated into this project's label set.

## Source Columns

- Text: `tweet`
- Domain relevance: `R1..R5` (binary)
- Facet relevance: `R1-1-1 .. R12-5-3` (binary)
- Ideology facets: `I1..I12` with values in `{0,1,2}` and `-1` for N/A

## Mapping Rules

1. `relevance`
- `1` if any domain/facet relevance field is active, else `0`.

2. `target`
- Derived from the dominant active facet family:
  - governance/political process facets -> `government_institutions` or `political_party_or_actor`
  - policy facets -> `policy_or_legislation`
  - identity/culture facets -> `social_group_or_identity`
  - otherwise -> `other`

3. `frame`
- Facet-level heuristic map:
  - market/tax/regulation -> `economic_freedom`
  - welfare/labor/redistribution -> `economic_redistribution`
  - policing/crime/border -> `law_and_order`
  - rights/liberties/equality -> `civil_rights`
  - climate/environment -> `environmental_protection`
  - fallback -> `other`

4. `stance`
- Derived from per-facet ideology polarity where available.
- If mixed active facets conflict, use `mixed_or_unclear`.

5. `economic_direction` and `social_direction`
- Computed from mapped ideology facets and normalized to `{-1,0,1}`.

6. `intensity`
- Computed from confidence and ideological separation magnitude.

7. `ambiguity`
- Higher when active facets are sparse or conflict with each other.

## Splits

- Recommended split policy: fixed train/val/test by row-index hash.
- Existing scripts:
  - [RoBERTa-base-mitweet.py](/Users/jacobkim/PSYC-681-Final-Project/models/RoBERTa-base-mitweet.py)
  - [tfidf-Task20.py](/Users/jacobkim/PSYC-681-Final-Project/models/tfidf-Task20.py)
