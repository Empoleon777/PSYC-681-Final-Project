# Label Schema

Version: `v1.0`  
Updated: `2026-04-21`

This is the shared label contract for data prep, annotation, weak supervision, and model heads.

## Discourse Labels

`relevance`
- `0`: not ideology-relevant
- `1`: ideology-relevant

`target`
- `government_institutions`
- `political_party_or_actor`
- `policy_or_legislation`
- `social_group_or_identity`
- `foreign_actor_or_geopolitics`
- `media_or_information`
- `other`

`stance`
- `support`
- `oppose`
- `mixed_or_unclear`

`frame`
- `economic_freedom`
- `economic_redistribution`
- `law_and_order`
- `civil_rights`
- `national_security`
- `environmental_protection`
- `public_health`
- `identity_and_values`
- `other`

`moralization`
- `0`: low
- `1`: medium
- `2`: high

`identity_signaling`
- `0`: low
- `1`: medium
- `2`: high

`evidence_spans`
- JSON array of span records:
  - `start_char` (inclusive)
  - `end_char` (exclusive)
  - `text`
  - `type` in `{claim, evidence, moralized_phrase, identity_phrase, other}`

## Ideology Labels

`economic_direction`
- `-1`: left/pro-redistribution
- `0`: mixed/neutral
- `1`: right/pro-market

`social_direction`
- `-1`: liberal/pro-rights
- `0`: mixed/neutral
- `1`: traditional/order

`intensity`
- `0`: low
- `1`: medium
- `2`: high

`ambiguity`
- `0`: clear
- `1`: some ambiguity
- `2`: high ambiguity

## Internal Weak-Label Fields

- `zecon`: economic score in `[-1, 1]`
- `zsoc`: social score in `[-1, 1]`
- `cs`: source confidence in `[0, 1]`
- `rs`: source reliability in `[0, 1]`
- `q`: quality score in `[0, 1]`

## MITweet Alignment Notes

MITweet fields in this repo:
- domain relevance: `R1..R5`
- facet relevance: `R1-1-1..R12-5-3`
- ideology facets: `I1..I12`, values in `{0,1,2}` and `-1` for N/A

Current mapping strategy:
- relevance from domain/facet activation
- target/stance/frame from facet + lexical/context heuristics
- economic/social direction from facet-to-axis mapping
- ambiguity from disagreement and conflict signals

Data links:
- [data_sources.md](/Users/jacobkim/PSYC-681-Final-Project/docs/data_sources.md)
