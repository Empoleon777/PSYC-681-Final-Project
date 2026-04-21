#!/usr/bin/env python3
"""Build weak ideology labels from lightweight lexical/rule signals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:  # pragma: no cover
    psycopg2 = None
    Json = None
    execute_values = None


@dataclass
class RuleHit:
    post_id: str
    user_id: str
    rule_family: str
    fired: bool
    econ_score: float
    soc_score: float
    confidence: float
    context_fit: float
    target_guess: str
    frame_guess: str
    evidence: str
    trust_weight: float = 0.0
    trust_audit_json: str = ""


def bound(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def has_phrase(text: str, phrases: List[str]) -> Optional[str]:
    for phrase in phrases:
        if phrase in text:
            return phrase
    return None


def local_context_match(
    text: str, keyword: str, required_terms: List[str], window: int = 80
) -> Tuple[bool, float, str]:
    idx = text.find(keyword)
    if idx < 0:
        return False, 0.0, ""
    left = max(0, idx - window)
    right = min(len(text), idx + len(keyword) + window)
    local = text[left:right]
    trigger = has_phrase(local, required_terms)
    if not trigger:
        return False, 0.0, local
    hit_count = sum(1 for t in required_terms if t in local)
    score = bound(0.5 + 0.15 * hit_count, 0.5, 1.0)
    return True, score, local


def score_identity_signal(post: Dict[str, str], flair_rules: Dict) -> RuleHit:
    text = (post.get("text") or "").lower()
    flair = (post.get("flair") or "").lower()

    post_id = post["post_id"]
    user_id = post["user_id"]

    for rule in flair_rules["flair_rules"]:
        hit = has_phrase(flair, [p.lower() for p in rule["patterns"]])
        if hit:
            return RuleHit(
                post_id=post_id,
                user_id=user_id,
                rule_family="flair_or_self_decl",
                fired=True,
                econ_score=float(rule["zecon"]),
                soc_score=float(rule["zsoc"]),
                confidence=float(rule["confidence"]),
                context_fit=1.0,
                target_guess="",
                frame_guess="",
                evidence=f"flair:{hit}",
            )

    for rule in flair_rules["self_declaration_rules"]:
        hit = has_phrase(text, [p.lower() for p in rule["patterns"]])
        if hit:
            return RuleHit(
                post_id=post_id,
                user_id=user_id,
                rule_family="flair_or_self_decl",
                fired=True,
                econ_score=float(rule["zecon"]),
                soc_score=float(rule["zsoc"]),
                confidence=float(rule["confidence"]),
                context_fit=0.85,
                target_guess="",
                frame_guess="",
                evidence=f"self_decl:{hit}",
            )

    return RuleHit(
        post_id=post_id,
        user_id=user_id,
        rule_family="flair_or_self_decl",
        fired=False,
        econ_score=0.0,
        soc_score=0.0,
        confidence=0.0,
        context_fit=0.0,
        target_guess="",
        frame_guess="",
        evidence="",
    )


def score_target_frame_signal(
    post: Dict[str, str],
    target_rules: Dict,
    frame_rules: Dict,
    blocked_counts: Dict[str, int],
) -> RuleHit:
    text = (post.get("text") or "").lower()
    post_id = post["post_id"]
    user_id = post["user_id"]

    target_guess = ""
    frame_guess = ""
    context_fits: List[float] = []
    evidence_parts: List[str] = []
    econ = 0.0
    soc = 0.0

    for target in target_rules["targets"]:
        for key in [k.lower() for k in target["keywords"]]:
            if key not in text:
                continue
            ok, fit, _ = local_context_match(
                text, key, [w.lower() for w in target["required_context"]]
            )
            if ok:
                target_guess = target["label"]
                context_fits.append(fit)
                evidence_parts.append(f"target:{key}")
            else:
                blocked_counts["target_keyword_blocked"] += 1
                blocked_counts[f"target_blocked::{target['label']}"] += 1

    for frame in frame_rules["frames"]:
        for key in [k.lower() for k in frame["keywords"]]:
            if key not in text:
                continue
            ok, fit, _ = local_context_match(
                text, key, [w.lower() for w in frame["required_context"]]
            )
            if ok:
                frame_guess = frame["label"]
                econ = float(frame["zecon"])
                soc = float(frame["zsoc"])
                context_fits.append(fit)
                evidence_parts.append(f"frame:{key}")
            else:
                blocked_counts["frame_keyword_blocked"] += 1
                blocked_counts[f"frame_blocked::{frame['label']}"] += 1

    if target_guess and frame_guess:
        fit = mean(context_fits) if context_fits else 0.6
        return RuleHit(
            post_id=post_id,
            user_id=user_id,
            rule_family="target_frame_lexical",
            fired=True,
            econ_score=econ,
            soc_score=soc,
            confidence=bound(0.5 + 0.4 * fit, 0.5, 0.9),
            context_fit=fit,
            target_guess=target_guess,
            frame_guess=frame_guess,
            evidence=";".join(evidence_parts),
        )

    return RuleHit(
        post_id=post_id,
        user_id=user_id,
        rule_family="target_frame_lexical",
        fired=False,
        econ_score=0.0,
        soc_score=0.0,
        confidence=0.0,
        context_fit=0.0,
        target_guess=target_guess,
        frame_guess=frame_guess,
        evidence="",
    )


def load_rule_audit(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    values: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        family = (row.get("source_name") or "").strip()
        if not family:
            continue
        try:
            ok = int(row.get("is_correct", "0"))
        except ValueError:
            continue
        values[family].append(1 if ok else 0)
    return {k: mean(v) for k, v in values.items() if v}


def estimate_trust_weight(
    hit: RuleHit, peer_hits: List[RuleHit], historical: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    prior = {
        "flair_or_self_decl": 0.75,
        "target_frame_lexical": 0.65,
    }.get(hit.rule_family, 0.6)

    if not hit.fired:
        audit = {
            "source_prior": prior,
            "context_fit": 0.0,
            "agreement": 0.0,
            "historical_accuracy": historical.get(hit.rule_family, 0.7),
        }
        return 0.0, audit

    peers = [p for p in peer_hits if p.fired and p.rule_family != hit.rule_family]
    if not peers:
        agreement = 0.6
    else:
        sims = []
        for peer in peers:
            dist = abs(hit.econ_score - peer.econ_score) + abs(hit.soc_score - peer.soc_score)
            sims.append(bound(1.0 - dist / 4.0, 0.0, 1.0))
        agreement = mean(sims)

    hist = historical.get(hit.rule_family, 0.7)
    trust = prior * (0.4 + 0.6 * hit.context_fit) * (0.5 + 0.5 * agreement) * hist
    trust = bound(trust, 0.0, 1.0)

    audit = {
        "source_prior": prior,
        "context_fit": hit.context_fit,
        "agreement": agreement,
        "historical_accuracy": hist,
    }
    return trust, audit


def tri_state_entropy(score: float) -> float:
    logits = [-(abs(score + 1)), -(abs(score)), -(abs(score - 1))]
    m = max(logits)
    weights = [math.exp(v - m) for v in logits]
    z = sum(weights)
    probs = [w / z for w in weights]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    return ent / math.log(3)


def infer_stance(frame_guess: str) -> str:
    if frame_guess in {"economic_freedom", "law_and_order", "national_security"}:
        return "support_conservative"
    if frame_guess in {"economic_redistribution", "civil_rights", "environmental_protection"}:
        return "support_progressive"
    return "mixed_or_unclear"


def projection_offset(target_guess: str, frame_guess: str, stance_guess: str) -> Tuple[float, float]:
    target_delta = {
        "government_institutions": (0.0, 0.05),
        "political_party_or_actor": (0.05, 0.05),
        "policy_or_legislation": (0.1, 0.0),
        "social_group_or_identity": (0.0, -0.1),
        "foreign_actor_or_geopolitics": (0.05, 0.1),
    }.get(target_guess, (0.0, 0.0))

    frame_delta = {
        "economic_freedom": (0.12, 0.02),
        "economic_redistribution": (-0.12, -0.02),
        "law_and_order": (0.02, 0.12),
        "civil_rights": (-0.02, -0.12),
        "environmental_protection": (-0.08, -0.03),
    }.get(frame_guess, (0.0, 0.0))

    stance_scalar = {
        "support_progressive": -1.0,
        "support_conservative": 1.0,
        "mixed_or_unclear": 0.0,
    }[stance_guess]

    return (
        target_delta[0] + frame_delta[0] + 0.08 * stance_scalar,
        target_delta[1] + frame_delta[1] + 0.08 * stance_scalar,
    )


def upsert_user_priors(rows: List[Dict[str, object]], dsn: str) -> None:
    if psycopg2 is None or Json is None or execute_values is None:
        raise SystemExit("psycopg2-binary is required for --write-db.")
    if not rows:
        return

    values = [
        (
            row["user_id"],
            float(row["zecon"]),
            float(row["zsoc"]),
            float(row["confidence"]),
            row["method"],
            Json(json.loads(row["details"])),
        )
        for row in rows
    ]

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO user_priors
                    (user_id, zecon, zsoc, confidence, method, details)
                VALUES %s
                ON CONFLICT (user_id) DO UPDATE
                SET zecon = EXCLUDED.zecon,
                    zsoc = EXCLUDED.zsoc,
                    confidence = EXCLUDED.confidence,
                    method = EXCLUDED.method,
                    details = EXCLUDED.details,
                    updated_at = NOW()
                """,
                values,
                page_size=1000,
            )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts.csv"))
    parser.add_argument("--flair-lexicon", type=Path, default=Path("pipeline/lexicons/flair_priors.json"))
    parser.add_argument("--target-lexicon", type=Path, default=Path("pipeline/lexicons/target_seeds.json"))
    parser.add_argument("--frame-lexicon", type=Path, default=Path("pipeline/lexicons/frame_seeds.json"))
    parser.add_argument("--historical-accuracy-json", type=Path, default=Path("outputs/weak_labels/source_accuracy.json"))
    parser.add_argument("--rule-audit-csv", type=Path, default=Path("outputs/weak_labels/rule_audit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/weak_labels"))
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--database-url", type=str, default=os.environ.get("DATABASE_URL", ""))
    args = parser.parse_args()

    posts = read_csv(args.raw_posts_csv)
    flair_rules = load_json(args.flair_lexicon)
    target_rules = load_json(args.target_lexicon)
    frame_rules = load_json(args.frame_lexicon)

    if args.rule_audit_csv.exists():
        historical = load_rule_audit(args.rule_audit_csv)
    elif args.historical_accuracy_json.exists():
        historical = load_json(args.historical_accuracy_json)
    else:
        historical = {}

    blocked_counts: Dict[str, int] = defaultdict(int)
    hit_rows: List[Dict[str, object]] = []
    post_rows: List[Dict[str, object]] = []
    user_buckets: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    for post in posts:
        flair_hit = score_identity_signal(post, flair_rules)
        tf_hit = score_target_frame_signal(post, target_rules, frame_rules, blocked_counts)
        hits = [flair_hit, tf_hit]

        for hit in hits:
            trust, audit = estimate_trust_weight(hit, hits, historical)
            hit.trust_weight = trust
            hit.trust_audit_json = json.dumps(audit, ensure_ascii=True)

            hit_rows.append(
                {
                    "post_id": hit.post_id,
                    "user_id": hit.user_id,
                    "source_name": hit.rule_family,
                    "fired": int(hit.fired),
                    "zecon": round(hit.econ_score, 6),
                    "zsoc": round(hit.soc_score, 6),
                    "cs": round(hit.confidence, 6),
                    "context_score": round(hit.context_fit, 6),
                    "rs": round(hit.trust_weight, 6),
                    "target_label": hit.target_guess,
                    "frame_label": hit.frame_guess,
                    "evidence": hit.evidence,
                    "reliability_audit": hit.trust_audit_json,
                }
            )

        usable = [h for h in hits if h.fired and h.trust_weight > 0]
        if usable:
            weights = [h.trust_weight * h.confidence for h in usable]
            denom = max(sum(weights), 1e-8)
            econ = sum(w * h.econ_score for w, h in zip(weights, usable)) / denom
            soc = sum(w * h.soc_score for w, h in zip(weights, usable)) / denom
            avg_trust = mean(h.trust_weight for h in usable)
        else:
            econ, soc, avg_trust = 0.0, 0.0, 0.0

        econ_ent = tri_state_entropy(econ)
        soc_ent = tri_state_entropy(soc)
        q = bound(avg_trust * (1.0 - 0.5 * (econ_ent + soc_ent)), 0.0, 1.0)

        best_tf = max(
            [h for h in usable if h.rule_family == "target_frame_lexical"],
            key=lambda h: h.confidence,
            default=None,
        )
        target_guess = best_tf.target_guess if best_tf else "other"
        frame_guess = best_tf.frame_guess if best_tf else "other"
        stance_guess = infer_stance(frame_guess)
        econ_delta, soc_delta = projection_offset(target_guess, frame_guess, stance_guess)

        post_rows.append(
            {
                "post_id": post["post_id"],
                "user_id": post["user_id"],
                "subreddit": post.get("subreddit", ""),
                "zecon_agg": round(econ, 6),
                "zsoc_agg": round(soc, 6),
                "econ_entropy": round(econ_ent, 6),
                "soc_entropy": round(soc_ent, 6),
                "q_score": round(q, 6),
                "low_quality": int(q < 0.35),
                "target_hypothesis": target_guess,
                "stance_hypothesis": stance_guess,
                "frame_hypothesis": frame_guess,
                "projected_zecon": round(bound(econ + econ_delta, -1.0, 1.0), 6),
                "projected_zsoc": round(bound(soc + soc_delta, -1.0, 1.0), 6),
            }
        )

        if flair_hit.fired:
            user_buckets[post["user_id"]].append(
                (flair_hit.econ_score, flair_hit.soc_score, flair_hit.confidence)
            )

    user_prior_rows: List[Dict[str, object]] = []
    for user_id, vals in user_buckets.items():
        user_prior_rows.append(
            {
                "user_id": user_id,
                "zecon": round(mean(v[0] for v in vals), 6),
                "zsoc": round(mean(v[1] for v in vals), 6),
                "confidence": round(mean(v[2] for v in vals), 6),
                "method": "flair_or_self_decl",
                "details": json.dumps({"n_posts": len(vals)}, ensure_ascii=True),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "source_events.csv", hit_rows)
    write_csv(args.output_dir / "post_weak_labels.csv", post_rows)
    write_csv(args.output_dir / "user_priors.csv", user_prior_rows)
    (args.output_dir / "source_accuracy.json").write_text(
        json.dumps(historical, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (args.output_dir / "false_positive_report.json").write_text(
        json.dumps(dict(blocked_counts), indent=2, ensure_ascii=True), encoding="utf-8"
    )

    if args.write_db:
        if not args.database_url:
            raise SystemExit("DATABASE_URL is required when --write-db is set.")
        upsert_user_priors(user_prior_rows, args.database_url)

    print(f"Processed posts: {len(posts)}")
    print(f"Source events: {len(hit_rows)}")
    print(f"Aggregated weak labels: {len(post_rows)}")
    print(f"User priors: {len(user_prior_rows)}")


if __name__ == "__main__":
    main()
