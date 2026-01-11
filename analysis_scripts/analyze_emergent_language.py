"""
Lightweight pipeline to analyze emergent language patterns in TinyVille logs.

Features:
- Walks over all resource_exchange JSON logs under a directory.
- Extracts per-message structured data to a CSV for quantitative analysis.
- Aggregates unknown tokens (not in per-game vocabulary).
- Aggregates tail n-grams (last 2–3 tokens) as candidate "code words".
- Exports small qualitative summaries (top patterns with example contexts).

Usage:
    python analyze_emergent_language.py \
        --logs_dir examples/logs \
        --output_dir results/emergent_language

This keeps dependencies to the standard library only.
"""

import argparse
import csv
import json
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PUNCT = set(string.punctuation)


@dataclass
class MessageRecord:
    game_id: str
    log_path: str
    round_index: int
    sender: str
    receiver: str
    blocked: bool
    block_reason: str
    content_raw: str
    tokens: List[str]
    norm_tokens: List[str]
    unknown_tokens: List[str]
    tail_bigram: str
    tail_trigram: str
    has_successful_trade: bool
    # Enhanced trade quality metrics
    num_successful_exchanges: int  # Number of successful exchanges in this round
    num_failed_exchanges: int  # Number of failed exchanges in this round
    total_value_transferred: int  # Total value transferred (sum of receiver_deltas)
    is_reciprocal_trade: bool  # Whether there are bidirectional exchanges
    # Reasoning fields
    rating_reason: str  # Agent's reasoning when rating partner (from same round)
    rating_message: str  # Message sent during rating (from same round)
    partner_rating: Optional[int]  # Rating given to partner in this round
    # Ground truth
    is_teammate_gt: Optional[bool]  # Ground truth: whether partner is actually a teammate


def normalize_token(tok: Any) -> str:
    """Normalize tokens: stringify, strip leading/trailing punctuation, lowercase."""
    if tok is None:
        return ""
    if not isinstance(tok, str):
        tok = str(tok)

    s = 0
    e = len(tok)
    while s < e and tok[s] in PUNCT:
        s += 1
    while e > s and tok[e - 1] in PUNCT:
        e -= 1
    core = tok[s:e]
    return core.lower()


def tokenize_content(content: Optional[str]) -> List[str]:
    if content is None:
        return []
    return [t for t in content.split() if t]


def tail_ngram(tokens: Sequence[str], n: int) -> str:
    if len(tokens) < n or n <= 0:
        return ""
    return " ".join(tokens[-n:])


def detect_successful_trade(round_data: Dict[str, Any]) -> bool:
    """Heuristic: any exchange entry without error and with non-zero payoff."""
    exchanges = round_data.get("exchange", []) or []
    for ex in exchanges:
        if ex.get("error"):
            continue
        deltas = ex.get("deltas") or {}
        if any(deltas.get(k, 0) != 0 for k in ("giver_delta", "receiver_delta")):
            return True
    return False


def analyze_trade_quality(round_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced trade quality analysis with multiple metrics.
    
    Returns:
        Dictionary with:
        - has_successful_trade: bool (backward compatible)
        - num_successful: int
        - num_failed: int
        - total_value_transferred: int (sum of receiver_deltas)
        - is_reciprocal: bool (whether there are bidirectional exchanges)
    """
    exchanges = round_data.get("exchange", []) or []
    
    # Separate successful and failed exchanges
    successful = []
    failed = []
    
    for ex in exchanges:
        if ex.get("error"):
            failed.append(ex)
        else:
            deltas = ex.get("deltas") or {}
            if any(deltas.get(k, 0) != 0 for k in ("giver_delta", "receiver_delta")):
                successful.append(ex)
            else:
                failed.append(ex)
    
    # Calculate total value transferred (sum of all receiver_deltas)
    total_value = sum(
        abs(ex.get("deltas", {}).get("receiver_delta", 0))
        for ex in successful
    )
    
    # Check for reciprocal trades (bidirectional exchanges)
    # Group exchanges by pair (sorted to handle A->B and B->A)
    pairs: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for ex in successful:
        giver = ex.get("giver", "")
        receiver = ex.get("receiver", "")
        if giver and receiver:
            # Use sorted tuple to handle both directions
            pair_key = tuple(sorted([giver, receiver]))
            pairs[pair_key].append(ex)
    
    # Check if any pair has exchanges in both directions
    is_reciprocal = False
    for pair_exchanges in pairs.values():
        # Get unique giver-receiver pairs
        directions = set()
        for ex in pair_exchanges:
            directions.add((ex.get("giver"), ex.get("receiver")))
        # If we have both A->B and B->A, it's reciprocal
        if len(directions) >= 2:
            # Check if there are exchanges in both directions
            givers = {ex.get("giver") for ex in pair_exchanges}
            receivers = {ex.get("receiver") for ex in pair_exchanges}
            if len(givers) >= 2 and len(receivers) >= 2:
                # More sophisticated check: see if A->B and B->A both exist
                for ex1 in pair_exchanges:
                    for ex2 in pair_exchanges:
                        if (ex1.get("giver") == ex2.get("receiver") and
                            ex1.get("receiver") == ex2.get("giver")):
                            is_reciprocal = True
                            break
                    if is_reciprocal:
                        break
                if is_reciprocal:
                    break
    
    return {
        "has_successful_trade": len(successful) > 0,
        "num_successful": len(successful),
        "num_failed": len(failed),
        "total_value_transferred": total_value,
        "is_reciprocal": is_reciprocal,
    }


def iter_logs(logs_dir: Path) -> Iterable[Path]:
    for path in logs_dir.rglob("*.json"):
        # Skip non-resource_exchange logs if present
        if "resource_exchange" not in path.name:
            continue
        yield path


def parse_log(path: Path) -> Tuple[List[MessageRecord], Dict[str, Any]]:
    data = json.loads(path.read_text())
    game_id = path.stem
    vocab_dict: Dict[str, Any] = data.get("vocabulary", {}) or {}
    vocab_tokens = set(vocab_dict.keys())

    records: List[MessageRecord] = []

    rounds = data.get("rounds", []) or []
    for r in rounds:
        r_idx = int(r.get("round", -1))
        
        # Enhanced trade quality analysis
        trade_quality = analyze_trade_quality(r)
        has_success = trade_quality["has_successful_trade"]
        
        chat = r.get("chat", []) or []
        
        # Build a map: sender -> rating info for this round
        rating_info: Dict[str, Dict[str, Any]] = {}
        rate_entries = r.get("rate", []) or []
        for rate_entry in rate_entries:
            player = rate_entry.get("player", "")
            rating_info[player] = {
                "rating_reason": rate_entry.get("rating_reason", ""),
                "rating_message": rate_entry.get("rating_message", ""),
                "rating": rate_entry.get("rating"),
            }
        
        # Build ground truth map: (player, partner) -> is_teammate
        gt_map: Dict[Tuple[str, str], bool] = {}
        feedback_entries = r.get("feedback", []) or []
        for feedback_entry in feedback_entries:
            player = feedback_entry.get("player", "")
            partner = feedback_entry.get("partner", "")
            is_teammate = feedback_entry.get("is_teammate", False)
            if player and partner:
                gt_map[(player, partner)] = is_teammate

        for m in chat:
            sender = m.get("sender", "")
            receiver = m.get("receiver", "")
            blocked = bool(m.get("blocked", False))
            block_reason = m.get("reason") or ""
            content = m.get("content")
            raw_content = m.get("raw_content") or m.get("raw")

            if isinstance(content, str):
                content_str = content
            elif raw_content is not None:
                content_str = str(raw_content)
            else:
                content_str = ""

            tokens = tokenize_content(content_str)
            norm_tokens = [normalize_token(t) for t in tokens if normalize_token(t)]

            unknown = [t for t in norm_tokens if t and t not in vocab_tokens]

            tb = tail_ngram(norm_tokens, 2)
            tt = tail_ngram(norm_tokens, 3)
            
            # Get reasoning info for this sender in this round
            sender_rating = rating_info.get(sender, {})
            rating_reason = sender_rating.get("rating_reason", "")
            rating_message = sender_rating.get("rating_message", "")
            partner_rating = sender_rating.get("rating")
            
            # Get ground truth
            is_teammate_gt = gt_map.get((sender, receiver))

            rec = MessageRecord(
                game_id=game_id,
                log_path=str(path),
                round_index=r_idx,
                sender=sender,
                receiver=receiver,
                blocked=blocked,
                block_reason=block_reason,
                content_raw=content_str,
                tokens=tokens,
                norm_tokens=norm_tokens,
                unknown_tokens=unknown,
                tail_bigram=tb,
                tail_trigram=tt,
                has_successful_trade=has_success,
                num_successful_exchanges=trade_quality["num_successful"],
                num_failed_exchanges=trade_quality["num_failed"],
                total_value_transferred=trade_quality["total_value_transferred"],
                is_reciprocal_trade=trade_quality["is_reciprocal"],
                rating_reason=rating_reason,
                rating_message=rating_message,
                partner_rating=partner_rating,
                is_teammate_gt=is_teammate_gt,
            )
            records.append(rec)

    meta = {
        "game_id": game_id,
        "vocab_size": len(vocab_tokens),
        "num_rounds": len(rounds),
        "log_path": str(path),
    }
    return records, meta


def write_messages_csv(out_path: Path, records: List[MessageRecord]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "log_path",
                "round",
                "sender",
                "receiver",
                "blocked",
                "block_reason",
                "has_successful_trade",
                "num_successful_exchanges",
                "num_failed_exchanges",
                "total_value_transferred",
                "is_reciprocal_trade",
                "content_raw",
                "tokens",
                "norm_tokens",
                "unknown_tokens",
                "tail_bigram",
                "tail_trigram",
                "rating_reason",
                "rating_message",
                "partner_rating",
                "is_teammate_gt",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.game_id,
                    r.log_path,
                    r.round_index,
                    r.sender,
                    r.receiver,
                    int(r.blocked),
                    r.block_reason,
                    int(r.has_successful_trade),
                    r.num_successful_exchanges,
                    r.num_failed_exchanges,
                    r.total_value_transferred,
                    int(r.is_reciprocal_trade),
                    r.content_raw,
                    " ".join(r.tokens),
                    " ".join(r.norm_tokens),
                    " ".join(r.unknown_tokens),
                    r.tail_bigram,
                    r.tail_trigram,
                    r.rating_reason,
                    r.rating_message,
                    r.partner_rating if r.partner_rating is not None else "",
                    int(r.is_teammate_gt) if r.is_teammate_gt is not None else "",
                ]
            )


def aggregate_unknown_tokens(records: List[MessageRecord]) -> List[Dict[str, Any]]:
    token_counts: Counter[str] = Counter()
    token_games: defaultdict[str, set] = defaultdict(set)
    token_ratings: defaultdict[str, List[int]] = defaultdict(list)

    for r in records:
        for t in r.unknown_tokens:
            token_counts[t] += 1
            token_games[t].add(r.game_id)
            if r.partner_rating is not None:
                token_ratings[t].append(r.partner_rating)

    rows: List[Dict[str, Any]] = []
    for tok, cnt in token_counts.most_common():
        ratings = token_ratings[tok]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        rows.append(
            {
                "token": tok,
                "count": cnt,
                "game_count": len(token_games[tok]),
                "avg_rating": round(avg_rating, 2) if avg_rating is not None else "",
                "rating_count": len(ratings),
            }
        )
    return rows


def aggregate_tail_ngrams(records: List[MessageRecord]) -> List[Dict[str, Any]]:
    bigram_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    bigram_games: defaultdict[str, set] = defaultdict(set)
    trigram_games: defaultdict[str, set] = defaultdict(set)
    bigram_ratings: defaultdict[str, List[int]] = defaultdict(list)
    trigram_ratings: defaultdict[str, List[int]] = defaultdict(list)

    for r in records:
        if r.tail_bigram:
            bigram_counts[r.tail_bigram] += 1
            bigram_games[r.tail_bigram].add(r.game_id)
            if r.partner_rating is not None:
                bigram_ratings[r.tail_bigram].append(r.partner_rating)
        if r.tail_trigram:
            trigram_counts[r.tail_trigram] += 1
            trigram_games[r.tail_trigram].add(r.game_id)
            if r.partner_rating is not None:
                trigram_ratings[r.tail_trigram].append(r.partner_rating)

    rows: List[Dict[str, Any]] = []
    for ngram, cnt in bigram_counts.most_common():
        ratings = bigram_ratings[ngram]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        rows.append(
            {
                "ngram": ngram,
                "n": 2,
                "count": cnt,
                "game_count": len(bigram_games[ngram]),
                "avg_rating": round(avg_rating, 2) if avg_rating is not None else "",
                "rating_count": len(ratings),
            }
        )
    for ngram, cnt in trigram_counts.most_common():
        ratings = trigram_ratings[ngram]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        rows.append(
            {
                "ngram": ngram,
                "n": 3,
                "count": cnt,
                "game_count": len(trigram_games[ngram]),
                "avg_rating": round(avg_rating, 2) if avg_rating is not None else "",
                "rating_count": len(ratings),
            }
        )
    return rows


def write_simple_csv(out_path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_high_ratings(records: List[MessageRecord], out_path: Path) -> None:
    """Analyze what patterns and reasoning lead to high ratings (4 = definitely teammate)."""
    high_rating_records = [r for r in records if r.partner_rating == 4]
    
    if not high_rating_records:
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# High Rating (4) Analysis\n")
            f.write("# No records with rating=4 found.\n")
        return
    
    # Analyze language patterns in high rating cases
    high_unknown_tokens: Counter[str] = Counter()
    high_tail_bigrams: Counter[str] = Counter()
    high_tail_trigrams: Counter[str] = Counter()
    high_rating_messages: Counter[str] = Counter()
    
    # Collect reasoning patterns
    reasoning_keywords: Counter[str] = Counter()
    all_reasonings: List[str] = []
    
    for r in high_rating_records:
        # Language patterns
        for t in r.unknown_tokens:
            high_unknown_tokens[t] += 1
        if r.tail_bigram:
            high_tail_bigrams[r.tail_bigram] += 1
        if r.tail_trigram:
            high_tail_trigrams[r.tail_trigram] += 1
        if r.rating_message:
            # Tokenize rating message to find code words
            rating_tokens = tokenize_content(r.rating_message)
            for t in rating_tokens:
                high_rating_messages[normalize_token(t)] += 1
        
        # Reasoning analysis
        if r.rating_reason:
            all_reasonings.append(r.rating_reason)
            # Extract common keywords from reasoning
            reason_lower = r.rating_reason.lower()
            # Look for common patterns
            keywords = [
                "teammate", "cooperative", "clear", "consistent", "pattern", "code",
                "signal", "recognize", "same", "similar", "match", "confirm", "guaranteed",
                "round 1", "round 2", "first", "second", "rules", "game rules",
                "helpful", "balanced", "mutual", "beneficial", "trade", "exchange",
                "offered", "requested", "gave", "received", "water", "grain", "meat",
                "fruit", "fish", "resource"
            ]
            for kw in keywords:
                if kw in reason_lower:
                    reasoning_keywords[kw] += 1
    
    # Write analysis
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# High Rating (4) Analysis\n")
        f.write("# This file analyzes what patterns and reasoning lead agents to rate partners as 4 (definitely teammate)\n\n")
        f.write(f"Total records with rating=4: {len(high_rating_records)}\n\n")
        
        # Top unknown tokens in high rating cases
        f.write("## Top Unknown Tokens in High Rating Cases\n")
        f.write("(Tokens not in vocabulary that appear when rating=4)\n\n")
        if high_unknown_tokens:
            for tok, cnt in high_unknown_tokens.most_common(20):
                f.write(f"  {tok}: {cnt} occurrences\n")
        else:
            f.write("  (none)\n")
        f.write("\n")
        
        # Top tail patterns in high rating cases
        f.write("## Top Tail Patterns in High Rating Cases\n")
        f.write("(Last 2-3 tokens of messages when rating=4)\n\n")
        f.write("Bigrams:\n")
        if high_tail_bigrams:
            for ngram, cnt in high_tail_bigrams.most_common(15):
                f.write(f"  {ngram}: {cnt} occurrences\n")
        else:
            f.write("  (none)\n")
        f.write("\nTrigrams:\n")
        if high_tail_trigrams:
            for ngram, cnt in high_tail_trigrams.most_common(15):
                f.write(f"  {ngram}: {cnt} occurrences\n")
        else:
            f.write("  (none)\n")
        f.write("\n")
        
        # Rating message patterns
        f.write("## Top Tokens in Rating Messages (High Rating Cases)\n")
        f.write("(Tokens used in rating_message when rating=4)\n\n")
        if high_rating_messages:
            for tok, cnt in high_rating_messages.most_common(20):
                f.write(f"  {tok}: {cnt} occurrences\n")
        else:
            f.write("  (none)\n")
        f.write("\n")
        
        # Reasoning keywords
        f.write("## Common Keywords in Reasoning (High Rating Cases)\n")
        f.write("(What agents mention when giving rating=4)\n\n")
        if reasoning_keywords:
            for kw, cnt in reasoning_keywords.most_common(30):
                f.write(f"  {kw}: {cnt} mentions\n")
        else:
            f.write("  (none)\n")
        f.write("\n")
        
        # Example cases with full reasoning
        f.write("## Example Cases: Full Reasoning for Rating=4\n")
        f.write("(Sample of actual reasoning text when agents rated 4)\n\n")
        # Show diverse examples
        seen_reasonings = set()
        example_count = 0
        for r in high_rating_records:
            if r.rating_reason and r.rating_reason not in seen_reasonings and example_count < 30:
                seen_reasonings.add(r.rating_reason)
                f.write(f"--- Example {example_count + 1} ---\n")
                f.write(f"Game: {r.game_id}, Round: {r.round_index}\n")
                f.write(f"Sender: {r.sender} -> Receiver: {r.receiver}\n")
                if r.content_raw:
                    f.write(f"Message: {r.content_raw}\n")
                f.write(f"Reasoning: {r.rating_reason}\n")
                if r.rating_message:
                    f.write(f"Rating message: {r.rating_message}\n")
                if r.tail_bigram or r.tail_trigram:
                    f.write(f"Tail pattern: {r.tail_bigram or r.tail_trigram}\n")
                if r.unknown_tokens:
                    f.write(f"Unknown tokens: {' '.join(r.unknown_tokens)}\n")
                f.write("\n")
                example_count += 1
                if example_count >= 30:
                    break


def analyze_ratings(records: List[MessageRecord]) -> Dict[str, Any]:
    """Analyze rating distribution and correlations with enhanced trade metrics."""
    rating_counts: Counter[int] = Counter()
    rating_by_success: Dict[bool, List[int]] = {True: [], False: []}
    rating_by_round: Dict[int, List[int]] = defaultdict(list)
    rating_by_reciprocal: Dict[bool, List[int]] = {True: [], False: []}
    rating_by_value: Dict[int, List[int]] = defaultdict(list)  # Group by value ranges
    
    for r in records:
        if r.partner_rating is not None:
            rating_counts[r.partner_rating] += 1
            rating_by_success[r.has_successful_trade].append(r.partner_rating)
            rating_by_round[r.round_index].append(r.partner_rating)
            rating_by_reciprocal[r.is_reciprocal_trade].append(r.partner_rating)
            
            # Group by value ranges
            if r.total_value_transferred > 0:
                if r.total_value_transferred <= 2:
                    rating_by_value[1].append(r.partner_rating)  # Low value (1-2)
                elif r.total_value_transferred <= 4:
                    rating_by_value[2].append(r.partner_rating)  # Medium value (3-4)
                else:
                    rating_by_value[3].append(r.partner_rating)  # High value (5+)
    
    total_ratings = sum(rating_counts.values())
    rating_dist = {i: rating_counts.get(i, 0) for i in [1, 2, 3, 4]}
    rating_pct = {i: (rating_dist[i] / total_ratings * 100) if total_ratings > 0 else 0 
                  for i in [1, 2, 3, 4]}
    
    avg_rating_with_success = sum(rating_by_success[True]) / len(rating_by_success[True]) if rating_by_success[True] else None
    avg_rating_without_success = sum(rating_by_success[False]) / len(rating_by_success[False]) if rating_by_success[False] else None
    avg_rating_reciprocal = sum(rating_by_reciprocal[True]) / len(rating_by_reciprocal[True]) if rating_by_reciprocal[True] else None
    avg_rating_non_reciprocal = sum(rating_by_reciprocal[False]) / len(rating_by_reciprocal[False]) if rating_by_reciprocal[False] else None
    
    return {
        "total_ratings": total_ratings,
        "rating_distribution": rating_dist,
        "rating_percentages": rating_pct,
        "avg_rating_with_success": round(avg_rating_with_success, 2) if avg_rating_with_success else None,
        "avg_rating_without_success": round(avg_rating_without_success, 2) if avg_rating_without_success else None,
        "avg_rating_reciprocal": round(avg_rating_reciprocal, 2) if avg_rating_reciprocal else None,
        "avg_rating_non_reciprocal": round(avg_rating_non_reciprocal, 2) if avg_rating_non_reciprocal else None,
        "rating_by_round": {r: round(sum(ratings) / len(ratings), 2) 
                            for r, ratings in rating_by_round.items() if ratings},
        "rating_by_value": {v: round(sum(ratings) / len(ratings), 2) 
                            for v, ratings in rating_by_value.items() if ratings},
    }


def write_qualitative_summaries(
    out_dir: Path,
    records: List[MessageRecord],
    top_unknown: int = 30,
    top_ngrams: int = 30,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index records by token/ngram for quick lookup
    by_unknown: defaultdict[str, List[MessageRecord]] = defaultdict(list)
    by_ngram: defaultdict[str, List[MessageRecord]] = defaultdict(list)

    for r in records:
        for t in r.unknown_tokens:
            by_unknown[t].append(r)
        if r.tail_bigram:
            by_ngram[f"2::{r.tail_bigram}"].append(r)
        if r.tail_trigram:
            by_ngram[f"3::{r.tail_trigram}"].append(r)

    # Unknown tokens
    unk_rows = aggregate_unknown_tokens(records)
    unk_path = out_dir / "summary_unknown_tokens.txt"
    with unk_path.open("w", encoding="utf-8") as f:
        f.write("# Top unknown tokens (not in per-game vocabulary)\n")
        f.write("#\n")
        f.write("# Field explanations:\n")
        f.write("#   Reasoning: Agent's internal reasoning when rating their partner (in natural language)\n")
        f.write("#   Rating message: Message sent by agent during rating phase (may contain code words in game language)\n")
        f.write("#\n\n")
        for row in unk_rows[:top_unknown]:
            tok = row["token"]
            f.write(f"Token: {tok} | count={row['count']} | games={row['game_count']}\n")
            examples = by_unknown.get(tok, [])[:5]
            for ex in examples:
                f.write(
                    f"  - [{ex.game_id} R{ex.round_index} {ex.sender}->{ex.receiver}] "
                    f"{ex.content_raw}\n"
                )
                # Show reasoning if available
                if ex.rating_reason:
                    f.write(f"    Reasoning: {ex.rating_reason}\n")
                if ex.rating_message:
                    f.write(f"    Rating message: {ex.rating_message}\n")
                if ex.partner_rating is not None:
                    f.write(f"    Rating: {ex.partner_rating} (1=not teammate, 4=teammate)\n")
            f.write("\n")

    # Tail n-grams
    ngram_rows = aggregate_tail_ngrams(records)
    ngram_path = out_dir / "summary_tail_ngrams.txt"
    with ngram_path.open("w", encoding="utf-8") as f:
        f.write("# Top tail n-grams (last 2–3 tokens of messages)\n")
        f.write("#\n")
        f.write("# Field explanations:\n")
        f.write("#   Reasoning: Agent's internal reasoning when rating their partner (in natural language)\n")
        f.write("#   Rating message: Message sent by agent during rating phase (may contain code words in game language)\n")
        f.write("#\n\n")
        for row in ngram_rows[:top_ngrams]:
            ngram = row["ngram"]
            n = row["n"]
            key = f"{n}::{ngram}"
            f.write(
                f"n-gram (n={n}): {ngram} | count={row['count']} | games={row['game_count']}\n"
            )
            examples = by_ngram.get(key, [])[:5]
            for ex in examples:
                f.write(
                    f"  - [{ex.game_id} R{ex.round_index} {ex.sender}->{ex.receiver}] "
                    f"{ex.content_raw}\n"
                )
                # Show reasoning if available
                if ex.rating_reason:
                    f.write(f"    Reasoning: {ex.rating_reason}\n")
                if ex.rating_message:
                    f.write(f"    Rating message: {ex.rating_message}\n")
                if ex.partner_rating is not None:
                    f.write(f"    Rating: {ex.partner_rating} (1=not teammate, 4=teammate)\n")
            f.write("\n")
    
    # Rating analysis summary
    rating_stats = analyze_ratings(records)
    rating_path = out_dir / "summary_ratings.txt"
    with rating_path.open("w", encoding="utf-8") as f:
        f.write("# Rating Analysis\n")
        f.write("# Rating scale: 1 = definitely not teammate, 4 = definitely teammate\n\n")
        f.write(f"Total ratings: {rating_stats['total_ratings']}\n\n")
        f.write("Rating distribution:\n")
        for rating in [1, 2, 3, 4]:
            count = rating_stats['rating_distribution'][rating]
            pct = rating_stats['rating_percentages'][rating]
            f.write(f"  Rating {rating}: {count} ({pct:.1f}%)\n")
        f.write("\n")
        f.write("Average rating by trade success:\n")
        if rating_stats['avg_rating_with_success']:
            f.write(f"  With successful trade: {rating_stats['avg_rating_with_success']}\n")
        if rating_stats['avg_rating_without_success']:
            f.write(f"  Without successful trade: {rating_stats['avg_rating_without_success']}\n")
        f.write("\n")
        f.write("Average rating by reciprocity:\n")
        if rating_stats['avg_rating_reciprocal']:
            f.write(f"  Reciprocal trade: {rating_stats['avg_rating_reciprocal']}\n")
        if rating_stats['avg_rating_non_reciprocal']:
            f.write(f"  Non-reciprocal trade: {rating_stats['avg_rating_non_reciprocal']}\n")
        f.write("\n")
        f.write("Average rating by value transferred:\n")
        value_labels = {1: "Low (1-2)", 2: "Medium (3-4)", 3: "High (5+)"}
        for value_range in sorted(rating_stats['rating_by_value'].keys()):
            f.write(f"  {value_labels.get(value_range, f'Range {value_range}')}: {rating_stats['rating_by_value'][value_range]}\n")
        f.write("\n")
        f.write("Average rating by round:\n")
        for round_num in sorted(rating_stats['rating_by_round'].keys()):
            f.write(f"  Round {round_num}: {rating_stats['rating_by_round'][round_num]}\n")
    
    # High rating (4) analysis
    high_rating_path = out_dir / "summary_high_rating_analysis.txt"
    analyze_high_ratings(records, high_rating_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze emergent language patterns in TinyVille logs."
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="examples/logs",
        help="Root directory containing resource_exchange_*.json logs (default: examples/logs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/emergent_language",
        help="Directory to write CSVs and qualitative summaries.",
    )
    args = parser.parse_args()

    logs_root = Path(args.logs_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    if not logs_root.exists():
        raise SystemExit(f"Logs directory does not exist: {logs_root}")

    all_records: List[MessageRecord] = []
    all_meta: List[Dict[str, Any]] = []

    for log_path in iter_logs(logs_root):
        recs, meta = parse_log(log_path)
        all_records.extend(recs)
        all_meta.append(meta)

    if not all_records:
        raise SystemExit(f"No message records found under {logs_root}")

    # Messages CSV
    write_messages_csv(out_root / "messages.csv", all_records)

    # Unknown tokens CSV
    unk_rows = aggregate_unknown_tokens(all_records)
    write_simple_csv(
        out_root / "unknown_tokens.csv",
        unk_rows,
        fieldnames=["token", "count", "game_count", "avg_rating", "rating_count"],
    )

    # Tail n-grams CSV
    ngram_rows = aggregate_tail_ngrams(all_records)
    write_simple_csv(
        out_root / "tail_ngrams.csv",
        ngram_rows,
        fieldnames=["ngram", "n", "count", "game_count", "avg_rating", "rating_count"],
    )

    # Per-log metadata CSV (vocab size, etc.)
    write_simple_csv(
        out_root / "logs_meta.csv",
        all_meta,
        fieldnames=["game_id", "log_path", "vocab_size", "num_rounds"],
    )

    # Qualitative summaries
    write_qualitative_summaries(out_root, all_records)

    print(f"Wrote analysis outputs to: {out_root}")


if __name__ == "__main__":
    main()



