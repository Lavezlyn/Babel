"""
Language Evolution Metrics Analysis

Implements comprehensive metrics for analyzing language emergence and evolution:
- Pattern Life Cycle Metrics
- Pattern Diffusion Metrics
- Pattern Evolution Path Metrics
- Pattern Selection Metrics
- Lexical Evolution Metrics
- Structural Evolution Metrics
"""

import argparse
import csv
import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any


@dataclass
class PatternRecord:
    """Record for tracking pattern usage across rounds and agents."""
    pattern: str
    pattern_type: str  # 'token', 'bigram', 'trigram'
    first_round: int
    first_agent: str
    first_pair: Tuple[str, str]
    usage_by_round: Dict[int, int]  # round -> count
    usage_by_agent: Dict[str, int]  # agent -> count
    usage_by_pair: Dict[Tuple[str, str], int]  # (sender, receiver) -> count
    ratings: List[float]  # All ratings when this pattern was used
    rounds: Set[int]  # All rounds where pattern appeared


def load_messages_csv(csv_path: Path) -> List[Dict]:
    """Load messages CSV file."""
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def parse_pair(sender: str, receiver: str) -> Tuple[str, str]:
    """Normalize pair representation."""
    return tuple(sorted([sender, receiver]))


# ============================================================================
# A. Pattern Life Cycle Metrics
# ============================================================================

def calculate_lifecycle_metrics(records: List[Dict]) -> Dict[str, Any]:
    """Calculate pattern life cycle metrics."""
    # Track patterns
    token_patterns: Dict[str, PatternRecord] = {}
    bigram_patterns: Dict[str, PatternRecord] = {}
    trigram_patterns: Dict[str, PatternRecord] = {}
    
    for record in records:
        game_id = record.get('game_id', '')
        round_num = int(record.get('round', 0))
        sender = record.get('sender', '')
        receiver = record.get('receiver', '')
        rating_str = record.get('partner_rating', '').strip()
        rating = float(rating_str) if rating_str else None
        
        pair = parse_pair(sender, receiver)
        
        # Process tokens
        unknown_tokens_str = record.get('unknown_tokens', '').strip()
        if unknown_tokens_str:
            tokens = unknown_tokens_str.split()
            for token in tokens:
                if token not in token_patterns:
                    token_patterns[token] = PatternRecord(
                        pattern=token,
                        pattern_type='token',
                        first_round=round_num,
                        first_agent=sender,
                        first_pair=pair,
                        usage_by_round={},
                        usage_by_agent={},
                        usage_by_pair={},
                        ratings=[],
                        rounds=set()
                    )
                pat = token_patterns[token]
                pat.usage_by_round[round_num] = pat.usage_by_round.get(round_num, 0) + 1
                pat.usage_by_agent[sender] = pat.usage_by_agent.get(sender, 0) + 1
                pat.usage_by_pair[pair] = pat.usage_by_pair.get(pair, 0) + 1
                pat.rounds.add(round_num)
                if rating is not None:
                    pat.ratings.append(rating)
        
        # Process bigrams
        bigram = record.get('tail_bigram', '').strip()
        if bigram:
            if bigram not in bigram_patterns:
                bigram_patterns[bigram] = PatternRecord(
                    pattern=bigram,
                    pattern_type='bigram',
                    first_round=round_num,
                    first_agent=sender,
                    first_pair=pair,
                    usage_by_round={},
                    usage_by_agent={},
                    usage_by_pair={},
                    ratings=[],
                    rounds=set()
                )
            pat = bigram_patterns[bigram]
            pat.usage_by_round[round_num] = pat.usage_by_round.get(round_num, 0) + 1
            pat.usage_by_agent[sender] = pat.usage_by_agent.get(sender, 0) + 1
            pat.usage_by_pair[pair] = pat.usage_by_pair.get(pair, 0) + 1
            pat.rounds.add(round_num)
            if rating is not None:
                pat.ratings.append(rating)
        
        # Process trigrams
        trigram = record.get('tail_trigram', '').strip()
        if trigram:
            if trigram not in trigram_patterns:
                trigram_patterns[trigram] = PatternRecord(
                    pattern=trigram,
                    pattern_type='trigram',
                    first_round=round_num,
                    first_agent=sender,
                    first_pair=pair,
                    usage_by_round={},
                    usage_by_agent={},
                    usage_by_pair={},
                    ratings=[],
                    rounds=set()
                )
            pat = trigram_patterns[trigram]
            pat.usage_by_round[round_num] = pat.usage_by_round.get(round_num, 0) + 1
            pat.usage_by_agent[sender] = pat.usage_by_agent.get(sender, 0) + 1
            pat.usage_by_pair[pair] = pat.usage_by_pair.get(pair, 0) + 1
            pat.rounds.add(round_num)
            if rating is not None:
                pat.ratings.append(rating)
    
    all_patterns = {**token_patterns, **bigram_patterns, **trigram_patterns}
    
    # Calculate metrics
    metrics = {
        'first_appearance_distribution': calculate_first_appearance_distribution(all_patterns),
        'stabilization_times': calculate_stabilization_times(all_patterns),
        'survival_rates': calculate_survival_rates(all_patterns),
        'extinction_rates': calculate_extinction_rates(all_patterns),
        'pattern_records': all_patterns
    }
    
    return metrics


def calculate_first_appearance_distribution(patterns: Dict[str, PatternRecord]) -> Dict[str, int]:
    """Calculate distribution of first appearance rounds."""
    early = 0  # 1-3
    mid = 0    # 4-7
    late = 0   # 8-14
    
    for pat in patterns.values():
        r = pat.first_round
        if 1 <= r <= 3:
            early += 1
        elif 4 <= r <= 7:
            mid += 1
        elif 8 <= r <= 14:
            late += 1
    
    return {'early': early, 'mid': mid, 'late': late}


def calculate_stabilization_times(patterns: Dict[str, PatternRecord]) -> List[int]:
    """Calculate time from first appearance to stabilization (≥3 uses)."""
    stabilization_times = []
    
    for pat in patterns.values():
        total_uses = sum(pat.usage_by_round.values())
        if total_uses >= 3:
            # Find first round with ≥3 cumulative uses
            cumulative = 0
            for round_num in sorted(pat.rounds):
                cumulative += pat.usage_by_round[round_num]
                if cumulative >= 3:
                    stabilization_time = round_num - pat.first_round
                    stabilization_times.append(stabilization_time)
                    break
    
    return stabilization_times


def calculate_survival_rates(patterns: Dict[str, PatternRecord]) -> Dict[str, float]:
    """Calculate survival rates: early patterns still used in late rounds."""
    early_patterns = [p for p in patterns.values() if 1 <= p.first_round <= 3]
    if not early_patterns:
        return {'survival_rate': 0.0, 'total_early': 0, 'survived': 0}
    
    survived = sum(1 for p in early_patterns if any(r >= 8 for r in p.rounds))
    
    return {
        'survival_rate': survived / len(early_patterns) * 100,
        'total_early': len(early_patterns),
        'survived': survived
    }


def calculate_extinction_rates(patterns: Dict[str, PatternRecord]) -> Dict[str, float]:
    """Calculate extinction rates: early patterns that disappeared."""
    early_patterns = [p for p in patterns.values() if 1 <= p.first_round <= 3]
    if not early_patterns:
        return {'extinction_rate': 0.0, 'total_early': 0, 'extinct': 0}
    
    extinct = sum(1 for p in early_patterns if not any(r >= 8 for r in p.rounds))
    
    return {
        'extinction_rate': extinct / len(early_patterns) * 100,
        'total_early': len(early_patterns),
        'extinct': extinct
    }


# ============================================================================
# B. Pattern Diffusion Metrics
# ============================================================================

def calculate_diffusion_metrics(patterns: Dict[str, PatternRecord]) -> Dict[str, Any]:
    """Calculate pattern diffusion metrics."""
    cross_agent_times = []
    adoption_counts = []
    cross_pair_counts = []
    
    for pat in patterns.values():
        # Cross-agent diffusion time
        agents = list(pat.usage_by_agent.keys())
        if len(agents) > 1:
            # Find when second agent started using it
            agent_first_rounds = {}
            for round_num in sorted(pat.rounds):
                # Find which agent used it in this round
                # (We need to track this from original records, simplified here)
                pass
        
        # Adoption rate (number of different agents)
        adoption_counts.append(len(pat.usage_by_agent))
        
        # Cross-pair diffusion (number of different pairs)
        cross_pair_counts.append(len(pat.usage_by_pair))
    
    return {
        'avg_adoption_rate': sum(adoption_counts) / len(adoption_counts) if adoption_counts else 0,
        'avg_cross_pair_count': sum(cross_pair_counts) / len(cross_pair_counts) if cross_pair_counts else 0,
        'adoption_distribution': Counter(adoption_counts)
    }


# ============================================================================
# C. Pattern Evolution Path Metrics
# ============================================================================

def calculate_evolution_paths(patterns: Dict[str, PatternRecord]) -> Dict[str, Any]:
    """Calculate pattern evolution paths."""
    # Group patterns by type
    bigrams = {k: v for k, v in patterns.items() if v.pattern_type == 'bigram'}
    trigrams = {k: v for k, v in patterns.items() if v.pattern_type == 'trigram'}
    
    # Compositional evolution: 2-gram → 3-gram
    compositional_paths = []
    for trigram_pat in trigrams.values():
        trigram_tokens = trigram_pat.pattern.split()
        if len(trigram_tokens) >= 2:
            bigram_prefix = ' '.join(trigram_tokens[:2])
            if bigram_prefix in bigrams:
                bigram_pat = bigrams[bigram_prefix]
                if bigram_pat.first_round < trigram_pat.first_round:
                    compositional_paths.append({
                        'from': bigram_prefix,
                        'to': trigram_pat.pattern,
                        'from_round': bigram_pat.first_round,
                        'to_round': trigram_pat.first_round
                    })
    
    # Simplification evolution: 3-gram → 2-gram
    simplification_paths = []
    for bigram_pat in bigrams.values():
        bigram_tokens = bigram_pat.pattern.split()
        # Check if this bigram is a prefix of any trigram
        for trigram_pat in trigrams.values():
            trigram_tokens = trigram_pat.pattern.split()
            if len(trigram_tokens) >= 2 and ' '.join(trigram_tokens[:2]) == bigram_pat.pattern:
                if trigram_pat.first_round < bigram_pat.first_round:
                    simplification_paths.append({
                        'from': trigram_pat.pattern,
                        'to': bigram_pat.pattern,
                        'from_round': trigram_pat.first_round,
                        'to_round': bigram_pat.first_round
                    })
    
    # Variant evolution: same core token
    tokens = {k: v for k, v in patterns.items() if v.pattern_type == 'token'}
    variant_groups = defaultdict(list)
    for token_pat in tokens.values():
        # Group by first token (simplified)
        core = token_pat.pattern.split()[0] if ' ' in token_pat.pattern else token_pat.pattern
        variant_groups[core].append(token_pat.pattern)
    
    return {
        'compositional_paths': compositional_paths,
        'simplification_paths': simplification_paths,
        'variant_groups': dict(variant_groups)
    }


# ============================================================================
# D. Pattern Selection Metrics
# ============================================================================

def calculate_selection_metrics(patterns: Dict[str, PatternRecord]) -> Dict[str, Any]:
    """Calculate pattern selection metrics."""
    high_rating_patterns = []
    low_rating_patterns = []
    
    for pat in patterns.values():
        if pat.ratings:
            avg_rating = sum(pat.ratings) / len(pat.ratings)
            total_uses = sum(pat.usage_by_round.values())
            
            if avg_rating >= 3.5:
                high_rating_patterns.append((pat.pattern, total_uses, avg_rating))
            elif avg_rating <= 2.0:
                low_rating_patterns.append((pat.pattern, total_uses, avg_rating))
    
    # Selection strength: frequency change rate
    # Simplified: compare early vs late usage for high vs low rating patterns
    return {
        'high_rating_count': len(high_rating_patterns),
        'low_rating_count': len(low_rating_patterns),
        'high_rating_patterns': high_rating_patterns[:10],
        'low_rating_patterns': low_rating_patterns[:10]
    }


# ============================================================================
# E. Lexical Evolution Metrics
# ============================================================================

def calculate_lexical_metrics(records: List[Dict]) -> Dict[str, Any]:
    """Calculate lexical evolution metrics."""
    innovation_by_round = defaultdict(int)
    tokens_by_round = defaultdict(set)
    all_tokens = set()
    
    for record in records:
        round_num = int(record.get('round', 0))
        unknown_tokens_str = record.get('unknown_tokens', '').strip()
        
        if unknown_tokens_str:
            tokens = unknown_tokens_str.split()
            for token in tokens:
                if token not in tokens_by_round[round_num]:
                    # New token for this round
                    innovation_by_round[round_num] += 1
                tokens_by_round[round_num].add(token)
                all_tokens.add(token)
    
    # Lexical reuse rate
    reuse_rates = {}
    for round_num in sorted(tokens_by_round.keys()):
        if round_num > 1:
            prev_tokens = set()
            for prev_round in range(1, round_num):
                prev_tokens.update(tokens_by_round.get(prev_round, set()))
            
            current_tokens = tokens_by_round[round_num]
            reused = len(current_tokens & prev_tokens)
            reuse_rate = reused / len(current_tokens) * 100 if current_tokens else 0
            reuse_rates[round_num] = reuse_rate
    
    # Core vocabulary (tokens in high-rating patterns)
    high_rating_records = [r for r in records if r.get('partner_rating', '').strip() and float(r.get('partner_rating', 0)) >= 3.5]
    core_tokens = Counter()
    for record in high_rating_records:
        unknown_tokens_str = record.get('unknown_tokens', '').strip()
        if unknown_tokens_str:
            for token in unknown_tokens_str.split():
                core_tokens[token] += 1
    
    return {
        'innovation_by_round': dict(innovation_by_round),
        'reuse_rates': reuse_rates,
        'core_vocabulary': dict(core_tokens.most_common(20)),
        'total_unique_tokens': len(all_tokens)
    }


# ============================================================================
# F. Structural Evolution Metrics
# ============================================================================

def calculate_structural_metrics(records: List[Dict]) -> Dict[str, Any]:
    """Calculate structural evolution metrics."""
    message_lengths_by_round = defaultdict(list)
    bigram_ratio_by_round = defaultdict(lambda: {'bigram': 0, 'trigram': 0, 'total': 0})
    
    for record in records:
        round_num = int(record.get('round', 0))
        content = record.get('content_raw', '')
        
        # Message length
        if content:
            tokens = content.split()
            message_lengths_by_round[round_num].append(len(tokens))
        
        # N-gram ratio
        bigram = record.get('tail_bigram', '').strip()
        trigram = record.get('tail_trigram', '').strip()
        
        if bigram:
            bigram_ratio_by_round[round_num]['bigram'] += 1
            bigram_ratio_by_round[round_num]['total'] += 1
        if trigram:
            bigram_ratio_by_round[round_num]['trigram'] += 1
            bigram_ratio_by_round[round_num]['total'] += 1
    
    # Average message length by round
    avg_lengths = {}
    for round_num, lengths in message_lengths_by_round.items():
        avg_lengths[round_num] = sum(lengths) / len(lengths) if lengths else 0
    
    # N-gram ratio by round
    ngram_ratios = {}
    for round_num, counts in bigram_ratio_by_round.items():
        total = counts['total']
        if total > 0:
            ngram_ratios[round_num] = {
                'bigram_ratio': counts['bigram'] / total * 100,
                'trigram_ratio': counts['trigram'] / total * 100
            }
    
    return {
        'avg_message_length_by_round': avg_lengths,
        'ngram_ratio_by_round': ngram_ratios
    }


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_language_evolution(messages_csv: Path, output_dir: Path) -> None:
    """Main analysis function."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading messages...")
    records = load_messages_csv(messages_csv)
    print(f"Loaded {len(records)} message records")
    
    # A. Life Cycle Metrics
    print("\nCalculating life cycle metrics...")
    lifecycle_metrics = calculate_lifecycle_metrics(records)
    
    # B. Diffusion Metrics
    print("Calculating diffusion metrics...")
    diffusion_metrics = calculate_diffusion_metrics(lifecycle_metrics['pattern_records'])
    
    # C. Evolution Paths
    print("Calculating evolution paths...")
    evolution_paths = calculate_evolution_paths(lifecycle_metrics['pattern_records'])
    
    # D. Selection Metrics
    print("Calculating selection metrics...")
    selection_metrics = calculate_selection_metrics(lifecycle_metrics['pattern_records'])
    
    # E. Lexical Metrics
    print("Calculating lexical metrics...")
    lexical_metrics = calculate_lexical_metrics(records)
    
    # F. Structural Metrics
    print("Calculating structural metrics...")
    structural_metrics = calculate_structural_metrics(records)
    
    # Write results
    print("\nWriting results...")
    write_evolution_results(
        output_dir,
        lifecycle_metrics,
        diffusion_metrics,
        evolution_paths,
        selection_metrics,
        lexical_metrics,
        structural_metrics
    )
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


def write_evolution_results(
    output_dir: Path,
    lifecycle: Dict,
    diffusion: Dict,
    evolution_paths: Dict,
    selection: Dict,
    lexical: Dict,
    structural: Dict
) -> None:
    """Write all evolution metrics to CSV files."""
    
    # Life cycle metrics
    with open(output_dir / 'lifecycle_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['First Appearance - Early (1-3)', lifecycle['first_appearance_distribution']['early']])
        writer.writerow(['First Appearance - Mid (4-7)', lifecycle['first_appearance_distribution']['mid']])
        writer.writerow(['First Appearance - Late (8-14)', lifecycle['first_appearance_distribution']['late']])
        writer.writerow(['Avg Stabilization Time', sum(lifecycle['stabilization_times']) / len(lifecycle['stabilization_times']) if lifecycle['stabilization_times'] else 0])
        writer.writerow(['Survival Rate (%)', lifecycle['survival_rates']['survival_rate']])
        writer.writerow(['Extinction Rate (%)', lifecycle['extinction_rates']['extinction_rate']])
    
    # Diffusion metrics
    with open(output_dir / 'diffusion_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Avg Adoption Rate', diffusion['avg_adoption_rate']])
        writer.writerow(['Avg Cross-Pair Count', diffusion['avg_cross_pair_count']])
    
    # Evolution paths
    with open(output_dir / 'evolution_paths.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Type', 'From', 'To', 'From Round', 'To Round'])
        for path in evolution_paths['compositional_paths']:
            writer.writerow(['Compositional', path['from'], path['to'], path['from_round'], path['to_round']])
        for path in evolution_paths['simplification_paths']:
            writer.writerow(['Simplification', path['from'], path['to'], path['from_round'], path['to_round']])
    
    # Lexical metrics
    with open(output_dir / 'lexical_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Innovation Count', 'Reuse Rate (%)'])
        for round_num in sorted(lexical['innovation_by_round'].keys()):
            innovation = lexical['innovation_by_round'][round_num]
            reuse = lexical['reuse_rates'].get(round_num, 0)
            writer.writerow([round_num, innovation, reuse])
    
    # Structural metrics
    with open(output_dir / 'structural_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Avg Message Length', 'Bigram Ratio (%)', 'Trigram Ratio (%)'])
        for round_num in sorted(structural['avg_message_length_by_round'].keys()):
            avg_len = structural['avg_message_length_by_round'][round_num]
            ngram_ratios = structural['ngram_ratio_by_round'].get(round_num, {})
            bigram_ratio = ngram_ratios.get('bigram_ratio', 0)
            trigram_ratio = ngram_ratios.get('trigram_ratio', 0)
            writer.writerow([round_num, avg_len, bigram_ratio, trigram_ratio])


def main():
    parser = argparse.ArgumentParser(description="Analyze language evolution metrics")
    parser.add_argument(
        "--messages_csv",
        type=Path,
        default=Path("results/emergent_language/messages.csv"),
        help="Path to messages CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/language_evolution"),
        help="Output directory for evolution metrics"
    )
    
    args = parser.parse_args()
    analyze_language_evolution(args.messages_csv, args.output_dir)


if __name__ == "__main__":
    main()

