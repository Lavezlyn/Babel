"""
Show detailed information about Code Words and N-grams in the main table.
Outputs the actual tokens/patterns and their meanings.
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# Resource types in the game
RESOURCE_TYPES = {"meat", "grain", "water", "fruit", "fish"}


def get_vocab_meanings_for_group(messages_data: List[Dict]) -> Dict[str, str]:
    """Get vocabulary mappings from log files in this group."""
    vocab_meanings = {}
    for msg in messages_data[:10]:  # Sample to get vocab
        log_path = msg.get('log_path', '')
        if log_path:
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                vocab = data.get('vocabulary', {})
                if vocab:
                    vocab_meanings.update(vocab)
                    break  # Use first available vocab
            except:
                continue
    return vocab_meanings


def contains_resource_name(pattern: str, vocab_meanings: Dict[str, str]) -> bool:
    """Check if pattern contains resource names."""
    tokens = pattern.lower().split()
    
    # Check direct resource names
    for token in tokens:
        if token in RESOURCE_TYPES:
            return True
    
    # Check vocabulary mappings
    for token in tokens:
        meaning = vocab_meanings.get(token, "").lower()
        if meaning in RESOURCE_TYPES:
            return True
    
    return False


def calculate_code_word_score(ratings: List[float], teammate_ratings: List[float], 
                              opponent_ratings: List[float], count: int) -> float:
    """Calculate Code Word Score (0-1) based on multiple criteria."""
    if not ratings or count < 5:
        return 0.0
    
    score = 0.0
    
    # 1. Average rating (40%)
    avg_rating = sum(ratings) / len(ratings)
    if avg_rating > 3.5:
        score += 0.4
    
    # 2. Rating consistency (20%)
    if len(ratings) > 1:
        mean = avg_rating
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        std = variance ** 0.5
        consistency = 1.0 / (1.0 + std) if std > 0 else 1.0
        if consistency > 0.7:
            score += 0.2
    
    # 3. Discrimination score (20%)
    if teammate_ratings and opponent_ratings:
        teammate_avg = sum(teammate_ratings) / len(teammate_ratings)
        opponent_avg = sum(opponent_ratings) / len(opponent_ratings)
        discrimination = teammate_avg - opponent_avg
        if discrimination > 0.5:
            score += 0.2
    
    # 4. Sample size (20%)
    if count >= 5:
        score += 0.2
    
    return score


def get_code_words_details(messages_data: List[Dict], vocab_meanings: Dict[str, str],
                           min_score: float = 0.7) -> List[Dict]:
    """Get detailed information about code words."""
    group_tokens = Counter()
    token_ratings: Dict[str, List[float]] = defaultdict(list)
    token_teammate_ratings: Dict[str, List[float]] = defaultdict(list)
    token_opponent_ratings: Dict[str, List[float]] = defaultdict(list)
    
    for msg in messages_data:
        unknown_str = msg.get('unknown_tokens', '').strip()
        rating_str = msg.get('partner_rating', '').strip()
        is_teammate_gt = msg.get('is_teammate_gt', '').strip()
        rating = None
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass
        
        if unknown_str:
            for token in unknown_str.split():
                group_tokens[token] += 1
                if rating is not None:
                    token_ratings[token].append(rating)
                    if is_teammate_gt == '1':
                        token_teammate_ratings[token].append(rating)
                    elif is_teammate_gt == '0':
                        token_opponent_ratings[token].append(rating)
    
    code_words = []
    for token, usage_count in group_tokens.items():
        if usage_count < 5:
            continue
        
        if contains_resource_name(token, vocab_meanings):
            continue
        
        ratings = token_ratings.get(token, [])
        teammate_ratings = token_teammate_ratings.get(token, [])
        opponent_ratings = token_opponent_ratings.get(token, [])
        
        code_word_score = calculate_code_word_score(
            ratings, teammate_ratings, opponent_ratings, usage_count
        )
        
        if code_word_score >= min_score:
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            meaning = vocab_meanings.get(token, "N/A (invented token)")
            
            code_words.append({
                'token': token,
                'meaning': meaning,
                'count': usage_count,
                'avg_rating': avg_rating,
                'rating_count': len(ratings),
                'code_word_score': code_word_score,
                'teammate_avg': sum(teammate_ratings) / len(teammate_ratings) if teammate_ratings else None,
                'opponent_avg': sum(opponent_ratings) / len(opponent_ratings) if opponent_ratings else None,
            })
    
    return sorted(code_words, key=lambda x: x['code_word_score'], reverse=True)


def get_ngrams_details(messages_data: List[Dict], vocab_meanings: Dict[str, str],
                      min_score: float = 0.7) -> List[Dict]:
    """Get detailed information about n-grams."""
    group_ngrams = Counter()
    ngram_ratings: Dict[str, List[float]] = defaultdict(list)
    ngram_teammate_ratings: Dict[str, List[float]] = defaultdict(list)
    ngram_opponent_ratings: Dict[str, List[float]] = defaultdict(list)
    
    for msg in messages_data:
        bigram = msg.get('tail_bigram', '').strip()
        trigram = msg.get('tail_trigram', '').strip()
        rating_str = msg.get('partner_rating', '').strip()
        is_teammate_gt = msg.get('is_teammate_gt', '').strip()
        rating = None
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass
        
        for ngram in [bigram, trigram]:
            if ngram:
                group_ngrams[ngram] += 1
                if rating is not None:
                    ngram_ratings[ngram].append(rating)
                    if is_teammate_gt == '1':
                        ngram_teammate_ratings[ngram].append(rating)
                    elif is_teammate_gt == '0':
                        ngram_opponent_ratings[ngram].append(rating)
    
    ngrams = []
    for ngram, usage_count in group_ngrams.items():
        if usage_count < 5:
            continue
        
        if contains_resource_name(ngram, vocab_meanings):
            continue
        
        ratings = ngram_ratings.get(ngram, [])
        teammate_ratings = ngram_teammate_ratings.get(ngram, [])
        opponent_ratings = ngram_opponent_ratings.get(ngram, [])
        
        code_word_score = calculate_code_word_score(
            ratings, teammate_ratings, opponent_ratings, usage_count
        )
        
        if code_word_score >= min_score:
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Translate n-gram tokens to meanings
            tokens = ngram.split()
            meanings = [vocab_meanings.get(t, f"[{t}]") for t in tokens]
            meaning_str = " ".join(meanings)
            
            ngrams.append({
                'ngram': ngram,
                'meaning': meaning_str,
                'count': usage_count,
                'avg_rating': avg_rating,
                'rating_count': len(ratings),
                'code_word_score': code_word_score,
                'teammate_avg': sum(teammate_ratings) / len(teammate_ratings) if teammate_ratings else None,
                'opponent_avg': sum(opponent_ratings) / len(opponent_ratings) if opponent_ratings else None,
            })
    
    return sorted(ngrams, key=lambda x: x['code_word_score'], reverse=True)


def parse_game_id(game_id: str) -> Dict[str, str]:
    """Parse game ID to extract model, condition, and hint."""
    # Format: resource_exchange_MODEL_HINT_condition_timestamp
    parts = game_id.split('_')
    model = 'unknown'
    condition = 'unknown'
    invention_hint = 'unknown'
    
    if 'gemini' in game_id.lower():
        model = 'gemini-2.5-flash'
    elif 'gpt' in game_id.lower():
        model = 'gpt-5.2'
    elif 'deepseek' in game_id.lower():
        model = 'deepseek-r1'
    
    if 'balanced' in game_id.lower():
        condition = 'balanced'
    elif 'reward' in game_id.lower():
        condition = 'reward_strong'
    elif 'penalty' in game_id.lower():
        condition = 'penalty_strong'
    
    # Check for hint (True/False in filename)
    if '_True_' in game_id or game_id.endswith('_True'):
        invention_hint = 'True'
    elif '_False_' in game_id or game_id.endswith('_False'):
        invention_hint = 'False'
    
    return {
        'model': model,
        'condition': condition,
        'invention_hint': invention_hint
    }


def load_csv(file_path: Path) -> List[Dict]:
    """Load CSV file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Show detailed Code Words and N-grams information"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="results/emergent_language",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/code_words_details.txt",
        help="Output text file path",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")
    
    # Load data
    print("Loading data...")
    messages_data = load_csv(data_dir / "messages.csv")
    print(f"Loaded {len(messages_data)} messages")
    
    # Group by model, condition, hint
    grouped_messages: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    
    for msg in messages_data:
        game_id = msg.get('game_id', '')
        info = parse_game_id(game_id)
        model = info['model']
        condition = info['condition']
        invention_hint = info['invention_hint']
        
        if model != 'unknown' and condition != 'unknown':
            key = (model, condition, invention_hint)
            grouped_messages[key].append(msg)
    
    # Generate report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("CODE WORDS AND N-GRAMS DETAILED INFORMATION\n")
        f.write("=" * 100 + "\n\n")
        f.write("This report shows the actual tokens/patterns that meet the Code Word Score ≥ 0.7 threshold,\n")
        f.write("excluding resource names, as used in the main table.\n\n")
        
        # Sort groups
        condition_order = {'balanced': 1, 'reward_strong': 2, 'penalty_strong': 3}
        sorted_groups = sorted(grouped_messages.items(), key=lambda x: (
            x[0][0],  # model
            condition_order.get(x[0][1], 999),  # condition
            x[0][2]  # hint
        ))
        
        for (model, condition, invention_hint), group_messages in sorted_groups:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Model: {model}\n")
            f.write(f"Condition: {condition}\n")
            f.write(f"Invention Hint: {invention_hint}\n")
            f.write("=" * 100 + "\n\n")
            
            # Get vocabulary
            vocab_meanings = get_vocab_meanings_for_group(group_messages)
            
            # Get code words
            code_words = get_code_words_details(group_messages, vocab_meanings, min_score=0.7)
            
            # Get n-grams
            ngrams = get_ngrams_details(group_messages, vocab_meanings, min_score=0.7)
            
            # Write code words
            f.write(f"CODE WORDS (Count: {len(code_words)})\n")
            f.write("-" * 100 + "\n")
            if code_words:
                for i, cw in enumerate(code_words, 1):
                    f.write(f"\n{i}. Token: {cw['token']}\n")
                    f.write(f"   Meaning: {cw['meaning']}\n")
                    f.write(f"   Usage Count: {cw['count']}\n")
                    f.write(f"   Avg Rating: {cw['avg_rating']:.2f} (n={cw['rating_count']})\n")
                    f.write(f"   Code Word Score: {cw['code_word_score']:.2f}\n")
                    if cw['teammate_avg'] is not None:
                        f.write(f"   Teammate Avg Rating: {cw['teammate_avg']:.2f}\n")
                    if cw['opponent_avg'] is not None:
                        f.write(f"   Opponent Avg Rating: {cw['opponent_avg']:.2f}\n")
            else:
                f.write("   (No code words found)\n")
            
            # Write n-grams
            f.write(f"\n\nN-GRAMS (Count: {len(ngrams)})\n")
            f.write("-" * 100 + "\n")
            if ngrams:
                for i, ng in enumerate(ngrams, 1):
                    f.write(f"\n{i}. N-gram: {ng['ngram']}\n")
                    f.write(f"   Meaning: {ng['meaning']}\n")
                    f.write(f"   Usage Count: {ng['count']}\n")
                    f.write(f"   Avg Rating: {ng['avg_rating']:.2f} (n={ng['rating_count']})\n")
                    f.write(f"   Code Word Score: {ng['code_word_score']:.2f}\n")
                    if ng['teammate_avg'] is not None:
                        f.write(f"   Teammate Avg Rating: {ng['teammate_avg']:.2f}\n")
                    if ng['opponent_avg'] is not None:
                        f.write(f"   Opponent Avg Rating: {ng['opponent_avg']:.2f}\n")
            else:
                f.write("   (No n-grams found)\n")
            
            f.write("\n")
    
    print(f"\nDetailed report saved to: {output_path}")
    print(f"Total groups analyzed: {len(sorted_groups)}")


if __name__ == "__main__":
    main()

