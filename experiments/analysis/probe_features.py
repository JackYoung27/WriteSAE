#!/usr/bin/env python3
"""Automated feature interpretability via statistical probing and vocabulary projection.

Instead of reading top-activating texts (which don't cluster semantically for GDN states),
this module correlates each feature's activation with measurable text properties across
the full corpus. GDN states encode compressed recurrent memory, not token-level semantics.
Features should correlate with document-level properties: register, complexity, structure.

Three probing approaches:
  1. Statistical probing: correlate feature activations with 50+ text properties
  2. Nonlinear probing: random forest predicts "feature active?" from all properties
  3. Contrastive probing: compare top-10% vs bottom-10% activation groups per property

Plus vocabulary projection: decode w vectors through the GDN output projection
to find what tokens each feature retrieves when queried.

Usage: called from run_modal.py via --stage probe-features.
"""
from __future__ import annotations

import gzip
import math
import re
import time
from collections import Counter

import numpy as np
import torch
from scipy import stats as scipy_stats


# Text property extraction (expanded: 50+ properties)

# Python and common programming keywords
_PYTHON_KW = frozenset([
    "def", "class", "import", "from", "return", "if", "else", "elif", "for",
    "while", "try", "except", "with", "as", "in", "not", "and", "or", "is",
    "True", "False", "None", "lambda", "yield", "raise", "pass", "break",
    "continue", "async", "await", "global", "nonlocal", "assert", "del",
])
_JS_KW = frozenset([
    "function", "var", "let", "const", "return", "if", "else", "for", "while",
    "switch", "case", "break", "continue", "new", "this", "class", "import",
    "export", "default", "async", "await", "try", "catch", "throw", "typeof",
    "instanceof", "null", "undefined", "true", "false",
])
_ALL_CODE_KW = _PYTHON_KW | _JS_KW

# Function words (English closed-class words the model might track)
_FUNCTION_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "same", "than", "too", "very",
    "just", "because", "if", "when", "while", "although", "though",
    "that", "which", "who", "whom", "whose", "what", "where", "how",
    "this", "these", "those", "it", "its", "he", "she", "they", "we",
    "I", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their",
])


def _ngram_repetition(words: list[str], n: int) -> float:
    """Fraction of n-grams that appear more than once."""
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(ngrams), 1)


def _bracket_nesting_depth(text: str) -> float:
    """Maximum nesting depth of brackets/parens/braces."""
    depth = 0
    max_depth = 0
    openers = set("([{")
    closers = set(")]}")
    for c in text:
        if c in openers:
            depth += 1
            max_depth = max(max_depth, depth)
        elif c in closers:
            depth = max(0, depth - 1)
    return float(max_depth)


def _detect_language(text: str) -> tuple[float, float, float]:
    """Rough language detection: CJK ratio, Cyrillic ratio, Latin ratio.

    Returns (cjk_ratio, cyrillic_ratio, latin_ratio).
    Qwen3.5 is multilingual, so features may specialize by script.
    """
    n = max(len(text), 1)
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
              or '\u3400' <= c <= '\u4dbf'
              or '\uf900' <= c <= '\ufaff'
              or '\U00020000' <= c <= '\U0002a6df'
              or '\u3040' <= c <= '\u309f'  # Hiragana
              or '\u30a0' <= c <= '\u30ff'  # Katakana
              or '\uac00' <= c <= '\ud7af')  # Korean
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return cjk / n, cyrillic / n, latin / n


def compute_text_properties(text: str) -> dict[str, float]:
    """Extract 50+ measurable properties from a text sequence.

    Properties span surface statistics, syntactic proxies, information-theoretic
    measures, positional markers, code indicators, and script detection.
    Each property captures a dimension that a recurrent memory might encode
    for next-token prediction.
    """
    n_chars = max(len(text), 1)
    words = text.split()
    n_words = max(len(words), 1)
    words_lower = [w.lower() for w in words]

    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', text)
    n_sentences = max(sum(1 for s in sentences if s.strip()), 1)

    # Word-level stats
    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_len = sum(word_lengths) / len(word_lengths)
    word_len_std = float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0

    # ---- ORIGINAL 15 PROPERTIES ----

    # Punctuation density
    punct_chars = sum(1 for c in text if c in '.,;:!?()[]{}"\'-')
    punct_density = punct_chars / n_chars

    # Numeric density
    digit_chars = sum(1 for c in text if c.isdigit())
    digit_density = digit_chars / n_chars

    # Number token count
    numbers = re.findall(r'\d+\.?\d*', text)
    number_count = len(numbers)

    # Uppercase ratio
    upper_chars = sum(1 for c in text if c.isupper())
    upper_ratio = upper_chars / n_chars

    # Newline density
    newline_count = text.count('\n')
    newline_density = newline_count / n_chars

    # Quote density
    quote_chars = (
        text.count('"') + text.count("'")
        + text.count('\u201c') + text.count('\u201d')
        + text.count('\u2018') + text.count('\u2019')
    )
    quote_density = quote_chars / n_chars

    # Code indicators
    code_chars = sum(1 for c in text if c in '{}();=<>|&^~`')
    code_density = code_chars / n_chars

    # URL presence
    has_url = float(bool(re.search(r'https?://|www\.', text)))

    # Average sentence length
    avg_sent_len = n_words / n_sentences

    # Type-token ratio
    unique_words = len(set(words_lower))
    type_token_ratio = unique_words / n_words

    # Whitespace density
    whitespace_chars = sum(1 for c in text if c.isspace())
    whitespace_density = whitespace_chars / n_chars

    # Character entropy (bits per character)
    char_counts = Counter(text)
    char_probs = np.array([c / n_chars for c in char_counts.values()])
    char_entropy = float(-np.sum(char_probs * np.log2(char_probs + 1e-12)))

    # Long word ratio
    long_word_ratio = sum(1 for w in words if len(w) >= 6) / n_words

    # ---- NEW: SYNTACTIC PROXIES ----

    # Function word ratio (proxy for syntax density)
    func_word_count = sum(1 for w in words_lower if w in _FUNCTION_WORDS)
    function_word_ratio = func_word_count / n_words

    # Content word ratio (inverse of function words)
    content_word_ratio = 1.0 - function_word_ratio

    # Comma density (clause boundary proxy)
    comma_count = text.count(',')
    comma_density = comma_count / n_chars

    # Semicolon + colon density (complex sentence proxy)
    semicolon_colon = text.count(';') + text.count(':')
    semicolon_colon_density = semicolon_colon / n_chars

    # Question mark density (interrogative content)
    question_density = text.count('?') / n_chars

    # Exclamation density (emphasis/emotion)
    exclamation_density = text.count('!') / n_chars

    # Bracket nesting depth
    max_nesting = _bracket_nesting_depth(text)

    # Open bracket count (unmatched structure)
    open_parens = text.count('(') - text.count(')')
    open_brackets = text.count('[') - text.count(']')
    open_braces = text.count('{') - text.count('}')
    open_bracket_count = float(max(0, open_parens) + max(0, open_brackets) + max(0, open_braces))

    # Sentence length variance (complexity variation)
    sent_texts = [s.strip() for s in sentences if s.strip()]
    sent_lengths = [len(s.split()) for s in sent_texts] if sent_texts else [0]
    sent_len_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # ---- NEW: INFORMATION-THEORETIC PROPERTIES ----

    # Compression ratio (gzip compressed / raw bytes)
    text_bytes = text.encode('utf-8')
    compressed = gzip.compress(text_bytes, compresslevel=6)
    compression_ratio = len(compressed) / max(len(text_bytes), 1)

    # Word entropy (bits per word-type)
    word_counts = Counter(words_lower)
    total_words = sum(word_counts.values())
    if total_words > 0:
        word_probs = np.array([c / total_words for c in word_counts.values()])
        word_entropy = float(-np.sum(word_probs * np.log2(word_probs + 1e-12)))
    else:
        word_entropy = 0.0

    # Bigram repetition (n-gram overlap within text)
    bigram_rep = _ngram_repetition(words_lower, 2)

    # Trigram repetition
    trigram_rep = _ngram_repetition(words_lower, 3)

    # Hapax legomena ratio (words appearing exactly once)
    hapax_count = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = hapax_count / max(len(word_counts), 1)

    # Vocabulary growth rate (unique words in second half vs first half)
    if n_words >= 4:
        mid = n_words // 2
        first_half_vocab = len(set(words_lower[:mid]))
        second_half_vocab = len(set(words_lower[mid:]))
        vocab_growth = second_half_vocab / max(first_half_vocab, 1)
    else:
        vocab_growth = 1.0

    # ---- NEW: POSITIONAL / STRUCTURAL PROPERTIES ----

    # Position of last sentence boundary (relative)
    last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    last_sent_boundary_pos = last_period / n_chars if last_period >= 0 else 1.0

    # Paragraph count (double newlines)
    paragraph_count = float(len(re.split(r'\n\s*\n', text)))

    # Leading whitespace (indentation at start)
    leading_ws = len(text) - len(text.lstrip())
    leading_whitespace = float(leading_ws)

    # Trailing whitespace
    # Line count
    line_count = float(text.count('\n') + 1)

    # Average line length
    lines = text.split('\n')
    avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)

    # Line length variance (code vs prose indicator)
    line_lengths = [len(line) for line in lines]
    line_len_std = float(np.std(line_lengths)) if len(line_lengths) > 1 else 0.0

    # ---- NEW: CODE-SPECIFIC PROPERTIES ----

    # Programming keyword count
    code_keyword_count = float(sum(1 for w in words if w in _ALL_CODE_KW))
    code_keyword_density = code_keyword_count / n_words

    # Indentation depth (average leading spaces per line)
    indent_depths = []
    for line in lines:
        if line.strip():
            indent_depths.append(len(line) - len(line.lstrip()))
    avg_indent = float(np.mean(indent_depths)) if indent_depths else 0.0
    max_indent = float(max(indent_depths)) if indent_depths else 0.0

    # Hash/comment density (# for Python, // for JS/C)
    comment_markers = text.count('#') + text.count('//')
    comment_density = comment_markers / n_chars

    # Equals sign density (assignment indicator)
    equals_density = text.count('=') / n_chars

    # Dot density (method chaining, attribute access)
    dot_density = text.count('.') / n_chars

    # ---- NEW: SCRIPT / LANGUAGE DETECTION ----

    cjk_ratio, cyrillic_ratio, latin_ratio = _detect_language(text)

    # Non-ASCII ratio
    non_ascii = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii / n_chars

    # ---- NEW: DISCOURSE / REGISTER PROPERTIES ----

    # Dialogue indicator: lines starting with speech patterns
    dialogue_lines = sum(1 for line in lines if line.strip().startswith(('"', '\u201c', "'", '-', '\u2014')))
    dialogue_ratio = dialogue_lines / max(len(lines), 1)

    # List indicator (lines starting with bullets, numbers)
    list_lines = sum(1 for line in lines
                     if re.match(r'^\s*[\-\*\u2022]\s', line)
                     or re.match(r'^\s*\d+[\.\)]\s', line))
    list_ratio = list_lines / max(len(lines), 1)

    # Header indicator (short lines followed by longer content)
    header_count = sum(1 for i, line in enumerate(lines)
                       if line.strip() and len(line.strip()) < 60
                       and (line.strip().endswith(':') or line.strip().startswith('#')
                            or (line == line.upper() and len(line.strip()) > 2)))
    header_density = header_count / max(len(lines), 1)

    # Pronoun density (narrative voice indicator)
    pronouns = {"i", "me", "my", "mine", "myself",
                "you", "your", "yours", "yourself",
                "he", "him", "his", "she", "her", "hers",
                "we", "us", "our", "ours", "they", "them", "their"}
    pronoun_count = sum(1 for w in words_lower if w in pronouns)
    pronoun_density = pronoun_count / n_words

    # First person ratio
    first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
    first_person_count = sum(1 for w in words_lower if w in first_person)
    first_person_ratio = first_person_count / n_words

    # Named entity proxy: capitalized words not at sentence start
    cap_words = 0
    for i, w in enumerate(words):
        if i == 0:
            continue
        prev = words[i-1] if i > 0 else ""
        if w[0].isupper() and not prev.endswith(('.', '!', '?', ':')):
            cap_words += 1
    capitalized_word_ratio = cap_words / n_words

    # ---- NEW: TOKEN-LEVEL PROPERTIES ----

    # Average character code (high = rare/unicode-heavy)
    avg_char_ord = float(np.mean([ord(c) for c in text[:1000]])) if text else 0.0

    # Digit-letter transition count (mixed alphanumeric content)
    transitions = 0
    for i in range(1, min(len(text), 2000)):
        if (text[i].isdigit() != text[i-1].isdigit()) and (text[i].isalnum() and text[i-1].isalnum()):
            transitions += 1
    digit_letter_transitions = transitions / n_chars

    # Consecutive repeated character ratio
    consec_repeats = sum(1 for i in range(1, len(text)) if text[i] == text[i-1])
    consec_repeat_ratio = consec_repeats / n_chars

    # Space-separated token length distribution (mean of log-lengths)
    if words:
        log_lengths = [math.log1p(len(w)) for w in words]
        mean_log_word_len = float(np.mean(log_lengths))
    else:
        mean_log_word_len = 0.0

    return {
        # Original 15
        "avg_word_len": avg_word_len,
        "punct_density": punct_density,
        "digit_density": digit_density,
        "number_count": float(number_count),
        "upper_ratio": upper_ratio,
        "newline_density": newline_density,
        "quote_density": quote_density,
        "code_density": code_density,
        "has_url": has_url,
        "avg_sent_len": avg_sent_len,
        "type_token_ratio": type_token_ratio,
        "whitespace_density": whitespace_density,
        "char_entropy": char_entropy,
        "long_word_ratio": long_word_ratio,
        "n_words": float(n_words),
        # Syntactic proxies
        "function_word_ratio": function_word_ratio,
        "content_word_ratio": content_word_ratio,
        "comma_density": comma_density,
        "semicolon_colon_density": semicolon_colon_density,
        "question_density": question_density,
        "exclamation_density": exclamation_density,
        "max_nesting_depth": max_nesting,
        "open_bracket_count": open_bracket_count,
        "word_len_std": word_len_std,
        "sent_len_std": sent_len_std,
        # Information-theoretic
        "compression_ratio": compression_ratio,
        "word_entropy": word_entropy,
        "bigram_repetition": bigram_rep,
        "trigram_repetition": trigram_rep,
        "hapax_ratio": hapax_ratio,
        "vocab_growth": vocab_growth,
        # Positional / structural
        "last_sent_boundary_pos": last_sent_boundary_pos,
        "paragraph_count": paragraph_count,
        "leading_whitespace": leading_whitespace,
        "line_count": line_count,
        "avg_line_length": avg_line_length,
        "line_len_std": line_len_std,
        # Code-specific
        "code_keyword_density": code_keyword_density,
        "avg_indent": avg_indent,
        "max_indent": max_indent,
        "comment_density": comment_density,
        "equals_density": equals_density,
        "dot_density": dot_density,
        # Script / language
        "cjk_ratio": cjk_ratio,
        "cyrillic_ratio": cyrillic_ratio,
        "latin_ratio": latin_ratio,
        "non_ascii_ratio": non_ascii_ratio,
        # Discourse / register
        "dialogue_ratio": dialogue_ratio,
        "list_ratio": list_ratio,
        "header_density": header_density,
        "pronoun_density": pronoun_density,
        "first_person_ratio": first_person_ratio,
        "capitalized_word_ratio": capitalized_word_ratio,
        # Token-level
        "avg_char_ord": avg_char_ord,
        "digit_letter_transitions": digit_letter_transitions,
        "consec_repeat_ratio": consec_repeat_ratio,
        "mean_log_word_len": mean_log_word_len,
    }


PROPERTY_NAMES = list(compute_text_properties("test text with some words 123 and things").keys())


def _empty_property_summary() -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "n_correlated_features": 0,
            "mean_abs_rho": 0.0,
            "max_abs_rho": 0.0,
        }
        for name in PROPERTY_NAMES
    }


# Statistical probing: correlate features with text properties

def probe_features(
    sae: torch.nn.Module,
    states: torch.Tensor,
    texts: list[str],
    batch_size: int = 512,
    min_frequency: float = 0.01,
    correlation_threshold: float = 0.15,
    p_threshold: float = 0.01,
) -> dict:
    """Correlate each feature's activation with text properties.

    For each alive feature, compute Spearman rank correlation between
    the feature's activation vector (across all N samples) and each
    text property vector. Features with |rho| > correlation_threshold
    and p < p_threshold are flagged as interpretable.

    Args:
        sae: trained SAE (on device, eval mode)
        states: (N, d_k, d_v) float32 tensor
        texts: list of N text strings
        batch_size: encoding batch size
        min_frequency: minimum activation frequency to consider a feature
        correlation_threshold: minimum |rho| to flag a correlation
        p_threshold: maximum p-value to flag a correlation

    Returns:
        dict with per-feature correlations and summary statistics
    """
    device = next(sae.parameters()).device
    N = states.shape[0]
    assert len(texts) == N, f"states ({N}) and texts ({len(texts)}) count mismatch"

    print(f"Computing text properties for {N} samples...")
    t0 = time.time()
    property_matrix = np.zeros((N, len(PROPERTY_NAMES)), dtype=np.float32)
    for i, text in enumerate(texts):
        props = compute_text_properties(text)
        for j, name in enumerate(PROPERTY_NAMES):
            property_matrix[i, j] = props[name]
    print(f"  Text properties: {time.time() - t0:.1f}s")

    print("Computing activations for all features...")
    t0 = time.time()
    all_acts = []
    for i in range(0, N, batch_size):
        batch = states[i : i + batch_size].to(device)
        with torch.no_grad():
            acts = sae.encode(batch)
        all_acts.append(acts.cpu().numpy())
    act_matrix = np.concatenate(all_acts, axis=0)  # (N, n_features)
    n_features = act_matrix.shape[1]
    print(f"  Activations: {act_matrix.shape} in {time.time() - t0:.1f}s")

    # Identify alive features
    freq = (act_matrix > 0).astype(np.float32).mean(axis=0)
    alive_mask = freq >= min_frequency
    alive_indices = np.where(alive_mask)[0]
    print(f"  Alive features (freq >= {min_frequency}): {len(alive_indices)}/{n_features}")

    if len(alive_indices) == 0:
        print("  No alive features; returning empty probe result.")
        return {
            "n_samples": N,
            "n_features_total": n_features,
            "n_alive": 0,
            "n_interpretable": 0,
            "n_interpretable_bonferroni": 0,
            "interpretable_fraction": 0.0,
            "interpretable_fraction_bonferroni": 0.0,
            "correlation_threshold": correlation_threshold,
            "p_threshold": p_threshold,
            "bonferroni_alpha": 0.0,
            "n_tests": 0,
            "property_names": PROPERTY_NAMES,
            "property_summary": _empty_property_summary(),
            "features": [],
        }

    # Compute Spearman correlations for each alive feature
    n_tests = len(alive_indices) * len(PROPERTY_NAMES)
    bonferroni_alpha = p_threshold / n_tests  # Bonferroni-corrected threshold
    print(f"Computing correlations for {len(alive_indices)} features x {len(PROPERTY_NAMES)} properties...")
    print(f"  Multiple testing: {n_tests} tests, Bonferroni alpha={bonferroni_alpha:.2e}")
    t0 = time.time()

    feature_results = []
    n_interpretable = 0
    n_interpretable_bonferroni = 0

    for fi in alive_indices:
        acts_i = act_matrix[:, fi]
        correlations = {}
        best_rho = 0.0
        best_prop = ""

        for j, prop_name in enumerate(PROPERTY_NAMES):
            prop_vals = property_matrix[:, j]
            # Skip if property has zero variance
            if np.std(prop_vals) < 1e-8:
                continue
            rho, pval = scipy_stats.spearmanr(acts_i, prop_vals)
            if np.isnan(rho):
                continue
            correlations[prop_name] = {"rho": float(rho), "p": float(pval)}
            if abs(rho) > abs(best_rho):
                best_rho = float(rho)
                best_prop = prop_name

        # Significant at uncorrected threshold
        significant = {
            k: v for k, v in correlations.items()
            if abs(v["rho"]) >= correlation_threshold and v["p"] < p_threshold
        }

        # Significant after Bonferroni correction
        significant_bonferroni = {
            k: v for k, v in correlations.items()
            if abs(v["rho"]) >= correlation_threshold and v["p"] < bonferroni_alpha
        }

        is_interpretable = len(significant) > 0
        if is_interpretable:
            n_interpretable += 1
        if len(significant_bonferroni) > 0:
            n_interpretable_bonferroni += 1

        feature_results.append({
            "feature_idx": int(fi),
            "frequency": float(freq[fi]),
            "mean_activation": float(acts_i.mean()),
            "best_property": best_prop,
            "best_rho": best_rho,
            "n_significant": len(significant),
            "n_significant_bonferroni": len(significant_bonferroni),
            "significant_correlations": dict(sorted(
                significant.items(), key=lambda x: abs(x[1]["rho"]), reverse=True
            )),
            "significant_correlations_bonferroni": dict(sorted(
                significant_bonferroni.items(), key=lambda x: abs(x[1]["rho"]), reverse=True
            )),
            "all_correlations": correlations,
        })

    print(f"  Correlations computed in {time.time() - t0:.1f}s")

    # Sort by number of significant correlations (most interpretable first)
    feature_results.sort(key=lambda x: (-x["n_significant"], -abs(x["best_rho"])))

    # Property-level summary: for each property, how many features correlate?
    property_summary = {}
    for prop_name in PROPERTY_NAMES:
        n_correlated = sum(
            1 for fr in feature_results
            if prop_name in fr.get("significant_correlations", {})
        )
        rhos = [
            fr["all_correlations"][prop_name]["rho"]
            for fr in feature_results
            if prop_name in fr.get("all_correlations", {})
        ]
        property_summary[prop_name] = {
            "n_correlated_features": n_correlated,
            "mean_abs_rho": float(np.mean(np.abs(rhos))) if rhos else 0.0,
            "max_abs_rho": float(np.max(np.abs(rhos))) if rhos else 0.0,
        }

    result = {
        "n_samples": N,
        "n_features_total": n_features,
        "n_alive": len(alive_indices),
        "n_interpretable": n_interpretable,
        "n_interpretable_bonferroni": n_interpretable_bonferroni,
        "interpretable_fraction": n_interpretable / max(len(alive_indices), 1),
        "interpretable_fraction_bonferroni": n_interpretable_bonferroni / max(len(alive_indices), 1),
        "correlation_threshold": correlation_threshold,
        "p_threshold": p_threshold,
        "bonferroni_alpha": bonferroni_alpha,
        "n_tests": n_tests,
        "property_names": PROPERTY_NAMES,
        "property_summary": property_summary,
        "features": feature_results,
    }

    print(f"\nResults: {n_interpretable}/{len(alive_indices)} features significant "
          f"(|rho| >= {correlation_threshold}, p < {p_threshold})")
    print(f"  After Bonferroni correction: {n_interpretable_bonferroni}/{len(alive_indices)} "
          f"(alpha={bonferroni_alpha:.2e})")

    for fr in feature_results[:20]:
        if fr["n_significant"] == 0:
            break
        sig_str = ", ".join(
            f"{k}={v['rho']:+.3f}"
            for k, v in list(fr["significant_correlations"].items())[:3]
        )
        print(f"  Feature {fr['feature_idx']:>4}: freq={fr['frequency']:.3f} "
              f"n_sig={fr['n_significant']} | {sig_str}")

    return result


# Nonlinear probing: random forest per feature

def probe_features_nonlinear(
    act_matrix: np.ndarray,
    property_matrix: np.ndarray,
    property_names: list[str],
    alive_indices: np.ndarray,
    freq: np.ndarray,
    n_folds: int = 5,
    accuracy_threshold: float = 0.60,
) -> dict:
    """For each alive feature, train a random forest to predict 'feature active (above median)' from all text properties.

    This catches nonlinear and combinatorial patterns that Spearman misses.
    A feature is interpretable if cross-validated accuracy exceeds chance (50%)
    by a meaningful margin.

    Returns dict with per-feature results and summary.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_alive = len(alive_indices)

    print(f"\nNonlinear probing: {n_alive} features, {n_folds}-fold CV, "
          f"threshold={accuracy_threshold:.0%}")
    t0 = time.time()

    # Standardize properties once
    scaler = StandardScaler()
    X = scaler.fit_transform(property_matrix)

    feature_results = []
    n_interpretable = 0

    for idx, fi in enumerate(alive_indices):
        acts_i = act_matrix[:, fi]

        # Binary label: above median activation (among nonzero activations, or overall)
        median_val = np.median(acts_i)
        y = (acts_i > median_val).astype(np.int32)

        # Skip if class balance is too extreme (<10% minority)
        minority_frac = min(y.mean(), 1.0 - y.mean())
        if minority_frac < 0.10:
            feature_results.append({
                "feature_idx": int(fi),
                "accuracy": 0.5,
                "importances": {},
                "interpretable": False,
                "skipped": True,
                "skip_reason": f"class imbalance ({minority_frac:.2f})",
            })
            continue

        # Stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_accs = []
        fold_importances = np.zeros(len(property_names))

        for train_idx, test_idx in skf.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=50, max_depth=6, min_samples_leaf=20,
                random_state=42, n_jobs=1,
            )
            clf.fit(X[train_idx], y[train_idx])
            acc = clf.score(X[test_idx], y[test_idx])
            fold_accs.append(acc)
            fold_importances += clf.feature_importances_

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        fold_importances /= n_folds

        # Top-3 important properties
        top_imp_idx = np.argsort(fold_importances)[::-1][:5]
        importances = {
            property_names[j]: float(fold_importances[j])
            for j in top_imp_idx if fold_importances[j] > 0.01
        }

        is_interpretable = mean_acc >= accuracy_threshold
        if is_interpretable:
            n_interpretable += 1

        feature_results.append({
            "feature_idx": int(fi),
            "accuracy": mean_acc,
            "accuracy_std": std_acc,
            "importances": importances,
            "interpretable": is_interpretable,
            "skipped": False,
        })

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{n_alive} features processed...")

    elapsed = time.time() - t0
    feature_results.sort(key=lambda x: -x["accuracy"])

    print(f"  Nonlinear probing: {elapsed:.1f}s")
    print(f"  Interpretable (accuracy >= {accuracy_threshold:.0%}): "
          f"{n_interpretable}/{n_alive} ({100 * n_interpretable / max(n_alive, 1):.1f}%)")

    # Top features
    for fr in feature_results[:10]:
        if not fr["interpretable"]:
            break
        imp_str = ", ".join(f"{k}={v:.3f}" for k, v in list(fr["importances"].items())[:3])
        print(f"  Feature {fr['feature_idx']:>4}: acc={fr['accuracy']:.3f} | {imp_str}")

    return {
        "n_alive": n_alive,
        "n_interpretable": n_interpretable,
        "interpretable_fraction": n_interpretable / max(n_alive, 1),
        "accuracy_threshold": accuracy_threshold,
        "n_folds": n_folds,
        "elapsed_s": elapsed,
        "features": feature_results,
    }


# Contrastive probing: top vs bottom activation groups

def probe_features_contrastive(
    act_matrix: np.ndarray,
    property_matrix: np.ndarray,
    property_names: list[str],
    alive_indices: np.ndarray,
    freq: np.ndarray,
    quantile: float = 0.10,
    min_effect_size: float = 0.50,  # Cohen's d threshold
) -> dict:
    """For each feature, compare properties of top-quantile vs bottom-quantile activation groups.

    More powerful than correlation for features with threshold effects:
    a feature that fires only on texts with digit_density > 0.05 will show
    a large between-group difference even if the correlation is weak.

    Effect size is Cohen's d. A feature is interpretable if any property
    shows |d| >= min_effect_size.

    Returns dict with per-feature contrastive results.
    """
    N = act_matrix.shape[0]
    n_alive = len(alive_indices)
    n_top = max(int(N * quantile), 10)
    n_bot = n_top

    print(f"\nContrastive probing: {n_alive} features, top/bottom {quantile:.0%} "
          f"({n_top} samples each), |d| >= {min_effect_size}")
    t0 = time.time()

    feature_results = []
    n_interpretable = 0

    for fi in alive_indices:
        acts_i = act_matrix[:, fi]

        # Top and bottom indices by activation
        sorted_idx = np.argsort(acts_i)
        bot_idx = sorted_idx[:n_bot]
        top_idx = sorted_idx[-n_top:]

        contrasts = {}
        best_d = 0.0
        best_prop = ""

        for j, prop_name in enumerate(property_names):
            prop_vals = property_matrix[:, j]
            top_vals = prop_vals[top_idx]
            bot_vals = prop_vals[bot_idx]

            # Cohen's d
            top_mean, bot_mean = top_vals.mean(), bot_vals.mean()
            top_std, bot_std = top_vals.std(), bot_vals.std()
            pooled_std = np.sqrt((top_std**2 + bot_std**2) / 2)

            if pooled_std < 1e-8:
                continue

            d = float((top_mean - bot_mean) / pooled_std)

            # Mann-Whitney U test for significance
            if len(top_vals) >= 5 and len(bot_vals) >= 5:
                _, pval = scipy_stats.mannwhitneyu(
                    top_vals, bot_vals, alternative='two-sided'
                )
            else:
                pval = 1.0

            contrasts[prop_name] = {
                "cohens_d": d,
                "p": float(pval),
                "top_mean": float(top_mean),
                "bot_mean": float(bot_mean),
            }

            if abs(d) > abs(best_d):
                best_d = d
                best_prop = prop_name

        # Significant contrasts (Bonferroni within this feature)
        n_props = len(property_names)
        bonf_alpha = 0.01 / n_props
        significant = {
            k: v for k, v in contrasts.items()
            if abs(v["cohens_d"]) >= min_effect_size and v["p"] < bonf_alpha
        }

        is_interpretable = len(significant) > 0
        if is_interpretable:
            n_interpretable += 1

        feature_results.append({
            "feature_idx": int(fi),
            "best_property": best_prop,
            "best_cohens_d": best_d,
            "n_significant": len(significant),
            "significant_contrasts": dict(sorted(
                significant.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True
            )),
            "all_contrasts": contrasts,
        })

    elapsed = time.time() - t0
    feature_results.sort(key=lambda x: (-x["n_significant"], -abs(x["best_cohens_d"])))

    print(f"  Contrastive probing: {elapsed:.1f}s")
    print(f"  Interpretable (|d| >= {min_effect_size}, Bonferroni p < {bonf_alpha:.2e}): "
          f"{n_interpretable}/{n_alive} ({100 * n_interpretable / max(n_alive, 1):.1f}%)")

    for fr in feature_results[:10]:
        if fr["n_significant"] == 0:
            break
        sig_str = ", ".join(
            f"{k}: d={v['cohens_d']:+.2f}"
            for k, v in list(fr["significant_contrasts"].items())[:3]
        )
        print(f"  Feature {fr['feature_idx']:>4}: n_sig={fr['n_significant']} | {sig_str}")

    return {
        "n_alive": n_alive,
        "n_interpretable": n_interpretable,
        "interpretable_fraction": n_interpretable / max(n_alive, 1),
        "quantile": quantile,
        "min_effect_size": min_effect_size,
        "elapsed_s": elapsed,
        "features": feature_results,
    }


# Unified enhanced probing: runs all three methods, computes union

def probe_features_enhanced(
    sae: torch.nn.Module,
    states: torch.Tensor,
    texts: list[str],
    batch_size: int = 512,
    min_frequency: float = 0.01,
    correlation_threshold: float = 0.15,
    p_threshold: float = 0.01,
    rf_accuracy_threshold: float = 0.60,
    contrastive_effect_size: float = 0.50,
) -> dict:
    """Run all three probing methods and compute union interpretability.

    1. Spearman correlation over the full property set
    2. Random forest nonlinear probing
    3. Contrastive top-vs-bottom probing

    A feature is 'interpretable' if ANY method flags it.
    Returns full results from all three methods plus union statistics.
    """
    device = next(sae.parameters()).device
    N = states.shape[0]
    assert len(texts) == N

    # --- Compute text properties ---
    print(f"Computing {len(PROPERTY_NAMES)} text properties for {N} samples...")
    t0 = time.time()
    property_matrix = np.zeros((N, len(PROPERTY_NAMES)), dtype=np.float32)
    for i, text in enumerate(texts):
        props = compute_text_properties(text)
        for j, name in enumerate(PROPERTY_NAMES):
            property_matrix[i, j] = props[name]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{N} texts processed...")
    print(f"  Text properties ({len(PROPERTY_NAMES)} props): {time.time() - t0:.1f}s")

    # --- Compute activations ---
    print("Computing activations...")
    t0 = time.time()
    all_acts = []
    for i in range(0, N, batch_size):
        batch = states[i : i + batch_size].to(device)
        with torch.no_grad():
            acts = sae.encode(batch)
        all_acts.append(acts.cpu().numpy())
    act_matrix = np.concatenate(all_acts, axis=0)
    n_features = act_matrix.shape[1]
    print(f"  Activations: {act_matrix.shape} in {time.time() - t0:.1f}s")

    # --- Identify alive features ---
    freq = (act_matrix > 0).astype(np.float32).mean(axis=0)
    alive_mask = freq >= min_frequency
    alive_indices = np.where(alive_mask)[0]
    n_alive = len(alive_indices)
    print(f"  Alive features: {n_alive}/{n_features}")

    if n_alive == 0:
        print("  No alive features; returning empty enhanced probe result.")
        empty_probe = {
            "n_samples": N,
            "n_features_total": n_features,
            "n_alive": 0,
            "n_interpretable": 0,
            "n_interpretable_bonferroni": 0,
            "interpretable_fraction": 0.0,
            "interpretable_fraction_bonferroni": 0.0,
            "correlation_threshold": correlation_threshold,
            "p_threshold": p_threshold,
            "bonferroni_alpha": 0.0,
            "n_tests": 0,
            "property_names": PROPERTY_NAMES,
            "property_summary": _empty_property_summary(),
            "features": [],
        }
        empty_method = {
            "n_alive": 0,
            "n_interpretable": 0,
            "interpretable_fraction": 0.0,
            "features": [],
        }
        return {
            "probe": empty_probe,
            "nonlinear": {
                **empty_method,
                "accuracy_threshold": rf_accuracy_threshold,
                "n_folds": 5,
                "elapsed_s": 0.0,
            },
            "contrastive": {
                **empty_method,
                "quantile": 0.10,
                "min_effect_size": contrastive_effect_size,
                "elapsed_s": 0.0,
            },
            "union": {
                "n_alive": 0,
                "n_interpretable_spearman": 0,
                "n_interpretable_rf": 0,
                "n_interpretable_contrastive": 0,
                "n_interpretable_union": 0,
                "interpretable_fraction_union": 0.0,
                "spearman_only": [],
                "rf_only": [],
                "contrastive_only": [],
                "all_three": [],
            },
        }

    # --- Method 1: Spearman correlations (expanded properties) ---
    print("\n" + "=" * 60)
    print(f"Method 1: Spearman Correlations ({len(PROPERTY_NAMES)} properties)")
    print("=" * 60)
    n_tests = n_alive * len(PROPERTY_NAMES)
    bonferroni_alpha = p_threshold / n_tests
    print(f"  {n_tests} tests, Bonferroni alpha={bonferroni_alpha:.2e}")
    t0 = time.time()

    spearman_results = []
    spearman_interpretable_set = set()

    for fi in alive_indices:
        acts_i = act_matrix[:, fi]
        correlations = {}
        best_rho = 0.0
        best_prop = ""

        for j, prop_name in enumerate(PROPERTY_NAMES):
            prop_vals = property_matrix[:, j]
            if np.std(prop_vals) < 1e-8:
                continue
            rho, pval = scipy_stats.spearmanr(acts_i, prop_vals)
            if np.isnan(rho):
                continue
            correlations[prop_name] = {"rho": float(rho), "p": float(pval)}
            if abs(rho) > abs(best_rho):
                best_rho = float(rho)
                best_prop = prop_name

        significant_bonferroni = {
            k: v for k, v in correlations.items()
            if abs(v["rho"]) >= correlation_threshold and v["p"] < bonferroni_alpha
        }

        if len(significant_bonferroni) > 0:
            spearman_interpretable_set.add(int(fi))

        spearman_results.append({
            "feature_idx": int(fi),
            "frequency": float(freq[fi]),
            "mean_activation": float(acts_i.mean()),
            "best_property": best_prop,
            "best_rho": best_rho,
            "n_significant_bonferroni": len(significant_bonferroni),
            "significant_correlations_bonferroni": dict(sorted(
                significant_bonferroni.items(), key=lambda x: abs(x[1]["rho"]), reverse=True
            )),
            "all_correlations": correlations,
        })

    spearman_results.sort(key=lambda x: (-x["n_significant_bonferroni"], -abs(x["best_rho"])))
    spearman_time = time.time() - t0
    print(f"  Spearman: {len(spearman_interpretable_set)}/{n_alive} interpretable "
          f"({100 * len(spearman_interpretable_set) / max(n_alive, 1):.1f}%) "
          f"in {spearman_time:.1f}s")

    # Property-level summary
    property_summary = {}
    for prop_name in PROPERTY_NAMES:
        rhos = [
            fr["all_correlations"][prop_name]["rho"]
            for fr in spearman_results
            if prop_name in fr.get("all_correlations", {})
        ]
        n_correlated = sum(
            1 for fr in spearman_results
            if prop_name in fr.get("significant_correlations_bonferroni", {})
        )
        property_summary[prop_name] = {
            "n_correlated_features": n_correlated,
            "mean_abs_rho": float(np.mean(np.abs(rhos))) if rhos else 0.0,
            "max_abs_rho": float(np.max(np.abs(rhos))) if rhos else 0.0,
        }

    # --- Method 2: Nonlinear (Random Forest) ---
    print("\n" + "=" * 60)
    print("Method 2: Nonlinear (Random Forest)")
    print("=" * 60)
    rf_result = probe_features_nonlinear(
        act_matrix, property_matrix, PROPERTY_NAMES,
        alive_indices, freq,
        accuracy_threshold=rf_accuracy_threshold,
    )
    rf_interpretable_set = {
        fr["feature_idx"] for fr in rf_result["features"]
        if fr["interpretable"]
    }

    # --- Method 3: Contrastive ---
    print("\n" + "=" * 60)
    print("Method 3: Contrastive (Top vs Bottom)")
    print("=" * 60)
    contrastive_result = probe_features_contrastive(
        act_matrix, property_matrix, PROPERTY_NAMES,
        alive_indices, freq,
        min_effect_size=contrastive_effect_size,
    )
    contrastive_interpretable_set = {
        fr["feature_idx"] for fr in contrastive_result["features"]
        if fr["n_significant"] > 0
    }

    # --- Union ---
    union_set = spearman_interpretable_set | rf_interpretable_set | contrastive_interpretable_set
    n_union = len(union_set)

    for fr in spearman_results:
        fi = fr["feature_idx"]
        fr["interpretable_spearman"] = fi in spearman_interpretable_set
        fr["interpretable_rf"] = fi in rf_interpretable_set
        fr["interpretable_contrastive"] = fi in contrastive_interpretable_set
        fr["interpretable_any"] = fi in union_set

    print("\n" + "=" * 60)
    print("UNION RESULTS")
    print("=" * 60)
    print(f"  Spearman only:     {len(spearman_interpretable_set)}/{n_alive} "
          f"({100 * len(spearman_interpretable_set) / max(n_alive, 1):.1f}%)")
    print(f"  Random Forest:     {len(rf_interpretable_set)}/{n_alive} "
          f"({100 * len(rf_interpretable_set) / max(n_alive, 1):.1f}%)")
    print(f"  Contrastive:       {len(contrastive_interpretable_set)}/{n_alive} "
          f"({100 * len(contrastive_interpretable_set) / max(n_alive, 1):.1f}%)")
    print(f"  UNION (any method): {n_union}/{n_alive} "
          f"({100 * n_union / max(n_alive, 1):.1f}%)")

    # Overlap analysis
    sp_only = spearman_interpretable_set - rf_interpretable_set - contrastive_interpretable_set
    rf_only = rf_interpretable_set - spearman_interpretable_set - contrastive_interpretable_set
    ct_only = contrastive_interpretable_set - spearman_interpretable_set - rf_interpretable_set
    all_three = spearman_interpretable_set & rf_interpretable_set & contrastive_interpretable_set
    print(f"  Spearman-only:     {len(sp_only)}")
    print(f"  RF-only:           {len(rf_only)}")
    print(f"  Contrastive-only:  {len(ct_only)}")
    print(f"  All three:         {len(all_three)}")

    # Build the 'probe' result consumed by figures/gen_probing_*.py and
    # downstream stages. Spearman is the main feature list.
    probe_compat = {
        "n_samples": N,
        "n_features_total": n_features,
        "n_alive": n_alive,
        "n_interpretable": len(spearman_interpretable_set),
        "n_interpretable_bonferroni": len(spearman_interpretable_set),
        "interpretable_fraction": len(spearman_interpretable_set) / max(n_alive, 1),
        "interpretable_fraction_bonferroni": len(spearman_interpretable_set) / max(n_alive, 1),
        "correlation_threshold": correlation_threshold,
        "p_threshold": p_threshold,
        "bonferroni_alpha": bonferroni_alpha,
        "n_tests": n_tests,
        "property_names": PROPERTY_NAMES,
        "property_summary": property_summary,
        "features": spearman_results,
    }

    return {
        "probe": probe_compat,
        "nonlinear": rf_result,
        "contrastive": contrastive_result,
        "union": {
            "n_alive": n_alive,
            "n_interpretable_spearman": len(spearman_interpretable_set),
            "n_interpretable_rf": len(rf_interpretable_set),
            "n_interpretable_contrastive": len(contrastive_interpretable_set),
            "n_interpretable_union": n_union,
            "interpretable_fraction_union": n_union / max(n_alive, 1),
            "spearman_only": sorted(sp_only),
            "rf_only": sorted(rf_only),
            "contrastive_only": sorted(ct_only),
            "all_three": sorted(all_three),
        },
    }


# Vocabulary projection: decode w vectors through model output projection

def project_features_to_vocab(
    sae: torch.nn.Module,
    model: torch.nn.Module,
    layer_idx: int,
    head_idx: int,
    top_k: int = 20,
) -> dict:
    """Project each feature's w vector through the GDN output projection to vocabulary space.

    The GDN retrieval mechanism: when query q attends to state S, it computes
        output = q^T @ S = sum_i c_i (q^T @ v_i) w_i

    So w_i is the "value" that gets returned. This value then passes through:
        1. The GDN layer's output projection (o_proj)
        2. The residual stream
        3. (Approximately) the unembedding matrix

    We project w_i through o_proj, then through the unembedding to get
    a vocabulary distribution. Top tokens tell us what information the
    feature stores for retrieval.

    Similarly, v_i is the "key" direction. We can project it through
    the GDN's key projection inverse to find what query directions
    would retrieve this feature.
    """
    from sae import MatrixSAE, BilinearMatrixSAE

    # Extract decoder vectors
    with torch.no_grad():
        if isinstance(sae, BilinearMatrixSAE):
            V_dec = sae.V_dec.float().cpu()  # (n_features, rank, d_k)
            W_dec = sae.W_dec.float().cpu()  # (n_features, rank, d_v)
        elif isinstance(sae, MatrixSAE):
            V_dec = sae.V.float().cpu()
            W_dec = sae.W.float().cpu()
        else:
            return {"error": "Vocabulary projection requires rank-1 SAE (MatrixSAE or BilinearMatrixSAE)"}

    # Squeeze rank dimension for rank-1
    if V_dec.ndim == 3 and V_dec.shape[1] == 1:
        V_dec = V_dec.squeeze(1)  # (n_features, d_k)
        W_dec = W_dec.squeeze(1)  # (n_features, d_v)
    elif V_dec.ndim == 3:
        # rank > 1: collapse each atom to its best rank-1 approximation
        # rather than silently discarding all but the first component.
        V_np = V_dec.numpy()
        W_np = W_dec.numpy()
        v_rank1: list[np.ndarray] = []
        w_rank1: list[np.ndarray] = []
        for v_factors, w_factors in zip(V_np, W_np):
            atom = np.einsum("rk,rv->kv", v_factors, w_factors)
            u, s, vt = np.linalg.svd(atom, full_matrices=False)
            top_sv = float(s[0]) if len(s) else 0.0
            scale = np.sqrt(top_sv) if top_sv > 0 else 0.0
            v_rank1.append((u[:, 0] * scale).astype(np.float32))
            w_rank1.append((vt[0] * scale).astype(np.float32))
        V_dec = torch.from_numpy(np.stack(v_rank1, axis=0))
        W_dec = torch.from_numpy(np.stack(w_rank1, axis=0))

    n_features = V_dec.shape[0]
    d_k = V_dec.shape[1]
    d_v = W_dec.shape[1]

    # Navigate to the GDN layer's output projection
    if hasattr(model, 'model'):
        layers = model.model.layers
        embed_tokens = model.model.embed_tokens
    else:
        layers = model.layers
        embed_tokens = model.embed_tokens

    gdn_layer = layers[layer_idx]
    linear_attn = gdn_layer.linear_attn

    # Qwen3.5 GDN uses fused projections:
    #   in_proj_qkv: (Q_dim + K_dim + V_dim, d_model) where Q=K=n_k_heads*head_k_dim, V=n_v_heads*head_v_dim
    #   out_proj: (d_model, V_dim) maps concatenated value heads back to residual stream
    n_k_heads = linear_attn.num_k_heads
    n_v_heads = linear_attn.num_v_heads
    head_k_dim = linear_attn.head_k_dim
    head_v_dim = linear_attn.head_v_dim

    out_proj_weight = linear_attn.out_proj.weight.float().cpu()  # (d_model, n_v_heads * head_v_dim)
    d_model = out_proj_weight.shape[0]
    n_heads = n_v_heads

    # Slice output projection for this head: columns [head_idx * d_v : (head_idx+1) * d_v]
    head_start = head_idx * head_v_dim
    head_end = head_start + head_v_dim
    o_proj_head = out_proj_weight[:, head_start:head_end]  # (d_model, d_v)

    # Get unembedding matrix (lm_head)
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.float().cpu()  # (vocab_size, d_model)
    else:
        unembed = embed_tokens.weight.float().cpu()  # tied embeddings

    vocab_size = unembed.shape[0]

    print(f"Projecting {n_features} features through out_proj ({d_v} -> {d_model}) "
          f"and unembed ({d_model} -> {vocab_size})")
    print(f"  n_k_heads={n_k_heads}, n_v_heads={n_v_heads}, head_k_dim={head_k_dim}, head_v_dim={head_v_dim}")

    # Project w vectors: w_i -> o_proj_head @ w_i -> residual direction -> unembed
    # w_residual = o_proj_head @ w_i, shape (d_model,)
    # logits_i = unembed @ w_residual, shape (vocab_size,)
    w_residual = W_dec @ o_proj_head.T  # (n_features, d_model)
    w_logits = w_residual @ unembed.T  # (n_features, vocab_size)

    # For v vectors: extract Q projection from the fused in_proj_qkv
    # in_proj_qkv rows: [0:Q_dim] = Q, [Q_dim:Q_dim+K_dim] = K, [Q_dim+K_dim:] = V
    in_proj_qkv = linear_attn.in_proj_qkv.weight.float().cpu()  # (Q+K+V, d_model)
    # Q projection for head_idx: rows [head_idx * head_k_dim : (head_idx+1) * head_k_dim]
    q_proj_head = in_proj_qkv[head_idx * head_k_dim : (head_idx + 1) * head_k_dim, :]  # (d_k, d_model)

    # v_i determines which queries retrieve this feature: q^T @ v_i
    # To find what tokens produce queries aligned with v_i:
    # q = q_proj_head @ x, so q^T @ v_i = x^T @ q_proj_head^T @ v_i
    # The "virtual input direction" for v_i is q_proj_head^T @ v_i
    v_input = V_dec @ q_proj_head  # (n_features, d_model)
    v_logits = v_input @ unembed.T  # (n_features, vocab_size)

    # For each feature, get top-k tokens
    feature_vocab = []
    for fi in range(n_features):
        # W (value/output) direction
        w_vals, w_idx = torch.topk(w_logits[fi], top_k)
        w_bottom_vals, w_bottom_idx = torch.topk(-w_logits[fi], top_k)

        # V (key/query) direction
        v_vals, v_idx = torch.topk(v_logits[fi], top_k)
        v_bottom_vals, v_bottom_idx = torch.topk(-v_logits[fi], top_k)

        feature_vocab.append({
            "feature_idx": fi,
            "w_top_tokens": w_idx.tolist(),
            "w_top_logits": w_vals.tolist(),
            "w_bottom_tokens": w_bottom_idx.tolist(),
            "w_bottom_logits": (-w_bottom_vals).tolist(),
            "v_top_tokens": v_idx.tolist(),
            "v_top_logits": v_vals.tolist(),
            "v_bottom_tokens": v_bottom_idx.tolist(),
            "v_bottom_logits": (-v_bottom_vals).tolist(),
            "w_residual_norm": float(w_residual[fi].norm()),
            "v_input_norm": float(v_input[fi].norm()),
        })

    return {
        "n_features": n_features,
        "d_model": d_model,
        "d_k": d_k,
        "d_v": d_v,
        "n_heads": n_heads,
        "vocab_size": vocab_size,
        "top_k": top_k,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "features": feature_vocab,
    }


def decode_token_ids(tokenizer, token_ids: list[int]) -> list[str]:
    """Decode a list of token IDs to their string representations."""
    return [tokenizer.decode([tid]) for tid in token_ids]


def format_vocab_projection(result: dict, tokenizer, n_features: int = 20) -> str:
    """Format vocabulary projection results into a readable report."""
    lines = []
    features = result["features"]

    # Sort by w_residual_norm (features with strongest output contribution)
    sorted_feats = sorted(features, key=lambda x: x["w_residual_norm"], reverse=True)

    for feat in sorted_feats[:n_features]:
        fi = feat["feature_idx"]
        w_tokens = decode_token_ids(tokenizer, feat["w_top_tokens"][:10])
        v_tokens = decode_token_ids(tokenizer, feat["v_top_tokens"][:10])
        w_bottom = decode_token_ids(tokenizer, feat["w_bottom_tokens"][:10])

        w_str = " | ".join(f"'{t}'" for t in w_tokens[:5])
        v_str = " | ".join(f"'{t}'" for t in v_tokens[:5])
        wb_str = " | ".join(f"'{t}'" for t in w_bottom[:5])

        lines.append(f"Feature {fi} (w_norm={feat['w_residual_norm']:.3f}, v_norm={feat['v_input_norm']:.3f})")
        lines.append(f"  RETRIEVES (w top):    {w_str}")
        lines.append(f"  RETRIEVES (w bottom): {wb_str}")
        lines.append(f"  QUERIED BY (v top):   {v_str}")
        lines.append("")

    return "\n".join(lines)
