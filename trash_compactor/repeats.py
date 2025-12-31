import Levenshtein

def normalize_sequence_length(seq, target_length=178):
    """
    Normalize sequence to target length (default 178bp for consensus calculation).
    - Sequences shorter than target are padded with 'N'
    - Sequences longer than target are truncated
    - Used for consensus calculation (coloring only)
    """
    if len(seq) < target_length:
        # Pad with 'N' characters
        return seq + 'N' * (target_length - len(seq))
    elif len(seq) > target_length:
        # Truncate to target length
        return seq[:target_length]
    else:
        return seq

def get_consensus(repeats):
    """
    From a list of repeats, finds the consensus sequence.
    Normalizes all sequences to 178bp before calculating consensus.
    """
    if not repeats:
        return 'N' * 178

    # Normalize all sequences to 178bp
    normalized_repeats = [normalize_sequence_length(r) for r in repeats]

    repeat_cons_vals = [{"A": 0, "T": 0, "C": 0, "G": 0, "N": 0} for _ in range(178)]

    for repeat in normalized_repeats:
        for idx, base in enumerate(repeat):
            if base in repeat_cons_vals[idx]:
                repeat_cons_vals[idx][base] += 1

    consensus = [max(i, key=lambda key: i[key]) for i in repeat_cons_vals]
    return ''.join(consensus)

def hamming_dist_from_cons(repeats, consensus):
    """
    takes a list of repeat sequences and finds the total hamming distance
    from the consensus. Warning, needs modifications to work on TRASH output
    from real files (i.e. not perfect 178bp sequences)
    """

    distances = [Levenshtein.distance(repeat, consensus) for repeat in repeats]
    return sum(distances)

def hamming_list(repeats, consensus):
    """
    from a list of repeat sequences, return each repeat's distance from the consensus
    """

    return [Levenshtein.distance(repeat, consensus) for repeat in repeats]
