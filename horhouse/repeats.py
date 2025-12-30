import Levenshtein

def get_consensus(repeats):
    """
    From a list of identically sized repeats, finds the consensus sequence
    """

    repeat_cons_vals = [{"A": 0, "T": 0, "C": 0, "G": 0} for _ in range(178)]

    for repeat in repeats:
        for idx, base in enumerate(repeat):
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
