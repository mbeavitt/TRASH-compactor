#!/usr/bin/env python3

import pandas as pd
import os
import argparse
import Levenshtein

CEN178_CONS = "AGTATAAGAACTTAAACCGCAACCCGATCTTAAAAGCCTAAGTAGTGTTTCCTTGTTAGAAGACACAAAGCCAAAGACTCATATGGACTTTGGCTACACCATGAAAGCTTTGAGAAGCAAGAAGAAGGTTGGTTAGTGTTTTGGAGTCGAATATGACTTGATGTCATGTGTATGATTG"

def get_consensus(repeats):
    """
    From a list of identically sized repeats, finds the consensus sequence
    """

    repeat_cons_vals = [{"A": 0, "T": 0, "C": 0, "G": 0} for i in range(178)]

    for repeat in repeats:
        for idx, base in enumerate(repeat):
            repeat_cons_vals[idx][base] += 1

    consensus = [max(i, key=lambda key: i[key]) for i in repeat_cons_vals]
    return ''.join(consensus)

def hamming_dist_from_cons(repeats):
    """
    takes a list of repeat sequences and finds the total hamming distance
    from the consensus. Warning, needs modifications to work on TRASH output
    from real files (i.e. not perfect 178bp sequences)
    """

    consensus = get_consensus(repeats)
    distances = [Levenshtein.distance(repeat, CEN178_CONS) for repeat in repeats]
    return sum(distances)

parser = argparse.ArgumentParser(
                    prog='hor-analysis',
                    description='Analyses output of TRASH',
                    epilog='text at bottom of help - todo')

parser.add_argument('input_table')

args = parser.parse_args()

'''
example = args.input[0]

hor_table = pd.read_csv(example)
print(hor_table.head())
print(list(hor_table.columns))
'''

with open("summary.tsv", "w") as file:
    print("run_id", end='\t', file=file)
    print("num_seqs", end='\t', file=file)
    print("num_unique_seqs", end='\t', file=file)
    print("total hamming distance", end='\n', file=file)

    input_table = pd.read_csv(args.input_table, sep='\t', header=None)
    file_paths = list(input_table.iloc[:, 1])

    for index, line in input_table.iterrows():
        run_id = line[0]
        path = line[1]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find input TRASH table: {path}")

        hor_table = pd.read_csv(path)

        print(run_id, end='\t', file=file)
        print(len(hor_table['sequence']), end='\t', file=file)
        print(len(list(set(hor_table['sequence']))), end='\t', file=file)
        print(hamming_dist_from_cons(hor_table['sequence']), end='\n', file=file)
