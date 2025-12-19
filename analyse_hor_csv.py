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

def hamming_dist_from_cons(repeats, consensus):
    """
    takes a list of repeat sequences and finds the total hamming distance
    from the consensus. Warning, needs modifications to work on TRASH output
    from real files (i.e. not perfect 178bp sequences)
    """

    distances = [Levenshtein.distance(repeat, CEN178_CONS) for repeat in repeats]
    return sum(distances)

parser = argparse.ArgumentParser(
                    prog='hor-analysis',
                    description='Analyses output of TRASH',
                    epilog='text at bottom of help - todo')

parser.add_argument('input_table')

args = parser.parse_args()

summary_table = pd.DataFrame(columns = ["run_id", "num_seqs", "num_unique_seqs", "total_hamming_distance", consensus])

input_table = pd.read_csv(args.input_table, sep='\t', header=None)
file_paths = list(input_table.iloc[:, 1])

for index, line in input_table.iterrows():
    run_id = line[0]
    path = line[1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find input TRASH table: {path}")

    hor_table = pd.read_csv(path)
    repeats_list = hor_table['sequence']

    consensus = get_consensus(repeats_list)

    summary_table_row = pd.DataFrame([[
        run_id,
        len(repeats_list),
        len(list(set(repeats_list))),
        hamming_dist_from_cons(repeats_list, consensus),
        consensus
    ]], columns=summary_table.columns)

    summary_table = pd.concat([summary_table, summary_table_row], ignore_index=True)

print(summary_table)
