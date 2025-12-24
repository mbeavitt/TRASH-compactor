#!/usr/bin/env python3
 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import repeats
import Levenshtein

CEN178_CONS = "AGTATAAGAACTTAAACCGCAACCCGATCTTAAAAGCCTAAGTAGTGTTTCCTTGTTAGAAGACACAAAGCCAAAGACTCATATGGACTTTGGCTACACCATGAAAGCTTTGAGAAGCAAGAAGAAGGTTGGTTAGTGTTTTGGAGTCGAATATGACTTGATGTCATGTGTATGATTG"

parser = argparse.ArgumentParser(
    prog='hor-analysis',
    description='Analyses output of TRASH')

parser.add_argument('input_table')

args = parser.parse_args()

summary_rows = []

input_table = pd.read_csv(args.input_table, sep='\t', header=None)
file_paths = list(input_table.iloc[:, 1])

for index, line in input_table.iterrows():
    run_id = line[0]
    path = line[1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find input TRASH table: {path}")

    hor_table = pd.read_csv(path)
    repeats_list = hor_table['sequence']

    consensus = repeats.get_consensus(repeats_list)

    summary_rows.append({
        "run_id": run_id,
        "num_repeats": len(repeats_list),
        "num_unique_repeats": len(list(set(repeats_list))),
        "local_consensus_distance_sum": repeats.hamming_dist_from_cons(repeats_list, consensus),
        "local_consensus_sequence": consensus,
        "local_consensus_dist_from_seed": Levenshtein.distance(CEN178_CONS, consensus),
        "mean_hors_per_repeat": hor_table['hors_formed_count'].mean()
    })

summary_table = pd.DataFrame(summary_rows)

print(summary_table)
summary_table.to_csv("summary.tsv", sep='\t', index=False)

# todo:
# - histogram (distribution) of average HOR values per run
# - above but scaled to number of repeats
# - distribution of distances from local consensus (overall array diversity)
# - distribution of distances from seed sequence (rate of change in diversity over time)
# - Some kind of 
