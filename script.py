#!/usr/bin/env python3

import pandas as pd
import os
import argparse

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
    print("num_unique_seqs", end='\n', file=file)

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
        print(len(list(set(hor_table['sequence']))), end='\n', file=file)

