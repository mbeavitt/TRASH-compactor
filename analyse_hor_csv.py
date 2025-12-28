#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import utils
import repeats
import Levenshtein

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

parser = argparse.ArgumentParser(
    prog='hor-analysis',
    description='Analyses output of TRASH')

parser.add_argument('--input')
parser.add_argument('--fasta')

args = parser.parse_args()

hor_table = utils.import_hor_table(args.input) # pd dataframe
fasta = utils.read_fasta(args.fasta) # list of dicts

seq = fasta[0]['sequence']

hor_table['hor_sequence'] = [
    utils.chunk_string(seq[s:e], 178) for s, e in zip(hor_table['start'], hor_table['end'])
]


