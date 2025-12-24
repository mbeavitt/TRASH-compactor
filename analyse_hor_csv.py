#!/usr/bin/env python3
 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import utils
import Levenshtein

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

parser = argparse.ArgumentParser(
    prog='hor-analysis',
    description='Analyses output of TRASH')

parser.add_argument('input')

args = parser.parse_args()

input_table = utils.import_hor_table(args.input)

print(input_table.head())
print(len(input_table))

num_hors = len(input_table)
