#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    prog='hor-analysis',
    description='Analyses output of TRASH')

parser.add_argument('input')

args = parser.parse_args()

input_table = pd.read_csv(args.input)

sns.set_theme()

n_bins=100

input_table = input_table.drop("block.B.size.bp", axis='columns')
input_table = input_table.drop("Unnamed: 0", axis='columns')
input_table = input_table.drop("end_A", axis='columns')
input_table = input_table.drop("end_B", axis='columns')
input_table = input_table.drop("direction.1.para_2.perp.", axis='columns')
input_table = input_table.drop("start.A.bp", axis='columns')
input_table = input_table.drop("start.B.bp", axis='columns')
input_table = input_table.drop("end.A.bp", axis='columns')
input_table = input_table.drop("end.B.bp", axis='columns')
input_table = input_table.drop("chrA", axis='columns')
input_table = input_table.drop("chrB", axis='columns')
input_table = input_table.rename(columns={"start_A": "start"})
input_table = input_table.rename(columns={"start_B": "end"})

input_table['normalised_pos'] = input_table['start'] / input_table['end'].iloc[-1]

sns.histplot(
    data=input_table,
    x="normalised_pos",  # or a normalised position column
    weights="block.size.in.units",
    bins=n_bins,
)

plt.xlabel("Normalised centromere position")
plt.ylabel("SNV per kbp")
plt.title("SNV per kbp across centromere")

plt.show()
