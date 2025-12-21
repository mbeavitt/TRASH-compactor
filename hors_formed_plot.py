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

hor_table = pd.read_csv(args.input)

sns.set_theme()

n_bins=100
hor_table['normalised_pos'] = hor_table['start'] / hor_table['end'].iloc[-1]

sns.histplot(
    data=hor_table,
    x="normalised_pos",  # or a normalised position column
    weights="hors_formed_tot_rep_normalised",
    bins=n_bins,
)

plt.xlabel("Normalised centromere position")
plt.ylabel("HOR formation sum")
plt.title("HOR formation across centromere")

plt.show()
