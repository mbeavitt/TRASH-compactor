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

# Extract block A and block B separately
hor_table['block_A_sequence'] = [
    utils.chunk_string(seq[sa:ea], 178)
    for sa, ea in zip(hor_table['block_A_start'], hor_table['block_A_end'])
]

hor_table['block_B_sequence'] = [
    utils.chunk_string(seq[sb:eb], 178)
    for sb, eb in zip(hor_table['block_B_start'], hor_table['block_B_end'])
]

# Calculate unique monomers in each block
hor_table['unique_monomers_A'] = hor_table['block_A_sequence'].apply(lambda x: len(set(x)))
hor_table['unique_monomers_B'] = hor_table['block_B_sequence'].apply(lambda x: len(set(x)))

# Calculate per-block quality (lower unique monomers = higher quality)
hor_table['block_A_quality'] = hor_table.apply(
    lambda row: row['block.size.in.units'] / row['unique_monomers_A'] if row['unique_monomers_A'] > 0 else 0,
    axis=1
)
hor_table['block_B_quality'] = hor_table.apply(
    lambda row: row['block.size.in.units'] / row['unique_monomers_B'] if row['unique_monomers_B'] > 0 else 0,
    axis=1
)

# Calculate block overlap (in repeat units)
def calc_block_overlap(row):
    # Convert bp positions to unit positions for overlap calculation
    a_start = row['start_A_units']
    a_end = row['end_A_units']
    b_start = row['start_B_units']
    b_end = row['end_B_units']

    # Calculate overlap
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    overlap = max(0, overlap_end - overlap_start)

    return overlap

hor_table['block_overlap_units'] = hor_table.apply(calc_block_overlap, axis=1)
hor_table['block_offset_units'] = abs(hor_table['start_B_units'] - hor_table['start_A_units'])

# Calculate inter-block distance (similarity between blocks)
# For overlapping blocks, compare position-by-position
def calc_inter_block_similarity(row):
    block_A = row['block_A_sequence']
    block_B = row['block_B_sequence']

    if len(block_A) == 0 or len(block_B) == 0:
        return 0

    overlap = int(row['block_overlap_units'])
    offset = int(row['block_offset_units'])

    if overlap > 0 and offset < len(block_A) and offset < len(block_B):
        # Blocks overlap - compare the overlapping region position-by-position
        distances = []
        for i in range(min(overlap, len(block_A) - offset, len(block_B))):
            a_idx = offset + i
            b_idx = i
            if a_idx < len(block_A) and b_idx < len(block_B):
                distances.append(Levenshtein.distance(block_A[a_idx], block_B[b_idx]))
        return sum(distances) / len(distances) if distances else 0
    else:
        # Non-overlapping - compare corresponding positions up to min length
        min_len = min(len(block_A), len(block_B))
        distances = [Levenshtein.distance(block_A[i], block_B[i]) for i in range(min_len)]
        return sum(distances) / len(distances) if distances else 0

hor_table['position_wise_distance'] = hor_table.apply(calc_inter_block_similarity, axis=1)

# Calculate HOR quality score
# Lower position-wise distance = better HOR (blocks are more similar)
# Higher block quality = better organization within each block
# We want LOW distance and HIGH quality
hor_table['hor_similarity'] = 1 / (1 + hor_table['position_wise_distance'])
hor_table['avg_block_quality'] = (hor_table['block_A_quality'] + hor_table['block_B_quality']) / 2

# Multiple scoring metrics for different purposes
# 1. Complexity score: emphasizes size and offset (periodic patterns)
hor_table['complexity_score'] = (
    hor_table['block.size.in.units'] *
    hor_table['avg_block_quality'] *
    hor_table['hor_similarity'] *
    hor_table['block_offset_units']
)

# 2. Quality score: emphasizes similarity and organization (best HOR structure)
# Size * similarity^2 * quality (weights similarity heavily)
hor_table['quality_score'] = (
    hor_table['block.size.in.units'] *
    (hor_table['hor_similarity'] ** 2) *
    hor_table['avg_block_quality']
)

# 3. Combined score: balanced view of all factors
# Geometric mean to balance size, quality, and similarity
hor_table['combined_score'] = (
    (hor_table['block.size.in.units'] ** 0.33) *
    (hor_table['avg_block_quality'] ** 0.33) *
    (hor_table['hor_similarity'] ** 0.33) *
    hor_table['block_offset_units']
)

# Filter for overlap-free HORs (blocks don't overlap)
overlap_free = hor_table[hor_table['block_overlap_units'] == 0].copy()

# Also calculate gap between blocks for overlap-free HORs
def calc_block_gap(row):
    a_end = row['end_A_units']
    b_start = row['start_B_units']
    return max(0, b_start - a_end)

overlap_free['block_gap_units'] = overlap_free.apply(calc_block_gap, axis=1)

# Display columns of interest
display_cols = ['block.size.in.units', 'block_offset_units', 'block_gap_units',
                'unique_monomers_A', 'unique_monomers_B',
                'avg_block_quality', 'position_wise_distance', 'hor_similarity',
                'quality_score', 'combined_score', 'complexity_score']

print(f"\n=== OVERLAP-FREE HORs ONLY ===")
print(f"Total HORs: {len(hor_table)}")
print(f"Overlap-free HORs: {len(overlap_free)} ({100*len(overlap_free)/len(hor_table):.1f}%)")
print(f"Overlapping HORs: {len(hor_table) - len(overlap_free)}")

if len(overlap_free) > 0:
    print(f"\n=== Overlap-free HOR Statistics ===")
    print(f"Similarity (hor_similarity):")
    print(f"  Mean: {overlap_free['hor_similarity'].mean():.3f}")
    print(f"  Median: {overlap_free['hor_similarity'].median():.3f}")
    print(f"  Max: {overlap_free['hor_similarity'].max():.3f}")
    print(f"\nPosition-wise distance:")
    print(f"  Mean: {overlap_free['position_wise_distance'].mean():.3f}")
    print(f"  Median: {overlap_free['position_wise_distance'].median():.3f}")
    print(f"  Min: {overlap_free['position_wise_distance'].min():.3f}")
    print(f"\nHORs with similarity > 0.5: {len(overlap_free[overlap_free['hor_similarity'] > 0.5])}")
    print(f"HORs with similarity > 0.7: {len(overlap_free[overlap_free['hor_similarity'] > 0.7])}")
    print(f"HORs with position_wise_distance < 1.0: {len(overlap_free[overlap_free['position_wise_distance'] < 1.0])}")

if len(overlap_free) > 0:
    print("\n=== Top overlap-free HORs by QUALITY SCORE ===")
    print("(Best combination of size, similarity, and organization - RECOMMENDED)")
    print(overlap_free.sort_values(by="quality_score", ascending=False)[display_cols].head(10))

    print("\n\n=== Top overlap-free HORs by COMBINED SCORE ===")
    print("(Balanced view using geometric mean)")
    print(overlap_free.sort_values(by="combined_score", ascending=False)[display_cols].head(10))

    print("\n\n=== Top overlap-free HORs by complexity score ===")
    print("(Emphasizes large size and offset)")
    print(overlap_free.sort_values(by="complexity_score", ascending=False)[display_cols].head(10))

    print("\n\n=== Top overlap-free HORs by block similarity ===")
    print("(Separate blocks with lowest position-wise distance)")
    print(overlap_free.sort_values(by="position_wise_distance", ascending=True)[display_cols].head(10))

    print("\n\n=== Largest overlap-free HORs by size ===")
    print(overlap_free.sort_values(by="block.size.in.units", ascending=False)[display_cols].head(10))
else:
    print("\nNo overlap-free HORs found in this dataset!")
