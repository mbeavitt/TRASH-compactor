#!/usr/bin/env python3

import argparse
import utils
import repeats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import Levenshtein

REPEAT_SIZE = 178

parser = argparse.ArgumentParser(
        prog='hor_viewer',
        description='plots HORs'
)

parser.add_argument('-i', '--input', help="input HOR table", required=True)
parser.add_argument('-f', '--fasta', help="Fasta file from which HOR table was generated", required=True)
parser.add_argument('-p', '--position', help="index of the requested HOR", required=True)

args = parser.parse_args()

hor_table = utils.import_hor_table(args.input)

hor_idx = int(args.position)
print("Displaying: ", hor_idx)
# HOR is a pairwise construct - extract block A and block B
# Use unit-based coordinates (already converted to bp)
start_A = int(hor_table.iloc[hor_idx,:]['block_A_start'])
end_A = int(hor_table.iloc[hor_idx,:]['block_A_end'])
start_B = int(hor_table.iloc[hor_idx,:]['block_B_start'])
end_B = int(hor_table.iloc[hor_idx,:]['block_B_end'])
hor_unit_size = int(hor_table.iloc[hor_idx,:]['block.size.in.units'])

fasta = utils.read_fasta(args.fasta)
# Extract block A and block B separately for pairwise comparison
block_A = fasta[0]['sequence'][start_A:end_A]
block_B = fasta[0]['sequence'][start_B:end_B]

print(f"Block A length: {len(block_A)/178} units")
print(f"Block B length: {len(block_B)/178} units")

# Split each block into repeats
block_A_repeats = utils.chunk_string(block_A, REPEAT_SIZE)
block_B_repeats = utils.chunk_string(block_B, REPEAT_SIZE)

# Get all unique repeats from both blocks
all_repeats = block_A_repeats + block_B_repeats
unique_repeats = sorted(list(set(all_repeats)))
unique_repeats_id_map = {rep: rep_id for rep_id, rep in enumerate(unique_repeats)}

cons = repeats.get_consensus(unique_repeats)
dist_list = repeats.hamming_list(unique_repeats, cons)

unique_repeats_dist_map = {rep: dist for rep, dist in zip(unique_repeats, dist_list)}
print(unique_repeats_dist_map)

# Create atomic representation for block A
block_A_atomic = []
for position, repeat_seq in enumerate(block_A_repeats):
    block_A_atomic.append({
        'position': position,
        'repeat_id': unique_repeats_id_map[repeat_seq],
        'distance': unique_repeats_dist_map[repeat_seq]
    })

# Create atomic representation for block B
block_B_atomic = []
for position, repeat_seq in enumerate(block_B_repeats):
    block_B_atomic.append({
        'position': position,
        'repeat_id': unique_repeats_id_map[repeat_seq],
        'distance': unique_repeats_dist_map[repeat_seq]
    })

print("\nBlock A atomic representation:")
print(block_A_atomic)
print("\nBlock B atomic representation:")
print(block_B_atomic)

# Visualize the pairwise HOR comparison
# Project unique repeat sequences into RGB color space using MDS
n_unique = len(unique_repeats)
distance_matrix = np.zeros((n_unique, n_unique))

# Compute pairwise distances between all unique repeats
for i in range(n_unique):
    for j in range(i+1, n_unique):
        dist = Levenshtein.distance(unique_repeats[i], unique_repeats[j])
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

# Apply MDS to project sequences into 3D RGB space
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
rgb_coords = mds.fit_transform(distance_matrix)

# Normalize to [0, 1] range for RGB values
scaler = MinMaxScaler()
rgb_coords = scaler.fit_transform(rgb_coords)
# Clip to ensure values are exactly within [0, 1] (avoid floating point errors)
rgb_coords = np.clip(rgb_coords, 0, 1)

# Create RGB arrays for both blocks
# Pad to make them the same length for visualization
max_len = max(len(block_A_atomic), len(block_B_atomic))
rgb_array = np.ones((2, max_len, 3))  # White background for padding

# Fill in block A (top row)
for idx, repeat in enumerate(block_A_atomic):
    repeat_id = repeat['repeat_id']
    rgb_array[0, idx] = rgb_coords[repeat_id]

# Fill in block B (bottom row)
for idx, repeat in enumerate(block_B_atomic):
    repeat_id = repeat['repeat_id']
    rgb_array[1, idx] = rgb_coords[repeat_id]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 5), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

# Main plot
ax1.imshow(rgb_array, aspect='auto', interpolation='nearest')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Block A', 'Block B'])
ax1.set_ylabel('Block')
ax1.set_title(f'HOR Pairwise Comparison (HOR index: {hor_idx}, unit size: {hor_unit_size})')

# Add legend showing unique repeat colors
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=rgb_coords[i],
                                  label=f'Repeat {i} (dist: {dist_list[i]})')
                   for i in range(min(10, n_unique))]  # Show first 10 to avoid clutter
ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)

# BP position annotations (secondary axis)
ax2.set_xlim(0, max_len)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Add BP position markers for block A
n_markers = min(10, len(block_A_repeats))
for i in range(n_markers):
    idx = int(i * len(block_A_repeats) / n_markers)
    bp_pos = start_A + idx * REPEAT_SIZE
    ax2.text(idx, 0.7, f'{bp_pos:,}', rotation=45, ha='right', va='bottom',
             fontsize=8, color='blue', alpha=0.7)
    ax2.axvline(idx, ymin=0.6, ymax=0.8, color='blue', alpha=0.3, linewidth=0.5)

# Add BP position markers for block B
n_markers_b = min(10, len(block_B_repeats))
for i in range(n_markers_b):
    idx = int(i * len(block_B_repeats) / n_markers_b)
    bp_pos = start_B + idx * REPEAT_SIZE
    ax2.text(idx, 0.3, f'{bp_pos:,}', rotation=45, ha='right', va='top',
             fontsize=8, color='red', alpha=0.7)
    ax2.axvline(idx, ymin=0.2, ymax=0.4, color='red', alpha=0.3, linewidth=0.5)

# Labels
ax2.text(-0.5, 0.7, 'Block A bp:', ha='right', va='center', fontsize=8, color='blue')
ax2.text(-0.5, 0.3, 'Block B bp:', ha='right', va='center', fontsize=8, color='red')
ax2.set_xlabel('Repeat index within block')

plt.tight_layout()
plt.show()

#for i in range(len(hor_table)):
#    hor_start = hor_table.iloc[i,:]['start'] 
#    hor_end = hor_table.iloc[i,:]['end'] 
#    hor_size = hor_end - hor_start
#    hor_unit_size = hor_table.iloc[i,:]['block.size.in.units']
#    print(hor_start, hor_end, hor_size, hor_unit_size)
