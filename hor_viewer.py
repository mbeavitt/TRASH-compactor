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
hor_start = int(hor_table.iloc[hor_idx,:]['start']) # including
hor_end = int(hor_table.iloc[hor_idx,:]['end']) # up to but not including
hor_unit_size = int(hor_table.iloc[hor_idx,:]['block.size.in.units'])

fasta = utils.read_fasta(args.fasta)
whole_hor = fasta[0]['sequence'][hor_start * REPEAT_SIZE: hor_end * REPEAT_SIZE]
split_hor = utils.chunk_string(whole_hor, hor_unit_size * REPEAT_SIZE)
split_hor_repeats = [utils.chunk_string(i, REPEAT_SIZE) for i in split_hor]

unique_repeats = sorted(list(set([i for j in split_hor_repeats for i in j])))
unique_repeats_id_map = {rep: rep_id for rep_id, rep in enumerate(unique_repeats)}

cons = repeats.get_consensus(unique_repeats)
dist_list = repeats.hamming_list(unique_repeats, cons)

unique_repeats_dist_map = {rep: dist for rep, dist in zip(unique_repeats, dist_list)}
print(unique_repeats_dist_map)

# Create atomic representation of the whole HOR
hor_atomic = []
for hor_unit in split_hor_repeats:
    for position_in_unit, repeat_seq in enumerate(hor_unit):
        hor_atomic.append({
            'position_in_unit': position_in_unit,
            'repeat_id': unique_repeats_id_map[repeat_seq],
            'distance': unique_repeats_dist_map[repeat_seq]
        })

print("\nAtomic HOR representation:")
print(hor_atomic)

# Verify assumption: within each HOR unit, no adjacent repeats have distance > 3
max_adjacent_distance_in_unit = 0
violations_in_units = []

for unit_idx, hor_unit in enumerate(split_hor_repeats):
    for pos in range(len(hor_unit) - 1):
        seq1 = hor_unit[pos]
        seq2 = hor_unit[pos + 1]

        dist = Levenshtein.distance(seq1, seq2)

        if dist > max_adjacent_distance_in_unit:
            max_adjacent_distance_in_unit = dist

        if dist > 3:
            violations_in_units.append({
                'unit_idx': unit_idx,
                'position_in_unit': pos,
                'distance': dist
            })

print(f"\nMax distance between adjacent repeats within a unit: {max_adjacent_distance_in_unit}")
if violations_in_units:
    print(f"Found {len(violations_in_units)} violations where adjacent distance > 3 within a unit:")
    for v in violations_in_units[:10]:
        print(f"  Unit {v['unit_idx']}, position {v['position_in_unit']}: distance = {v['distance']}")
else:
    print("✓ Assumption verified: all adjacent repeats within units have distance <= 3")

# Visualize the atomic HOR representation
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

# Create RGB array for the visualization
# Each repeat gets a color based on its sequence, not its position
rgb_array = np.zeros((1, len(hor_atomic), 3))

for idx, repeat in enumerate(hor_atomic):
    repeat_id = repeat['repeat_id']
    rgb_array[0, idx] = rgb_coords[repeat_id]

# Plot
fig, ax = plt.subplots(figsize=(20, 2))
ax.imshow(rgb_array, aspect='auto', interpolation='nearest')
ax.set_yticks([])
ax.set_xlabel('Repeat index in HOR')
ax.set_title(f'HOR Atomic Representation (HOR index: {hor_idx}, unit size: {hor_unit_size})')

# Add legend showing unique repeat colors
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=rgb_coords[i],
                                  label=f'Repeat {i} (dist: {dist_list[i]})')
                   for i in range(min(10, n_unique))]  # Show first 10 to avoid clutter
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)

plt.tight_layout()
plt.show()

#for i in range(len(hor_table)):
#    hor_start = hor_table.iloc[i,:]['start'] 
#    hor_end = hor_table.iloc[i,:]['end'] 
#    hor_size = hor_end - hor_start
#    hor_unit_size = hor_table.iloc[i,:]['block.size.in.units']
#    print(hor_start, hor_end, hor_size, hor_unit_size)
