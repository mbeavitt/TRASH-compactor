#!/usr/bin/env python3
"""HORhouse - Interactive CLI for HOR analysis"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import Levenshtein

from . import utils, repeats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

REPEAT_SIZE = 178

class HORhouse:
    """Interactive HOR analysis tool"""

    def __init__(self, hor_table_path, fasta_path, output_dir="horhouse_output"):
        self.hor_table_path = hor_table_path
        self.fasta_path = fasta_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("Welcome to HORhouse")
        print(f"Output directory: {self.output_dir}")
        print("\nLoading data...")

        self.hor_table = utils.import_hor_table(hor_table_path)
        self.fasta = utils.read_fasta(fasta_path)
        self.seq = self.fasta[0]['sequence']

        print("Calculating metrics...")
        self._calculate_metrics()

        print("Computing global repeat colors...")
        self._compute_global_colors()

        self.current_selection = None

    def _calculate_metrics(self):
        """Calculate all HOR quality metrics"""

        # Extract block sequences
        self.hor_table['block_A_sequence'] = [
            utils.chunk_string(self.seq[sa:ea], REPEAT_SIZE)
            for sa, ea in zip(self.hor_table['block_A_start'], self.hor_table['block_A_end'])
        ]

        self.hor_table['block_B_sequence'] = [
            utils.chunk_string(self.seq[sb:eb], REPEAT_SIZE)
            for sb, eb in zip(self.hor_table['block_B_start'], self.hor_table['block_B_end'])
        ]

        # Calculate unique monomers
        self.hor_table['unique_monomers_A'] = self.hor_table['block_A_sequence'].apply(lambda x: len(set(x)))
        self.hor_table['unique_monomers_B'] = self.hor_table['block_B_sequence'].apply(lambda x: len(set(x)))

        # Block quality
        self.hor_table['block_A_quality'] = self.hor_table.apply(
            lambda row: row['block.size.in.units'] / row['unique_monomers_A'] if row['unique_monomers_A'] > 0 else 0,
            axis=1
        )
        self.hor_table['block_B_quality'] = self.hor_table.apply(
            lambda row: row['block.size.in.units'] / row['unique_monomers_B'] if row['unique_monomers_B'] > 0 else 0,
            axis=1
        )
        self.hor_table['avg_block_quality'] = (self.hor_table['block_A_quality'] + self.hor_table['block_B_quality']) / 2

        # Block overlap
        def calc_overlap(row):
            overlap_start = max(row['start_A_units'], row['start_B_units'])
            overlap_end = min(row['end_A_units'], row['end_B_units'])
            return max(0, overlap_end - overlap_start)

        self.hor_table['block_overlap_units'] = self.hor_table.apply(calc_overlap, axis=1)
        self.hor_table['block_offset_units'] = abs(self.hor_table['start_B_units'] - self.hor_table['start_A_units'])

        # Gap for overlap-free HORs
        def calc_gap(row):
            if row['block_overlap_units'] > 0:
                return 0
            return max(0, row['start_B_units'] - row['end_A_units'])

        self.hor_table['block_gap_units'] = self.hor_table.apply(calc_gap, axis=1)

        # Inter-block similarity
        def calc_similarity(row):
            block_A = row['block_A_sequence']
            block_B = row['block_B_sequence']
            if len(block_A) == 0 or len(block_B) == 0:
                return 0
            min_len = min(len(block_A), len(block_B))
            distances = [Levenshtein.distance(block_A[i], block_B[i]) for i in range(min_len)]
            return sum(distances) / len(distances) if distances else 0

        self.hor_table['position_wise_distance'] = self.hor_table.apply(calc_similarity, axis=1)
        self.hor_table['hor_similarity'] = 1 / (1 + self.hor_table['position_wise_distance'])

        # Calculate internal diversity within each block
        def calc_internal_diversity(row):
            """Average pairwise distance within a block"""
            block_A = row['block_A_sequence']
            block_B = row['block_B_sequence']

            # Internal diversity for block A
            if len(block_A) > 1:
                distances_A = []
                for i in range(len(block_A)):
                    for j in range(i+1, len(block_A)):
                        distances_A.append(Levenshtein.distance(block_A[i], block_A[j]))
                diversity_A = sum(distances_A) / len(distances_A) if distances_A else 0
            else:
                diversity_A = 0

            # Internal diversity for block B
            if len(block_B) > 1:
                distances_B = []
                for i in range(len(block_B)):
                    for j in range(i+1, len(block_B)):
                        distances_B.append(Levenshtein.distance(block_B[i], block_B[j]))
                diversity_B = sum(distances_B) / len(distances_B) if distances_B else 0
            else:
                diversity_B = 0

            # Return max internal diversity (most diverse block)
            return max(diversity_A, diversity_B)

        self.hor_table['internal_diversity'] = self.hor_table.apply(calc_internal_diversity, axis=1)

        # Diversity-similarity score: high internal diversity + high inter-block similarity + large size
        # High score = large HORs with complex internal structure but blocks match each other well
        self.hor_table['diversity_similarity_score'] = (
            self.hor_table['block.size.in.units'] *
            self.hor_table['internal_diversity'] *
            self.hor_table['hor_similarity']
        )

        print(f"Loaded {len(self.hor_table)} HORs")
        overlap_free = self.hor_table[self.hor_table['block_overlap_units'] == 0]
        print(f"{len(overlap_free)} overlap-free HORs ({100*len(overlap_free)/len(self.hor_table):.1f}%)")

    def _compute_global_colors(self):
        """Compute global color mapping for all unique repeats across dataset"""

        # Collect all unique repeats across all HORs
        all_sequences = set()
        for _, row in self.hor_table.iterrows():
            all_sequences.update(row['block_A_sequence'])
            all_sequences.update(row['block_B_sequence'])

        self.global_repeats = sorted(list(all_sequences))
        n_global = len(self.global_repeats)

        print(f"  Found {n_global} unique repeat sequences across all HORs")

        # Build global distance matrix
        print(f"  Building distance matrix ({n_global} unique repeats)...")
        global_distance_matrix = np.zeros((n_global, n_global))

        for i in range(n_global):
            for j in range(i+1, n_global):
                dist = Levenshtein.distance(self.global_repeats[i], self.global_repeats[j])
                global_distance_matrix[i, j] = dist
                global_distance_matrix[j, i] = dist

        # Project to 3D RGB space using MDS
        print(f"  Projecting to 3D color space...")
        mds = MDS(n_components=3, metric='precomputed', n_init=1, init='random', random_state=42)
        rgb_coords = mds.fit_transform(global_distance_matrix)

        # Normalize to [0, 1]
        scaler = MinMaxScaler()
        rgb_coords = scaler.fit_transform(rgb_coords)
        rgb_coords = np.clip(rgb_coords, 0, 1)

        # Create global color map: sequence -> RGB color
        self.global_color_map = {seq: rgb_coords[i] for i, seq in enumerate(self.global_repeats)}

        print(f"  Complete")

    def visualize_hor(self, hor_idx, save_png=True):
        """Visualize a single HOR and optionally save as PNG"""

        row = self.hor_table.iloc[hor_idx]

        # Get block sequences
        block_A_repeats = row['block_A_sequence']
        block_B_repeats = row['block_B_sequence']

        # Get consensus and distances for this HOR (for legend only)
        all_repeats = block_A_repeats + block_B_repeats
        unique_repeats = sorted(list(set(all_repeats)))
        cons = repeats.get_consensus(unique_repeats)
        dist_list = repeats.hamming_list(unique_repeats, cons)
        unique_repeats_dist_map = {rep: dist for rep, dist in zip(unique_repeats, dist_list)}

        # Create RGB arrays using global colors
        rgb_array_A = np.ones((1, len(block_A_repeats), 3))
        rgb_array_B = np.ones((1, len(block_B_repeats), 3))

        for idx, seq in enumerate(block_A_repeats):
            rgb_array_A[0, idx] = self.global_color_map[seq]

        for idx, seq in enumerate(block_B_repeats):
            rgb_array_B[0, idx] = self.global_color_map[seq]

        # Create figure with 4 subplots (Block A + bp, Block B + bp)
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 3, 1], hspace=0.3)

        # Block A plot
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(rgb_array_A, aspect='auto', interpolation='nearest')
        ax1.set_yticks([])
        ax1.set_ylabel('Block A', fontsize=10)
        ax1.set_title(f'HOR {hor_idx} | Size: {int(row["block.size.in.units"])} units | '
                     f'Similarity: {row["hor_similarity"]:.3f} | Gap: {int(row["block_gap_units"])} units')
        ax1.set_xticks([])

        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1,
                         facecolor=self.global_color_map[seq],
                         label=f'Repeat {i} (dist: {unique_repeats_dist_map[seq]})')
            for i, seq in enumerate(unique_repeats[:10])
        ]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 1), fontsize=8)

        # Block A bp annotations
        ax2 = fig.add_subplot(gs[1])
        ax2.set_xlim(0, len(block_A_repeats))
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        start_A = int(row['block_A_start'])
        n_markers_A = min(10, len(block_A_repeats))
        for i in range(n_markers_A):
            idx = int(i * len(block_A_repeats) / n_markers_A)
            bp_pos = start_A + idx * REPEAT_SIZE
            ax2.text(idx, 0.5, f'{bp_pos:,}', rotation=45, ha='right', va='bottom', fontsize=8)
            ax2.axvline(idx, ymin=0.3, ymax=0.5, color='black', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('Repeat index', fontsize=9)

        # Block B plot
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(rgb_array_B, aspect='auto', interpolation='nearest')
        ax3.set_yticks([])
        ax3.set_ylabel('Block B', fontsize=10)
        ax3.set_xticks([])

        # Block B bp annotations
        ax4 = fig.add_subplot(gs[3])
        ax4.set_xlim(0, len(block_B_repeats))
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        start_B = int(row['block_B_start'])
        n_markers_B = min(10, len(block_B_repeats))
        for i in range(n_markers_B):
            idx = int(i * len(block_B_repeats) / n_markers_B)
            bp_pos = start_B + idx * REPEAT_SIZE
            ax4.text(idx, 0.5, f'{bp_pos:,}', rotation=45, ha='right', va='bottom', fontsize=8)
            ax4.axvline(idx, ymin=0.3, ymax=0.5, color='black', alpha=0.3, linewidth=0.5)
        ax4.set_xlabel('Repeat index', fontsize=9)

        plt.tight_layout()

        if save_png:
            output_file = self.output_dir / f"hor_{hor_idx}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
        else:
            plt.show()

        plt.close()

    def run(self):
        """Main interactive loop"""

        while True:
            print(f"\n{'='*70}")
            print("HORhouse - What would you like to do?")
            print(f"{'='*70}")
            print("  1. Browse overlap-free HORs (sort & filter)")
            print("  2. Visualize specific HOR by index")
            print("  3. Export current selection to CSV")
            print("  4. Show statistics")
            print("  q. Quit")
            print(f"{'='*70}")

            choice = input("\nYour choice: ").strip()

            if choice == 'q':
                print("\nGoodbye from HORhouse")
                break
            elif choice == '1':
                self._browse_hors()
            elif choice == '2':
                self._visualize_by_index()
            elif choice == '3':
                self._export_selection()
            elif choice == '4':
                self._show_statistics()
            else:
                print("Invalid choice")

    def _browse_hors(self):
        """Browse and filter HORs"""

        overlap_free = self.hor_table[self.hor_table['block_overlap_units'] == 0].copy()

        if len(overlap_free) == 0:
            print("\nNo overlap-free HORs found")
            return

        print(f"\n{'='*70}")
        print("Sort by:")
        print("  1. Block size (largest first)")
        print("  2. Block similarity (most similar first)")
        print("  3. Block quality (best organization)")
        print("  4. Block offset (largest offset)")
        print("  5. Diversity-similarity score (diverse internally, similar between blocks)")
        print("  6. Custom filter (similarity threshold)")
        print(f"{'='*70}")

        sort_choice = input("\nSort by: ").strip()

        if sort_choice == '1':
            sorted_hors = overlap_free.sort_values(by='block.size.in.units', ascending=False)
            title = "Largest HORs"
        elif sort_choice == '2':
            sorted_hors = overlap_free.sort_values(by='hor_similarity', ascending=False)
            title = "Most Similar HORs"
        elif sort_choice == '3':
            sorted_hors = overlap_free.sort_values(by='avg_block_quality', ascending=False)
            title = "Best Quality HORs"
        elif sort_choice == '4':
            sorted_hors = overlap_free.sort_values(by='block_offset_units', ascending=False)
            title = "Largest Offset HORs"
        elif sort_choice == '5':
            sorted_hors = overlap_free.sort_values(by='diversity_similarity_score', ascending=False)
            title = "High Diversity + High Inter-Block Similarity"
        elif sort_choice == '6':
            threshold = float(input("Minimum similarity (0-1, e.g., 0.5): "))
            sorted_hors = overlap_free[overlap_free['hor_similarity'] >= threshold].sort_values(
                by='block.size.in.units', ascending=False
            )
            title = f"HORs with similarity >= {threshold}"
        else:
            print("Invalid choice")
            return

        if len(sorted_hors) == 0:
            print("\nNo HORs match this criteria")
            return

        self.current_selection = sorted_hors

        # Display top results
        print(f"\n{title}")
        display_cols = ['block.size.in.units', 'block_offset_units', 'block_gap_units',
                       'avg_block_quality', 'hor_similarity', 'internal_diversity',
                       'diversity_similarity_score']
        print(sorted_hors[display_cols].head(20))

        # Ask to visualize
        viz = input("\nVisualize top N HORs? (enter number or 'n' to skip): ").strip()
        if viz.lower() != 'n':
            try:
                n = int(viz)
                print(f"\nGenerating {n} visualizations...")
                for idx in sorted_hors.head(n).index:
                    self.visualize_hor(idx)
                print(f"\nDone! Check {self.output_dir}/")
            except Exception as e:
                print(f"Error: {e}")

    def _visualize_by_index(self):
        """Visualize specific HOR by index"""
        idx_str = input("\nEnter HOR index: ").strip()
        try:
            idx = int(idx_str)
            if idx < 0 or idx >= len(self.hor_table):
                print(f"Index out of range (0-{len(self.hor_table)-1})")
                return
            print(f"\nGenerating visualization for HOR {idx}...")
            self.visualize_hor(idx)
        except Exception as e:
            print(f"Error: {e}")

    def _export_selection(self):
        """Export current selection to CSV"""
        if self.current_selection is None:
            print("\nNo selection to export. Browse HORs first.")
            return

        output_file = self.output_dir / "selected_hors.csv"
        export_cols = ['block.size.in.units', 'block_offset_units', 'block_gap_units',
                      'unique_monomers_A', 'unique_monomers_B', 'avg_block_quality',
                      'hor_similarity', 'internal_diversity', 'diversity_similarity_score',
                      'start_A_units', 'end_A_units', 'start_B_units', 'end_B_units']
        self.current_selection[export_cols].to_csv(output_file)
        print(f"\nExported {len(self.current_selection)} HORs to {output_file}")

    def _show_statistics(self):
        """Show HOR statistics"""
        overlap_free = self.hor_table[self.hor_table['block_overlap_units'] == 0]

        print(f"\n{'='*70}")
        print("HOR Statistics")
        print(f"{'='*70}")
        print(f"Total HORs: {len(self.hor_table)}")
        print(f"Overlap-free: {len(overlap_free)} ({100*len(overlap_free)/len(self.hor_table):.1f}%)")
        print(f"\nSimilarity (overlap-free):")
        print(f"  Mean: {overlap_free['hor_similarity'].mean():.3f}")
        print(f"  Median: {overlap_free['hor_similarity'].median():.3f}")
        print(f"  Max: {overlap_free['hor_similarity'].max():.3f}")
        print(f"\nHORs with similarity > 0.5: {len(overlap_free[overlap_free['hor_similarity'] > 0.5])}")
        print(f"HORs with similarity > 0.7: {len(overlap_free[overlap_free['hor_similarity'] > 0.7])}")
        print(f"{'='*70}")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        prog='horhouse',
        description='HORhouse - Interactive HOR analysis tool')

    parser.add_argument('--input', required=True, help='HOR table CSV file')
    parser.add_argument('--fasta', required=True, help='FASTA file')
    parser.add_argument('--output', default='horhouse_output', help='Output directory (default: horhouse_output)')

    args = parser.parse_args()

    app = HORhouse(args.input, args.fasta, args.output)
    app.run()


if __name__ == '__main__':
    main()
