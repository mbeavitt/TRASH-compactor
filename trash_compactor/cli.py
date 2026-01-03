#!/usr/bin/env python3
"""TRASH-compactor - Interactive CLI for HOR analysis"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import Levenshtein

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import MDS
    HAS_UMAP = False
    print("Warning: UMAP not installed, falling back to MDS. Install with: pip install umap-learn")

from . import utils, repeats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class TRASHCompactor:
    """Interactive HOR analysis tool"""

    def __init__(self, hor_table_path, fasta_path, repeats_table_path, chromosome=None, output_dir="trash_compactor_output", cache_only=False, cache_dir=None, color_method='umap'):
        self.hor_table_path = hor_table_path
        self.fasta_path = fasta_path
        self.repeats_table_path = repeats_table_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_only = cache_only
        self.color_method = color_method.lower()

        print("Welcome to TRASH-compactor")
        print(f"Output directory: {self.output_dir}")

        # Determine cache path
        print("\nChecking cache...")
        # Calculate cache hash based on input files
        import time
        t0 = time.time()
        print("  Calculating input hash...")
        input_hash = utils.calculate_input_hash(hor_table_path, fasta_path, repeats_table_path)
        t1 = time.time()
        print(f"  Hash calculated in {t1-t0:.2f}s: {input_hash[:16]}...")
        cache_path = utils.get_cache_path(input_hash, cache_dir=cache_dir, fasta_path=fasta_path, chromosome=chromosome)
        print(f"  Cache location: {cache_path}")

        # Try to load from cache
        print("  Loading cache (if exists)...")
        t0 = time.time()
        cached_table = utils.load_cache(cache_path)
        t1 = time.time()
        use_cache = cached_table is not None

        if use_cache:
            print(f"  Cache loaded successfully in {t1-t0:.2f}s")
            self.hor_table = cached_table
        else:
            print("  No cache found, will calculate from scratch...")

        print("\nLoading data...")

        if not use_cache:
            self.hor_table = utils.import_hor_table(hor_table_path)

        self.repeats_table = utils.import_repeats_table(repeats_table_path)
        self.fasta = utils.read_fasta(fasta_path)

        # Handle multi-sequence FASTA
        if len(self.fasta) == 1:
            # Single sequence - use it directly
            self.seq = self.fasta[0]['sequence']
            self.seq_name = self.fasta[0]['header']
            print(f"Using single sequence: {self.seq_name}")
        else:
            # Multi-sequence FASTA
            if chromosome is None:
                # Try to extract chromosome from HOR table
                if 'chrA' in self.hor_table.columns and len(self.hor_table) > 0:
                    chr_full_name = self.hor_table['chrA'].iloc[0]
                    # Extract Chr1, Chr2, etc. from "ANGE-B-10.ragtag_scaffolds.fa_Chr1"
                    if '_Chr' in chr_full_name:
                        chromosome = chr_full_name.split('_Chr')[-1].split('_')[0]
                        chromosome = 'Chr' + chromosome
                        print(f"Auto-detected chromosome from HOR table: {chromosome}")

            if chromosome is None:
                raise ValueError("Multi-sequence FASTA requires --chromosome parameter")

            self.seq_name = chromosome
            self.seq = utils.select_sequence_by_name(self.fasta, chromosome)
            if self.seq is None:
                available = [f['header'] for f in self.fasta]
                raise ValueError(f"Chromosome '{chromosome}' not found in FASTA. Available: {available}")
            print(f"Selected chromosome: {self.seq_name}")

        # Pre-filter repeats table to only the chromosome we're analyzing (performance optimization)
        if not use_cache:
            original_count = len(self.repeats_table)
            self.repeats_table = self.repeats_table[
                self.repeats_table['seq_name'] == self.seq_name
            ].copy()
            print(f"Loaded {len(self.repeats_table)} repeats from repeats table")
            if original_count > len(self.repeats_table):
                print(f"  (Filtered to {self.seq_name} from {original_count} total repeats)")

            # OPTIMIZATION: Pre-sort and convert to numpy arrays for O(log n) binary search
            print("  Preparing repeat data structures for fast interval queries...")
            self.repeats_table = self.repeats_table.sort_values('start').reset_index(drop=True)

            # Convert to numpy for fast binary search-based lookups (50-100x faster)
            self.repeat_starts = self.repeats_table['start'].values
            self.repeat_ends = self.repeats_table['end'].values
            self.repeat_seqs = self.repeats_table['sequence'].values
            self.repeat_positions = self.repeats_table['start'].values

            print("\nCalculating metrics...")
            self._calculate_metrics()

            print("\nSaving cache...")
            utils.save_cache(self.hor_table, cache_path)

            # If cache-only mode, we're done - skip color computation and interactive setup
            if self.cache_only:
                print("Cache created successfully!")
                return
        else:
            # Still need to filter repeats table for visualization and on-demand sequence extraction
            original_count = len(self.repeats_table)
            self.repeats_table = self.repeats_table[
                self.repeats_table['seq_name'] == self.seq_name
            ].copy()
            print(f"Loaded {len(self.repeats_table)} repeats from repeats table (for visualization)")

            # IMPORTANT: Set up repeat data structures for on-demand sequence extraction
            print("  Preparing repeat data structures for on-demand sequence extraction...")
            self.repeats_table = self.repeats_table.sort_values('start').reset_index(drop=True)
            self.repeat_starts = self.repeats_table['start'].values
            self.repeat_ends = self.repeats_table['end'].values
            self.repeat_seqs = self.repeats_table['sequence'].values
            self.repeat_positions = self.repeats_table['start'].values

        print("\nComputing global repeat colors...")
        self._compute_global_colors()

        self.current_selection = None

    def _get_block_sequences(self, block_start, block_end):
        """
        Extract sequences for a block on-demand from repeats table.
        Returns (sequences, positions) tuple.
        """
        return utils.get_repeats_in_range_fast(
            self.repeat_starts,
            self.repeat_ends,
            self.repeat_seqs,
            self.repeat_positions,
            block_start,
            block_end
        )

    def _calculate_metrics(self):
        """Calculate all HOR quality metrics"""

        # Extract block sequences and positions from repeats table
        # NOTE: We keep sequences in temporary lists for metric calculation,
        # but DON'T store them in DataFrame to save ~400-500 MB
        block_A_sequences = []
        block_B_sequences = []
        block_A_positions = []
        block_B_positions = []

        total_hors = len(self.hor_table)
        progress_interval = max(1000, total_hors // 100)  # Print every 1000 HORs or 1%, whichever is larger

        for hor_num, (idx, row) in enumerate(self.hor_table.iterrows(), 1):
            # Get repeats for block A with positions (using fast binary search)
            repeats_A, pos_A = self._get_block_sequences(
                row['block_A_start'],
                row['block_A_end']
            )

            # Get repeats for block B with positions (using fast binary search)
            repeats_B, pos_B = self._get_block_sequences(
                row['block_B_start'],
                row['block_B_end']
            )

            block_A_sequences.append(repeats_A)
            block_B_sequences.append(repeats_B)
            block_A_positions.append(pos_A)
            block_B_positions.append(pos_B)

            # Print progress
            if hor_num % progress_interval == 0 or hor_num == total_hors:
                percent = (hor_num / total_hors) * 100
                print(f"  Processed {hor_num:,}/{total_hors:,} HORs ({percent:.1f}%)")

        print("  Storing positions (not sequences) to save memory...")
        # MEMORY OPTIMIZATION: Only store positions, not sequences (~400-500 MB savings)
        self.hor_table['block_A_positions'] = block_A_positions
        self.hor_table['block_B_positions'] = block_B_positions

        print("  Calculating repeat counts...")
        # Add columns for actual repeat counts found (from temporary lists)
        self.hor_table['actual_repeats_A'] = [len(seqs) for seqs in block_A_sequences]
        self.hor_table['actual_repeats_B'] = [len(seqs) for seqs in block_B_sequences]

        print("  Flagging mismatches...")
        # Flag HORs with repeat count mismatches
        self.hor_table['repeat_mismatch_A'] = (
            self.hor_table['actual_repeats_A'] != self.hor_table['block.size.in.units']
        )
        self.hor_table['repeat_mismatch_B'] = (
            self.hor_table['actual_repeats_B'] != self.hor_table['block.size.in.units']
        )

        # Summary of mismatches
        mismatch_A_count = self.hor_table['repeat_mismatch_A'].sum()
        mismatch_B_count = self.hor_table['repeat_mismatch_B'].sum()
        if mismatch_A_count > 0 or mismatch_B_count > 0:
            print(f"Note: {mismatch_A_count} HORs have block A repeat count mismatch, {mismatch_B_count} have block B mismatch")
            print(f"      (actual vs expected from HOR table - see 'repeat_mismatch_A/B' columns)")

        print("  Calculating unique monomers...")
        # Calculate unique monomers (from temporary lists)
        self.hor_table['unique_monomers_A'] = [len(set(seqs)) for seqs in block_A_sequences]
        self.hor_table['unique_monomers_B'] = [len(set(seqs)) for seqs in block_B_sequences]

        print("  Calculating block quality metrics...")
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

        print("  Calculating block overlaps and gaps...")
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

        print("  Pre-computing Levenshtein distance matrix for unique sequences...")
        # Collect all unique sequences (from temporary lists)
        all_sequences = set()
        for sequences in block_A_sequences:
            all_sequences.update(sequences)
        for sequences in block_B_sequences:
            all_sequences.update(sequences)

        unique_sequences = list(all_sequences)
        n_unique = len(unique_sequences)
        print(f"    Found {n_unique} unique sequences")

        # Store unique sequences for later use (e.g., global color computation)
        self.unique_sequences = unique_sequences

        # Pre-compute distance matrix directly in NumPy (avoids 1.2GB intermediate dict)
        print(f"    Computing {n_unique * (n_unique - 1) // 2:,} pairwise distances directly to NumPy matrix...")
        seq_to_id = {seq: i for i, seq in enumerate(unique_sequences)}
        distance_matrix = np.zeros((n_unique, n_unique), dtype=np.float32)

        for i in range(n_unique):
            for j in range(i + 1, n_unique):
                seq_i, seq_j = unique_sequences[i], unique_sequences[j]
                dist = Levenshtein.distance(seq_i, seq_j)
                # Symmetric matrix - set both [i,j] and [j,i]
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist


        print("  Calculating inter-block similarity (using matrix lookups)...")
        # Inter-block similarity (using temporary lists)
        position_wise_distances = []
        for block_A, block_B in zip(block_A_sequences, block_B_sequences):
            if len(block_A) == 0 or len(block_B) == 0:
                position_wise_distances.append(0)
                continue
            min_len = min(len(block_A), len(block_B))
            # Use pre-allocated array and matrix indexing for performance
            distances = np.empty(min_len, dtype=np.float32)
            for i in range(min_len):
                seq_a_id = seq_to_id[block_A[i]]
                seq_b_id = seq_to_id[block_B[i]]
                distances[i] = distance_matrix[seq_a_id, seq_b_id]
            position_wise_distances.append(distances.mean() if min_len > 0 else 0)

        self.hor_table['position_wise_distance'] = position_wise_distances
        self.hor_table['hor_similarity'] = 1 / (1 + self.hor_table['position_wise_distance'])


        print("  Calculating internal diversity (using matrix lookups)...")
        # Calculate internal diversity within each block (using temporary lists)
        internal_diversities = []
        for block_A, block_B in zip(block_A_sequences, block_B_sequences):
            # Internal diversity for block A
            if len(block_A) > 1:
                # Pre-allocate numpy array for performance (avoids 496M list.append calls)
                n_pairs_A = len(block_A) * (len(block_A) - 1) // 2
                distances_A = np.empty(n_pairs_A, dtype=np.float32)
                idx = 0
                for i in range(len(block_A)):
                    seq_i_id = seq_to_id[block_A[i]]
                    for j in range(i+1, len(block_A)):
                        seq_j_id = seq_to_id[block_A[j]]
                        # Use matrix indexing (407M dict.get calls eliminated)
                        distances_A[idx] = distance_matrix[seq_i_id, seq_j_id]
                        idx += 1
                diversity_A = distances_A.mean()
            else:
                diversity_A = 0

            # Internal diversity for block B
            if len(block_B) > 1:
                # Pre-allocate numpy array for performance
                n_pairs_B = len(block_B) * (len(block_B) - 1) // 2
                distances_B = np.empty(n_pairs_B, dtype=np.float32)
                idx = 0
                for i in range(len(block_B)):
                    seq_i_id = seq_to_id[block_B[i]]
                    for j in range(i+1, len(block_B)):
                        seq_j_id = seq_to_id[block_B[j]]
                        # Use matrix indexing (407M dict.get calls eliminated)
                        distances_B[idx] = distance_matrix[seq_i_id, seq_j_id]
                        idx += 1
                diversity_B = distances_B.mean()
            else:
                diversity_B = 0

            # Append max internal diversity (most diverse block)
            internal_diversities.append(max(diversity_A, diversity_B))

        self.hor_table['internal_diversity'] = internal_diversities


        print("  Calculating diversity-similarity scores...")
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

        # Use unique sequences computed during metric calculation
        # (avoids re-extracting sequences from DataFrame which no longer stores them)
        if hasattr(self, 'unique_sequences'):
            self.global_repeats = sorted(self.unique_sequences)
        else:
            # Fallback: extract sequences from positions (for cached data)
            print("  Extracting unique sequences from cached positions...")
            all_sequences = set()

            # Check if we have sequence columns (legacy cache)
            if 'block_A_sequence' in self.hor_table.columns:
                for _, row in self.hor_table.iterrows():
                    all_sequences.update(row['block_A_sequence'])
                    all_sequences.update(row['block_B_sequence'])
            else:
                # New cache format: extract sequences on-demand using positions
                for _, row in self.hor_table.iterrows():
                    seqs_A, _ = self._get_block_sequences(row['block_A_start'], row['block_A_end'])
                    seqs_B, _ = self._get_block_sequences(row['block_B_start'], row['block_B_end'])
                    all_sequences.update(seqs_A)
                    all_sequences.update(seqs_B)

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

        # Project to 3D RGB space using UMAP or MDS
        if self.color_method == 'umap' and HAS_UMAP:
            print(f"  Projecting to 3D color space using UMAP...")
            # UMAP is much faster than MDS and often gives better embeddings
            import warnings
            warnings.filterwarnings('ignore', message='using precomputed metric')
            warnings.filterwarnings('ignore', message='n_jobs value')
            warnings.filterwarnings('ignore', message='Graph is not fully connected')

            reducer = UMAP(
                n_components=3,
                metric='precomputed',
                n_neighbors=min(15, n_global - 1),
                min_dist=0.1,
                n_jobs=1,  # Explicit to avoid warning
                init='random',  # Avoid spectral init issues
                random_state=42
            )
            rgb_coords = reducer.fit_transform(global_distance_matrix)
        else:
            if self.color_method == 'umap' and not HAS_UMAP:
                print(f"  UMAP not available, falling back to MDS. Install with: pip install umap-learn")
            print(f"  Projecting to 3D color space using MDS...")
            mds = MDS(n_components=3, metric='precomputed', n_init=1, init='random', random_state=42)
            rgb_coords = mds.fit_transform(global_distance_matrix)

        # Normalize to [0, 1] for valid RGB values
        scaler = MinMaxScaler()
        rgb_coords = scaler.fit_transform(rgb_coords)
        rgb_coords = np.clip(rgb_coords, 0, 1)

        # Create 3D visualization using the SCALED RGB coordinates
        # Color each point by its actual RGB value so you can verify the mapping
        if self.color_method == 'umap':
            print(f"  Creating 3D UMAP visualization (colored by actual RGB values)...")
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            # Color points by their actual RGB values (position = color in this plot)
            scatter = ax.scatter(rgb_coords[:, 0], rgb_coords[:, 1], rgb_coords[:, 2],
                               c=rgb_coords, s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
            ax.set_xlabel('Red (UMAP 1)')
            ax.set_ylabel('Green (UMAP 2)')
            ax.set_zlabel('Blue (UMAP 3)')
            ax.set_title(f'3D UMAP Color Space: {n_global} Unique Sequences\n(Points colored by their RGB values)')

            umap_plot_path = self.output_dir / 'umap_3d_sequences.png'
            plt.savefig(umap_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved 3D UMAP plot to {umap_plot_path}")

        # Create global color map: sequence -> RGB color
        self.global_color_map = {seq: rgb_coords[i] for i, seq in enumerate(self.global_repeats)}

        print(f"  Complete")

    def visualize_hor(self, hor_idx, save_png=True):
        """Visualize a single HOR and optionally save as PNG"""

        row = self.hor_table.iloc[hor_idx]

        # Get block sequences and positions
        # Extract sequences on-demand (not stored in DataFrame to save memory)
        block_A_repeats, block_A_pos = self._get_block_sequences(
            row['block_A_start'],
            row['block_A_end']
        )
        block_B_repeats, block_B_pos = self._get_block_sequences(
            row['block_B_start'],
            row['block_B_end']
        )

        # Get expected count and block boundaries
        expected_count = int(row['block.size.in.units'])
        block_A_start = row['block_A_start']
        block_A_end = row['block_A_end']
        block_B_start = row['block_B_start']
        block_B_end = row['block_B_end']

        # Get consensus and distances for this HOR (for legend only)
        all_repeats = block_A_repeats + block_B_repeats
        unique_repeats = sorted(list(set(all_repeats)))
        if len(unique_repeats) > 0:
            cons = repeats.get_consensus(unique_repeats)
            dist_list = repeats.hamming_list(unique_repeats, cons)
            unique_repeats_dist_map = {rep: dist for rep, dist in zip(unique_repeats, dist_list)}
        else:
            unique_repeats_dist_map = {}

        # Build RGB arrays with actual repeat positions and sizes
        # Use block size in bp as the width (more granular than just repeat count)
        block_A_size_bp = block_A_end - block_A_start
        block_B_size_bp = block_B_end - block_B_start

        # Create pixel arrays (1 pixel per bp for precision)
        MISSING_COLOR = [0.8, 0.8, 0.8]  # Light gray for missing repeats
        rgb_array_A = np.ones((1, block_A_size_bp, 3)) * MISSING_COLOR
        rgb_array_B = np.ones((1, block_B_size_bp, 3)) * MISSING_COLOR

        # Fill in block A repeats at their actual positions with actual sizes
        for seq, pos in zip(block_A_repeats, block_A_pos):
            # Calculate pixel range for this repeat
            start_px = max(0, pos - block_A_start)
            # Get repeat size from repeats table
            repeat_mask = (
                (self.repeats_table['seq_name'] == self.seq_name) &
                (self.repeats_table['start'] == pos)
            )
            if repeat_mask.sum() > 0:
                repeat_size = self.repeats_table[repeat_mask].iloc[0]['width']
                end_px = min(block_A_size_bp, start_px + repeat_size)

                if seq in self.global_color_map:
                    rgb_array_A[0, start_px:end_px] = self.global_color_map[seq]

        # Fill in block B repeats at their actual positions with actual sizes
        for seq, pos in zip(block_B_repeats, block_B_pos):
            start_px = max(0, pos - block_B_start)
            repeat_mask = (
                (self.repeats_table['seq_name'] == self.seq_name) &
                (self.repeats_table['start'] == pos)
            )
            if repeat_mask.sum() > 0:
                repeat_size = self.repeats_table[repeat_mask].iloc[0]['width']
                end_px = min(block_B_size_bp, start_px + repeat_size)

                if seq in self.global_color_map:
                    rgb_array_B[0, start_px:end_px] = self.global_color_map[seq]

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
        ax2.set_xlim(0, block_A_size_bp)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        start_A = int(row['block_A_start'])

        n_markers_A = min(10, expected_count + 1)
        for i in range(n_markers_A):
            px = int(i * block_A_size_bp / (n_markers_A - 1)) if n_markers_A > 1 else 0
            bp_pos = start_A + px
            ax2.text(px, 0.5, f'{bp_pos:,}', rotation=45, ha='right', va='bottom', fontsize=8)
            ax2.axvline(px, ymin=0.3, ymax=0.5, color='black', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('Position (bp)', fontsize=9)

        # Block B plot
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(rgb_array_B, aspect='auto', interpolation='nearest')
        ax3.set_yticks([])
        ax3.set_ylabel('Block B', fontsize=10)
        ax3.set_xticks([])

        # Block B bp annotations
        ax4 = fig.add_subplot(gs[3])
        ax4.set_xlim(0, block_B_size_bp)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        start_B = int(row['block_B_start'])

        n_markers_B = min(10, expected_count + 1)
        for i in range(n_markers_B):
            px = int(i * block_B_size_bp / (n_markers_B - 1)) if n_markers_B > 1 else 0
            bp_pos = start_B + px
            ax4.text(px, 0.5, f'{bp_pos:,}', rotation=45, ha='right', va='bottom', fontsize=8)
            ax4.axvline(px, ymin=0.3, ymax=0.5, color='black', alpha=0.3, linewidth=0.5)
        ax4.set_xlabel('Position (bp)', fontsize=9)

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
        prog='trash-compactor',
        description='TRASH-compactor - Interactive HOR analysis tool for TRASH output')

    parser.add_argument('--input', required=True, help='HOR table CSV file')
    parser.add_argument('--fasta', required=True, help='FASTA file')
    parser.add_argument('--repeats', required=True, help='Repeats table CSV file (e.g., test_data.csv or all.repeats.csv)')
    parser.add_argument('--chromosome', help='Chromosome name (required for multi-sequence FASTA, optional for single-sequence)')
    parser.add_argument('--output', default='trash_compactor_output', help='Output directory (default: trash_compactor_output)')
    parser.add_argument('--cache-only', action='store_true', help='Only calculate and cache results, do not enter interactive mode')
    parser.add_argument('--cache-dir', help='Directory for cache files (default: .trash_compactor_cache in current directory)')
    parser.add_argument('--color-method', choices=['umap', 'mds'], default='umap',
                       help='Method for projecting sequences to RGB color space (default: umap)')

    args = parser.parse_args()

    app = TRASHCompactor(args.input, args.fasta, args.repeats, args.chromosome, args.output,
                         cache_only=args.cache_only, cache_dir=args.cache_dir, color_method=args.color_method)

    if not args.cache_only:
        app.run()
    else:
        print("\nCache-only mode: Exiting without entering interactive CLI.")


if __name__ == '__main__':
    main()
