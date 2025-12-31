import pandas as pd
import hashlib
import os

def import_hor_table(table_path):
    """
    Reads and cleans a standard HOR table output from TRASH (not repeat table)
    """

    input_table = pd.read_csv(table_path)

    if (input_table['block.B.size.bp'] != input_table['block.A.size.bp']).any():
        print("ERROR: assumption that block B size == block A size violated")
    input_table = input_table.drop("block.B.size.bp", axis='columns')
    input_table = input_table.drop("Unnamed: 0", axis='columns')
    input_table = input_table.drop("direction.1.para_2.perp.", axis='columns')
    input_table = input_table.drop("chrA", axis='columns')
    input_table = input_table.drop("chrB", axis='columns')
    # Keep block A and B positions - HOR is a pairwise construct of these two blocks
    input_table = input_table.rename(columns={"start.A.bp": "start_A_bp"})
    input_table = input_table.rename(columns={"end.A.bp": "end_A_bp"})
    input_table = input_table.rename(columns={"start.B.bp": "start_B_bp"})
    input_table = input_table.rename(columns={"end.B.bp": "end_B_bp"})
    input_table = input_table.rename(columns={"block.A.size.bp": "block_size"})
    # Convert bp coordinates to 0-based indexing
    input_table['block_A_start'] = input_table['start_A_bp'] - 1
    input_table['block_A_end'] = input_table['end_A_bp']
    input_table['block_B_start'] = input_table['start_B_bp'] - 1
    input_table['block_B_end'] = input_table['end_B_bp']
    # Keep unit-based coordinates for reference
    input_table = input_table.rename(columns={"start_A": "start_A_units"})
    input_table = input_table.rename(columns={"end_A": "end_A_units"})
    input_table = input_table.rename(columns={"start_B": "start_B_units"})
    input_table = input_table.rename(columns={"end_B": "end_B_units"})
    # HOR is block A + block B (not the sequence in between)
    input_table['hor_size_bp'] = 2 * input_table['block_size']

    return input_table

def read_fasta(fasta_path):
    "very basic fasta reader"

    with open(fasta_path, "r") as file:
        contents = file.read()

    contents = contents.split('>')
    seqs = []

    for seq in contents:
        if seq:
            header = seq.split('\n')[0].strip()
            sequence = ''.join(seq.split('\n')[1:]).strip()
            seqs.append({"header": header, "sequence": sequence})

    return seqs

def chunk_string(string, chunk_size):
    output = []
    if len(string) % chunk_size != 0:
        n_chunks = int((len(string) // chunk_size) + 1)
    else:
        n_chunks = int(len(string) / chunk_size)
    for i in range(n_chunks):
        output.append(string[i*chunk_size:i*chunk_size+chunk_size])

    return output

def import_repeats_table(csv_path):
    """
    Load and normalize repeats table from CSV.
    Handles both test_data.csv and all.repeats.csv formats.
    Returns DataFrame with standardized columns: seq_name, start, end, width, sequence
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    # test_data.csv has: seqID, sequence
    # all.repeats.csv has: seq.name, seq
    if 'seqID' in df.columns:
        df = df.rename(columns={'seqID': 'seq_name'})
    elif 'seq.name' in df.columns:
        df = df.rename(columns={'seq.name': 'seq_name'})

    if 'seq' in df.columns and 'sequence' not in df.columns:
        df = df.rename(columns={'seq': 'sequence'})

    # Convert seq_name to categorical for faster comparisons
    df['seq_name'] = df['seq_name'].astype('category')

    return df

def get_repeats_in_range(repeats_df, seq_name, start, end, return_positions=False):
    """
    Get all repeats that overlap with a specific coordinate range.

    A repeat overlaps if it starts before the block ends AND ends after the block starts.
    This uses standard genomic interval overlap logic.

    Args:
        return_positions: If True, returns (sequences, positions) tuple where positions
                         are the start coordinates of each repeat

    Returns:
        List of sequences sorted by start position, or (sequences, positions) tuple
    """
    # Use standard overlap logic for genomic intervals
    mask = (
        (repeats_df['seq_name'] == seq_name) &
        (repeats_df['start'] < end) &      # Repeat starts before block ends
        (repeats_df['end'] > start)        # Repeat ends after block starts
    )

    filtered = repeats_df[mask].sort_values('start')

    if return_positions:
        return list(filtered['sequence']), list(filtered['start'])
    else:
        return list(filtered['sequence'])

def select_sequence_by_name(fasta_list, chromosome_name):
    """
    Select a sequence from FASTA list by chromosome name.
    Returns the sequence string, or None if not found.
    """
    for entry in fasta_list:
        header = entry['header']
        # Match exact chromosome name or if it's part of header
        if header == chromosome_name or header.startswith(chromosome_name):
            return entry['sequence']

    return None


def calculate_input_hash(hor_table_path, fasta_path, repeats_table_path):
    """
    Calculate a hash of the input files to use for caching.
    Returns a hex string hash.
    """
    hasher = hashlib.sha256()

    # Hash each input file
    for filepath in [hor_table_path, fasta_path, repeats_table_path]:
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            while chunk := f.read(8192):
                hasher.update(chunk)

    return hasher.hexdigest()


def get_cache_path(input_hash, output_dir):
    """
    Get the cache file path for a given input hash.
    Cache is stored in the output directory.
    """
    cache_dir = os.path.join(output_dir, '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'hor_table_{input_hash}.csv')


def load_cache(cache_path):
    """
    Load cached HOR table from CSV file.
    Returns the DataFrame or None if cache doesn't exist.
    """
    if os.path.exists(cache_path):
        try:
            # Read the CSV, handling list columns
            df = pd.read_csv(cache_path)

            # Convert string representations of lists back to actual lists
            # These columns contain lists that need to be parsed
            list_columns = ['block_A_sequence', 'block_B_sequence', 'block_A_positions', 'block_B_positions']
            for col in list_columns:
                if col in df.columns:
                    df[col] = df[col].apply(eval)

            return df
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
    return None


def save_cache(hor_table, cache_path):
    """
    Save HOR table to cache as CSV file.
    """
    try:
        hor_table.to_csv(cache_path, index=False)

        # Report cache size
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"Cache saved: {size_mb:.2f} MB at {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
