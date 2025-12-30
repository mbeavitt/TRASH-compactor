import pandas as pd

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
