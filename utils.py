import pandas as pd

def import_hor_table(table_path):
    """
    Reads and cleans a standard HOR table output from TRASH (not repeat table)
    """

    input_table = pd.read_csv(table_path)

    input_table = input_table.drop("block.B.size.bp", axis='columns')
    input_table = input_table.drop("Unnamed: 0", axis='columns')
    input_table = input_table.drop("start_A", axis='columns')
    input_table = input_table.drop("start_B", axis='columns')
    input_table = input_table.drop("end_A", axis='columns')
    input_table = input_table.drop("end_B", axis='columns')
    input_table = input_table.drop("direction.1.para_2.perp.", axis='columns')
    input_table = input_table.drop("start.B.bp", axis='columns')
    input_table = input_table.drop("end.A.bp", axis='columns')
    input_table = input_table.drop("chrA", axis='columns')
    input_table = input_table.drop("chrB", axis='columns')
    input_table = input_table.rename(columns={"start.A.bp": "start"})
    input_table = input_table.rename(columns={"end.B.bp": "end"})
    input_table['start'] = input_table['start'] - 1 # convert to 0 based

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
