# HORhouse

Interactive analysis and visualization tool for Higher Order Repeats (HORs) in genomic sequences.

## Installation

```bash
pip install .
```

## Usage

```bash
horhouse --input HORs.csv --fasta genome.fa --repeats repeats.csv --chromosome Chr1
```

### Required Arguments

- `--input`: HOR table CSV file
- `--fasta`: Reference genome FASTA file
- `--repeats`: Repeats table CSV file with columns: seq_name, start, end, sequence

### Optional Arguments

- `--chromosome`: Chromosome name (required for multi-sequence FASTA)
- `--output`: Output directory (default: horhouse_output)
- `--cache-only`: Calculate and cache results without entering interactive mode
- `--cache-file`: Load specific cache file instead of automatic detection
- `--color-method`: Color projection method: umap (default) or mds

## Features

The program allows the user to interactively filter a table of higher order repeats (HORs), outputting a UMAP/MDS plot
of the repeats and a cache by default, and optionally plots a selected HOR or top n HORs sorted by some metric.

The cache itself is the table of HORs with interesting metrics added and is useful in its own right.

## Cache Management

Caches are stored in `.horhouse_cache/` in the current directory with filenames based on the FASTA file:

```
.horhouse_cache/ANGE-B-10.ragtag_scaffolds_f0f44ad5.csv
```

Load a specific cache:
```bash
horhouse --input ... --fasta ... --repeats ... --cache-file path/to/cache.csv
```

## Performance

For large datasets (700K+ HORs), initial computation takes 1-3 minutes. Cached runs load in ~20 seconds,
which is not ideal and might be improved later

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib
- Levenshtein
- umap-learn (optional, falls back to MDS)
