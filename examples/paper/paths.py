from pathlib import Path

repo_dir = Path(__file__).absolute().parent.parent

# directory containing paper related info
paper_dir = repo_dir / 'paper/'

# directory containing data files
data_dir = paper_dir / 'data/'

# directory containing results
results_dir = paper_dir / 'results/'

# create directories that don't exist
for d in [data_dir, results_dir]:
    d.mkdir(exist_ok = True)
