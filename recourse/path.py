import os

# infer core directories
pkg_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(pkg_dir)

data_dir = os.path.join(repo_dir, 'data')
results_dir = os.path.join(repo_dir, 'results')

# create directories that don't exist
for d in [data_dir, results_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
