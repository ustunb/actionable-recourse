import numpy as np
import pandas as pd
from paper.paths import *
pd.options.mode.chained_assignment = None

# input vars
data_name = 'german'
raw_data_file = '%s/%s/%s_raw.csv' % (data_dir, data_name, data_name)
processed_file = '%s/%s_processed.csv' % (data_dir, data_name)

#data processing
raw_df = pd.read_csv(raw_data_file, index_col = 0)
processed_df = pd.DataFrame(raw_df)

# Save to CSV
processed_df = processed_df + 0 #convert boolean values to numeric
processed_df = processed_df.reset_index(drop = True)
processed_df.to_csv(processed_file, header = True, index = False)
