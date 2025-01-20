# builds index file in the beginning for training and validation data so it doesn't screw with DDP runs

import os
from ..net.utility import ChessDataset

# builds index file, change these to correct location
tr = r'/Volumes/HP X900W/data_tr.csv'
val = r'/Volumes/HP X900W/data_val.csv'

print("Training Data...")
data_tr = ChessDataset(tr)

print("Validation Data...")
data_val = ChessDataset(val)

print("Done! If this was too quick, it probably already detected an index file.")