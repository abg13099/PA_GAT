#!/bin/bash

# --- Run experiments on chosen datasets --- 
# Enyzmes, Proteins, REDDIT-BINARY, Collab, IMDB-B, NCI1

python3 main.py ENZYMES
python3 main.py PROTEINS 4.662860280026271 4.762074362660967 0.011099458404294896 
python3 main.py REDDIT-BINARY 
python3 main.py COLLAB
python3 main.py IMDB-B 
python3 main.py NCI1 0.9420738460772539 5.948746813219773 0.1025308337933703 
python3 main.py obgb-ppa
python3 main.py obgb-mohliv
