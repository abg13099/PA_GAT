#!/bin/bash

# --- Run experiments on chosen datasets --- 
# Enyzmes Proteins REDDIT-BINARY Collab IMDB-B NCI1

python3 main.py Cora 0.820 0.219 0.008 
python3 main.py Citeseer 0.924 5.532 0.854 
python3 main.py Pubmed 0.281 7.969 0.540
python3 main.py Texas 0.259 0.365 0.161
python3 main.py Wisconsin 0.158 0.662 0.628
python3 main.py Cornell 0.203 0.157 0.398 
python3 main.py Physics 0.788 0.205 0.136
python3 main.py CS 1.304 0.101 2.475
