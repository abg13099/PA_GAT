#!/bin/bash
#
# Runs variance study on the target datasets
#

python3 variance_study.py ENZYMES 25
python3 variance_study.py PROTEINS 25
python3 variance_study.py COLLAB 25
python3 variance_study.py REDDIT-BINARY 25
python3 variance_study.py DD 25
python3 variance_study.py NCI1 25
