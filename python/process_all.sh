#!/bin/bash

python process_raw_data.py ./data/192/raw ./data/192/

python process_raw_data.py ./data/192/raw ./data/192/ -t mr

python process_raw_data.py ./data/192/raw ./data/192/ -t ct

python process_raw_data.py ./data/64/raw ./data/64/

python process_raw_data.py ./data/64/raw ./data/64/ -t mr

python process_raw_data.py ./data/64/raw ./data/64/ -t ct
