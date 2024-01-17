#!/bin/bash

python vamp_prior_analysis.py -n 2 -i most_balanced

python vamp_prior_analysis.py -n 5 -i system_0
python vamp_prior_analysis.py -n 5 -i system_1
python vamp_prior_analysis.py -n 5 -i most_balanced
