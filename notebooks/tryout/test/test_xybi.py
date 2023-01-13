# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: csp
#     language: python
#     name: csp
# ---

# %%
import torch
import time
import numpy as np
import pandas as pd
import scvi
import scanpy as sc

import matplotlib.pyplot as plt
from matplotlib import rcParams


import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/cross_species_prediction/')
import constraint_pancreas_example.test.model.test_model as tm
import importlib
importlib.reload(tm)
from constraint_pancreas_example.test.model.test_model_xybi import test_model

# %%
test_model()

# %%
