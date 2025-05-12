#!/usr/bin/env python
import pandas as pd
import numpy as np
import networkx
import os
from scipy.stats import truncnorm, lognorm
from uuid import uuid5, UUID
from datetime import datetime

# Configuration parameters -----------------------------------------------------
currentdate = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
n_runs = 100
n_agents = 1000
alpha = 1.0

# Make dirs
outputdir = f"out-{currentdate}"
os.makedirs(outputdir)


# Behaviours -------------------------------------------------------------------
BEHAVIOURS_NS = UUID('{bfcb5d54-e392-474c-b81b-26d7fb8cd6cf}')
behaviours_df = pd.DataFrame({
    "name": ["Cycle", "Walk", "PT", "Drive"]
})
behaviours_df["uuid"] = [str(uuid5(BEHAVIOURS_NS, name)) for name in behaviours_df["name"]]
behaviours_df.to_json(f"{outputdir}/behaviours.json", orient="records")
cycle = uuid5(BEHAVIOURS_NS, "Cycle")
walk = uuid5(BEHAVIOURS_NS, "Walk")
pt = uuid5(BEHAVIOURS_NS, "PT")
drive = uuid5(BEHAVIOURS_NS, "Drive")

# Beliefs ----------------------------------------------------------------------
# beliefs on down, b1 positive, b1 negative, b2 positive, etc. No b3 negative.
# Across: cycling mean, cycling sd, walk mean, walk sd, pt mean, pt sd, drive mean, drive sd.
perceptions_means_and_sd = np.array([
    [0.53, 0.74*alpha, 0.54, 0.68*alpha, 0.45, 0.87*alpha, 0.27, 0.87*alpha],
    [-0.39, 0.79*alpha, -0.30, 0.73*alpha, 0.08, 0.76*alpha, -0.07, 0.47*alpha],
    [0.54, 0.66*alpha, 0.21, 0.70*alpha, 0.45, 0.76*alpha, 0.63, 0.67*alpha],
    [-0.24, 0.71*alpha, -0.04, 0.60*alpha, -0.05, 0.58*alpha, -0.42, 0.79*alpha],
    [0.29, 0.73*alpha, 0.25, 0.75*alpha, 0.25, 0.50*alpha, 0.29, 0.95*alpha],
    [0.19, 0.86*alpha, 0.38, 0.64*alpha, 0.18, 0.82*alpha, 0.38, 0.79*alpha],
    [0.15, 0.99*alpha, -0.13, 0.64*alpha, 0.17, 0.98*alpha, 0.00, 1.00*alpha],
    [0.40, 0.72*alpha, 0.30, 0.69*alpha, 0.14, 0.88*alpha, 0.43, 0.76*alpha],
    [-0.70, 0.48*alpha, -0.43, 0.53*alpha, -0.40, 0.55*alpha, 0.00, 1.15*alpha]
])