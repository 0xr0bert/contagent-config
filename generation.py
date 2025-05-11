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
