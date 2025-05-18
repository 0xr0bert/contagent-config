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
relationship_sd = 0.05

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
BELIEFS_NS = UUID('{6da81031-d93b-4269-9046-598924305983}')

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

# belief * belief. No b3 negative
relationships_means = np.array([
    [0, 0, 0.3, -0.3, 0.1, 0.1, -0.1, 0, 0],
    [0, 0, -0.3, 0.3, -0.1, -0.1, 0.1, 0, 0],
    [0.3, -0.3, 0, 0, 0, 0.3, -0.3, 0, 0],
    [-0.3, 0.3, 0, 0, 0, -0.3, 0.3, 0, 0],
    [0.1, 0.1, 0, 0, 0, 0, 0, 0, 0],
    [0.1, -0.1 ,0.3, -0.3, 0, 0, 0, 0, 0],
    [-0.1, 0.1, -0.3, 0.3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

belief_names = np.array([
    "Group 1 Positive",
    "Group 1 Negative",
    "Group 2 Positive",
    "Group 2 Negative",
    "Group 3 Positive",
    "Group 4 Positive",
    "Group 4 Negative",
    "Group 5 Positive",
    "Group 5 Negative"
])

# Agents -----------------------------------------------------------------------
AGENT_NS = UUID('{e46fc665-ab0d-4cbb-adce-6fad4a0e6455}')

def random_activation():
    a, b = (-1 - 0) / 0.1, (1 - 0) / 0.1
    rng = np.random.default_rng()
    if rng.random() <= 0.5:
        return 0.0
    else:
        return truncnorm.rvs(a, b, loc=0, scale=0.1)

# Run data ---------------------------------------------------------------------

def create_run_data(run: int):

    # Beliefs

    belief_n_ns = uuid5(BELIEFS_NS, f"beliefs_{run}")
    beliefs_n_df = pd.DataFrame({"name": belief_names})
    beliefs_n_df["uuid"] = [str(uuid5(belief_n_ns, name)) for name in belief_names]

    perceptions = []
    relationships = []
    for i in range(len(beliefs_n_df)):
        cycle_a = (-1 - perceptions_means_and_sd[i][0]) / perceptions_means_and_sd[i][1]
        cycle_b = (1 - perceptions_means_and_sd[i][0]) / perceptions_means_and_sd[i][1]
        walk_a = (-1 - perceptions_means_and_sd[i][2]) / perceptions_means_and_sd[i][3]
        walk_b = (1 - perceptions_means_and_sd[i][2]) / perceptions_means_and_sd[i][3]
        pt_a = (-1 - perceptions_means_and_sd[i][4]) / perceptions_means_and_sd[i][5]
        pt_b = (1 - perceptions_means_and_sd[i][4]) / perceptions_means_and_sd[i][5]
        drive_a = (-1 - perceptions_means_and_sd[i][6]) / perceptions_means_and_sd[i][7]
        drive_b = (1 - perceptions_means_and_sd[i][6]) / perceptions_means_and_sd[i][7]
        
        perceptions.append({
            cycle: truncnorm.rvs(
                cycle_a, cycle_b, loc=perceptions_means_and_sd[i][0],
                scale=perceptions_means_and_sd[i][1]
            ),
            walk: truncnorm.rvs(
                walk_a, walk_b, loc=perceptions_means_and_sd[i][2],
                scale=perceptions_means_and_sd[i][3]
            ),
            pt: truncnorm.rvs(
                pt_a, pt_b, loc=perceptions_means_and_sd[i][4],
                scale=perceptions_means_and_sd[i][5]
            ),
            drive: truncnorm.rvs(
                drive_a, drive_b, loc=perceptions_means_and_sd[i][6],
                scale=perceptions_means_and_sd[i][7]
            )
        })

        relationships_for_belief = {}

        for j in range(len(beliefs_n_df)):
            a = (0 - relationships_means[i][j]) / relationship_sd
            b = np.inf
            relationships_for_belief[beliefs_n_df.iloc[j]["uuid"]] = \
                truncnorm.rvs(
                    a, b, loc=relationships_means[i][j], scale=relationship_sd
                )

        relationships.append(relationships_for_belief)
    
    beliefs_n_df["perceptions"] = perceptions
    beliefs_n_df["relationships"] = relationships

    beliefs_n_df.to_json(f"{outputdir}/beliefs_{run}.json", orient="records")

    # Agents

    agent_n_ns = uuid5(AGENT_NS, f"agents_{run}")
    agents_n_df = pd.DataFrame({
        "uuid": [str(uuid5(agent_n_ns, str(i))) for i in range(n_agents)]
    })
    agents_n_df["activations"] = [
        [{
            u: random_activation()
            for u in beliefs_n_df["uuid"]
        }] for i in range(n_agents)
    ]

    network = networkx.watts_strogatz_graph(n=n_agents, k=10, p=0.3)
    rng = np.random.default_rng()
    for i in range(n_agents):
        if rng.random() <= 0.8:
            network.add_edge(i, i)
    
    for edge in network.edges:
        network.edges[edge[0], edge[1]]["weight"] = lognorm.rvs(1)

    friends = np.full(n_agents, {})
    for edge in network.edges:
        friends[edge[0]][agents_n_df.iloc[edge[1]]["uuid"]] = network.edges[edge[0], edge[1]]["weight"]
    
    agents_n_df["friends"] = friends

    delta_a, delta_b = (1E-6 - 0.7) / 0.05, np.inf
    agents_n_df["deltas"] = [
        {
            belief_uuid: truncnorm.rvs(
                delta_a, delta_b, loc=0.7, scale=0.05
            ) for belief_uuid in beliefs_n_df["uuid"] 
        } for _ in range(n_agents)
    ]

    prs = []

    for i in range(n_agents):
        prs_for_agent = {}
        for j in range(len(beliefs_n_df)):
            prs_for_belief = {}
            for k in range(len(behaviours_df)):
                mu = perceptions_means_and_sd[j][k*2]
                sigma = perceptions_means_and_sd[j][k*2+1]
                a, b = (-1 - mu) / sigma, (1 - mu) / sigma
                prs_for_belief[behaviours_df.iloc[k]["uuid"]] = truncnorm.rvs(
                    a, b, loc=mu, scale=sigma
                )
            prs_for_agent[beliefs_n_df.iloc[j]["uuid"]] = prs_for_belief
        prs.append(prs_for_agent)
    
    agents_n_df["performance_relationships"] = prs
    agents_n_df["actions"] = [[] for _ in range(n_agents)]

    agents_n_df.to_json(f"{outputdir}/agents_{run}.json.zst", compression="zstd", orient="records")

for i in range(n_runs):
    create_run_data(i)