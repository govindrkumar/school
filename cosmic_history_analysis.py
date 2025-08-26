# -*- coding: utf-8 -*-
"""
Cosmic History Full Analysis Script
All file paths are relative (e.g., 'data/universe_expansion.csv')
Each graph is saved using plt.savefig() after plotting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import io

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ... [previous sections: Universe Expansion, CMB, SFR, Abundance, etc.] ...

#######################################################
# Fossil Diversity Over Geological Time
#######################################################

# Downloaded fossil diversity dataset should be placed in data/pbdb_occurrences.csv
df_fossil = pd.read_csv(os.path.join(DATA_DIR, 'pbdb_occurrences.csv'), low_memory=False)

# Major period mapping (define as in your notebook)
stage_to_period = {
    'Cryogenian':'Precambrian', 'Tonian':'Precambrian', 'Ediacaran':'Precambrian',
    'Furongian':'Cambrian', 'Paibian':'Cambrian', 'Delamaran':'Cambrian', 'Stage 3':'Cambrian',
    'Middle Cambrian':'Cambrian', 'Stage 2':'Cambrian',
    'Tremadoc':'Ordovician', 'Tremadocian':'Ordovician', 'Arenig':'Ordovician', 'Darriwilian':'Ordovician',
    'Middle Ordovician':'Ordovician', 'Late Ordovician':'Ordovician',
    'Llandovery':'Silurian', 'Wenlock':'Silurian', 'Ludlow':'Silurian', 'Pridoli':'Silurian',
    'Lochkovian':'Devonian', 'Pragian':'Devonian', 'Emsian':'Devonian', 'Eifelian':'Devonian',
    'Givetian':'Devonian', 'Frasnian':'Devonian', 'Famennian':'Devonian',
    'Kinderhookian':'Carboniferous', 'Tournaisian':'Carboniferous',
    'Bashkirian':'Carboniferous', 'Moscovian':'Carboniferous', 'Kasimovian':'Carboniferous', 'Gzhelian':'Carboniferous',
    'Asselian':'Permian', 'Sakmarian':'Permian', 'Artinskian':'Permian', 'Wordian':'Permian',
    'Wuchiapingian':'Permian', 'Changhsingian':'Permian',
    'Induan':'Triassic', 'Olenekian':'Triassic', 'Anisian':'Triassic', 'Ladinian':'Triassic',
    'Carnian':'Triassic', 'Norian':'Triassic', 'Rhaetian':'Triassic',
    'Hettangian':'Jurassic', 'Sinemurian':'Jurassic', 'Pliensbachian':'Jurassic', 'Toarcian':'Jurassic',
    'Aalenian':'Jurassic', 'Bajocian':'Jurassic', 'Bathonian':'Jurassic', 'Callovian':'Jurassic',
    'Oxfordian':'Jurassic', 'Kimmeridgian':'Jurassic', 'Tithonian':'Jurassic',
    'Berriasian':'Cretaceous','Valanginian':'Cretaceous','Hauterivian':'Cretaceous','Barremian':'Cretaceous',
    'Aptian':'Cretaceous','Albian':'Cretaceous','Cenomanian':'Cretaceous','Turonian':'Cretaceous',
    'Coniacian':'Cretaceous','Santonian':'Cretaceous','Campanian':'Cretaceous','Maastrichtian':'Cretaceous',
    'Paleocene':'Paleogene','Eocene':'Paleogene','Oligocene':'Paleogene',
    'Miocene':'Neogene','Pliocene':'Neogene',
    'Pleistocene':'Quaternary','Holocene':'Quaternary'
}
df_fossil['major_period'] = df_fossil['early_interval'].map(stage_to_period)

major_periods = df_fossil['major_period'].dropna().unique()
for period in major_periods:
    period_df = df_fossil[df_fossil['major_period'] == period].copy()
    species_count = period_df['early_interval'].value_counts().sort_index()
    plt.figure(figsize=(10,5))
    species_count.plot(kind='bar', color='teal')
    plt.title(f"Species Distribution in {period}")
    plt.xlabel("Stage / Early Interval")
    plt.ylabel("Number of Occurrences")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{period}_species_distribution.png")
    plt.close()

#######################################################
# Mass Extinction Events
#######################################################

mass_extinctions = {
    "End-Ordovician": (443.4, 29144),
    "Late Devonian": (372.2, 23814),
    "End-Permian": (251.9, 10648),
    "End-Triassic": (201.3, 13497),
    "End-Cretaceous": (66.0, 102199)
}
times = [v[0] for v in mass_extinctions.values()]
counts = [v[1] for v in mass_extinctions.values()]
labels = list(mass_extinctions.keys())
colors = ['red', 'blue', 'green', 'purple', 'orange']
markers = ['o', 's', '^', 'D', 'P']

plt.figure(figsize=(10,6))
plt.plot(times, counts, linestyle='-', color='gray', alpha=0.5)
for t, c, l, col, mark in zip(times, counts, labels, colors, markers):
    plt.plot(t, c, marker=mark, color=col, markersize=10, linestyle='None', label=l)
plt.gca().invert_xaxis()
plt.title("Mass Extinction Events Through Time")
plt.xlabel("Time (Million Years Ago)")
plt.ylabel("Number of Species Lost")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig('mass_extinctions.png')
plt.show()

#######################################################
# Human Brain Evolution
#######################################################

data_brain = {
    "Species": [
        "Australopithecus anamensis", "Australopithecus afarensis", "Australopithecus africanus",
        "Australopithecus garhi", "Australopithecus sediba", "Paranthropus aethiopicus",
        "Paranthropus boisei", "Paranthropus robustus", "Homo habilis", "Homo rudolfensis",
        "Homo erectus", "Homo ergaster", "Homo antecessor", "Homo heidelbergensis",
        "Homo neanderthalensis", "Denisovans", "Homo floresiensis", "Homo naledi", "Homo sapiens"
    ],
    "Time_range_MYA": [
        "4.2–3.9", "3.9–2.9", "3.0–2.1", "2.5–2.4", "1.98–1.78",
        "2.7–2.3", "2.3–1.2", "2.0–1.2", "2.4–1.4", "2.4–1.8",
        "1.9–0.1", "1.9–1.4", "1.2–0.8", "0.7–0.2", "0.4–0.04",
        "0.3–0.05", "0.1–0.05", "0.335–0.236", "0.3–0"
    ],
    "Cranial_capacity_cc": [
        "365–370", "375–550", "420–500", "450", "420–450",
        "410", "500–550", "500–550", "510–600", "700–800",
        "600–1100", "600–910", "1000–1150", "1100–1300",
        "1200–1750", "1200–1600", "380–420", "465–610", "1200–1600"
    ]
}
df_brain = pd.DataFrame(data_brain)
time_mid = []
for val in df_brain["Time_range_MYA"]:
    parts = val.replace("present", "0").split("–")
    nums = [float(p) for p in parts]
    time_mid.append(np.mean(nums))
capacity_mid = []
for val in df_brain["Cranial_capacity_cc"]:
    parts = val.split("–")
    if len(parts) == 1:
        capacity_mid.append(float(parts[0]))
    else:
        capacity_mid.append(np.mean([float(p) for p in parts]))
df_brain["Time_mid_MYA"] = time_mid
df_brain["Cranial_capacity_mid"] = capacity_mid

plt.figure(figsize=(10,6))
colors = plt.cm.tab20(np.linspace(0,1,len(df_brain)))
for i in range(len(df_brain)):
    plt.plot(df_brain["Time_mid_MYA"][i], df_brain["Cranial_capacity_mid"][i], marker="o", markersize=8,
             color=colors[i], label=df_brain["Species"][i])
plt.plot(df_brain["Time_mid_MYA"], df_brain["Cranial_capacity_mid"], linestyle="--", color="gray", alpha=0.5)
plt.gca().invert_xaxis()
plt.xlabel("Time (Million Years Ago)")
plt.ylabel("Cranial Capacity (cc)")
plt.title("Brain Size vs Time (Hominids)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig('hominid_brain_size_vs_time.png')
plt.show()

plt.figure(figsize=(12,6))
plt.bar(df_brain["Species"], df_brain["Cranial_capacity_mid"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Cranial Capacity (cc)")
plt.title("Cranial Capacity of Different Hominid Species")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig('hominid_brain_size_bar.png')
plt.show()

#######################################################
# Human Population Growth
#######################################################

data_pop = {
    "Year_BP": [
        -1_000_000, -800_000, -500_000, -200_000, -100_000, -50_000, -20_000, -10_000,
        -5_000, -2_000, -1_000, -500, -200, -100, -50, -20, -10, -5, -2, -1, 0, 50, 100
    ],
    "Population_millions": [
        0.01, 0.02, 0.05, 0.1, 0.2, 1.0, 2.0, 4.0, 20.0, 60.0, 200.0, 400.0, 600.0, 1000.0,
        2000.0, 3000.0, 4500.0, 6000.0, 7800.0, 8000.0, 9000.0, 10400.0, 11000.0
    ],
    "Event": [
        "Early humans appear", "Homo erectus emerges", "Early settlements", "Homo sapiens emerges",
        "Early societies", "Ice Age small groups", "Post-Ice Age", "Agriculture Revolution",
        "Bronze Age", "Early civilizations", "1000 AD population", "1500 AD population", "1800 AD",
        "1900 AD", "1950 AD", "1980 AD", "2000 AD", "2015 AD", "2023 AD", "2025 AD",
        "2050 Projection", "2100 Projection", "2150 Projection"
    ]
}
df_pop = pd.DataFrame(data_pop)
colors_pop = plt.cm.tab20.colors
markers_pop = ['o','s','^','D','P','X','*','h','H','+','x','1','2','3','4','8','p','v','<','>']

def plot_segment(df_segment, title, xlim, key_xticks, fname):
    plt.figure(figsize=(18,5))
    plt.plot(df_segment["Year_BP"], df_segment["Population_millions"], linestyle='-', color='gray', alpha=0.5)
    for year, pop, event, col, mark in zip(df_segment["Year_BP"], df_segment["Population_millions"],
                                           df_segment["Event"], colors_pop, markers_pop):
        plt.plot(year, pop, marker=mark, color=col, markersize=10, linestyle='None', label=event)
    plt.gca().invert_xaxis()
    plt.yscale('log')
    plt.xlabel("Years Before Present")
    plt.ylabel("Population (millions, log scale)")
    plt.title(title)
    plt.xlim(xlim)
    plt.xticks(key_xticks, [f"{abs(int(y))}" for y in key_xticks], rotation=45)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

prehistory = df_pop[df_pop["Year_BP"] <= -10_000]
plot_segment(prehistory, "Human Population: Prehistory", (-1_000_000, -10_000),
             [-1_000_000, -800_000, -500_000, -200_000, -100_000, -50_000, -20_000, -10_000],
             "human_population_prehistory.png")

historical = df_pop[(df_pop["Year_BP"] > -10_000) & (df_pop["Year_BP"] <= 0)]
plot_segment(historical, "Human Population: Historical Period", (-10_000, 0),
             [-10_000, -5_000, -2_000, -1_000, -500, -200, -100, -50, -20, -10, -5, -2, -1, 0],
             "human_population_historical.png")

modern = df_pop[df_pop["Year_BP"] > 0]
plot_segment(modern, "Human Population: Modern & Future Projections", (0, 2150),
             [0, 50, 100], "human_population_modern.png")

#######################################################
# Technological Growth Curve
#######################################################

# Prehistoric tech timeline
df_pre = pd.DataFrame({
    'Year': [-2_500_000, -300_000, -40_000, -10_000, -5_000, -3_500, -1_200],
    'Event': [
        'Stone Tools', 'Control of Fire', 'Early Art & Culture', 'Agriculture', 'Bronze Age',
        'Writing', 'Iron Age'
    ]
})
df_pre['Tech_Level'] = range(1, len(df_pre)+1)
markers_tech = ['o', 's', '^', 'D', 'v', 'p', '*']
colors_tech = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']

plt.figure(figsize=(20,5))
for i in range(len(df_pre)):
    plt.plot(df_pre['Year'][i], df_pre['Tech_Level'][i], marker=markers_tech[i], color=colors_tech[i], markersize=10,
             linestyle='None', label=df_pre['Event'][i])
plt.plot(df_pre['Year'], df_pre['Tech_Level'], linestyle='-', color='darkgreen', alpha=0.5)
plt.xlabel("Year (BC = negative)")
plt.ylabel("Cumulative Tech Level")
plt.title("Prehistoric Technological Growth")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Events")
plt.tight_layout()
plt.savefig('tech_growth_prehistoric.png')
plt.show()

# Historic tech timeline
df_hist = pd.DataFrame({
    'Year': [0, 500, 800, 105, 1200, 1400, 1600, 1760, 1850, 1900],
    'Event': [
        'Roman Empire', 'Early Medieval Tools', 'Early Medieval Tools',
        'Invention of Paper', 'High Medieval Tech', 'Printing Press / Renaissance',
        'Scientific Revolution', 'Industrial Revolution', 'Electricity & Telegraph', 'Pre-Modern Tech'
    ]
})
df_hist = df_hist.sort_values('Year').reset_index(drop=True)
df_hist['Tech_Level'] = range(1, len(df_hist)+1)
markers_hist = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
colors_hist = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'olive', 'grey']

plt.figure(figsize=(22,5))
for i in range(len(df_hist)):
    plt.plot(df_hist['Year'][i], df_hist['Tech_Level'][i], marker=markers_hist[i], color=colors_hist[i], markersize=10,
             linestyle='None', label=df_hist['Event'][i])
plt.plot(df_hist['Year'], df_hist['Tech_Level'], linestyle='-', color='darkblue', alpha=0.5)
plt.xlabel("Year (AD)")
plt.ylabel("Cumulative Tech Level")
plt.title("Historical Technological Growth (0 → 1900 AD)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Events")
plt.tight_layout()
plt.savefig('tech_growth_historical.png')
plt.show()

# Modern tech timeline
df_mod = pd.DataFrame({
    'Year': [1940, 1970, 2000, 2020],
    'Event': [
        'Atomic Age', 'Information Age / Computers', 'Internet', 'AI Acceleration'
    ]
})
df_mod['Tech_Level'] = range(1, len(df_mod)+1)
markers_mod = ['o', 's', '^', 'D']
colors_mod = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(22,5))
for i in range(len(df_mod)):
    plt.plot(df_mod['Year'][i], df_mod['Tech_Level'][i], marker=markers_mod[i], color=colors_mod[i], markersize=10,
             linestyle='None', label=df_mod['Event'][i])
plt.plot(df_mod['Year'], df_mod['Tech_Level'], linestyle='-', color='darkred', alpha=0.5)
plt.xlabel("Year (AD)")
plt.ylabel("Cumulative Tech Level")
plt.title("Modern Technological Growth (1900 → Present/Future)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Events")
plt.tight_layout()
plt.savefig('tech_growth_modern.png')
plt.show()

# Export combined tech growth CSV (optional)
df_tech_growth = pd.concat([df_pre, df_hist, df_mod], ignore_index=True)
df_tech_growth.to_csv(os.path.join(DATA_DIR, 'df_tech_growth.csv'), index=False)