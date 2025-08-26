
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import io

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

############################################
# Universe Expansion (time vs scale factor)
############################################

df_expansion = pd.read_csv(os.path.join(DATA_DIR, 'universe_expansion.csv'))
plt.figure()
df_expansion.plot(kind='line')
plt.xlabel('Age (Billion Years)')
plt.ylabel('Scale Factor')
plt.title('Universe Expansion')
plt.tight_layout()
plt.savefig('universe_expansion.png')
plt.show()

############################################
# CMB Temperature vs Time
############################################

df_cmb = pd.read_csv(os.path.join(DATA_DIR, 'cmb_temperature_data.csv'))
plt.figure(figsize=(10, 6))
plt.plot(df_cmb['Age (Gyr)'], df_cmb['CMB Temperature (K)'], color='blue', linestyle='-')
plt.xlabel('Age of the Universe (Gyr)')
plt.ylabel('CMB Temperature (K)')
plt.title('CMB Temperature vs Age of the Universe')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('cmb_temperature.png')
plt.show()

############################################
# Star Formation Rate vs Time
############################################

obs_data = {
    "Redshift": [0.05, 0.3, 0.5, 0.7, 1.0, 1.1, 1.75, 2.2, 2.3, 3.05,
                 3.8, 4.9, 5.9, 7.0, 7.9, 7.0, 8.0],
    "log_SFRD": [-1.82, -1.50, -1.39, -1.20, -1.25, -1.02, -0.75, -0.87, -0.75, -0.97,
                 -1.29, -1.42, -1.65, -1.79, -2.09, -2.00, -2.21],
}
df_obs = pd.DataFrame(obs_data)
df_obs["SFRD"] = 10**df_obs["log_SFRD"]

z_values = np.linspace(0, 10, 300)
param_00151, param_2_9, param_5_6, param_2_7 = 0.0151, 2.9, 5.6, 2.7
sfr_user = (param_00151 * (1 + ((1 + z_values)/param_2_9)**param_5_6)) / ((1 + z_values)**param_2_7)
a_madau, b_madau, c_madau = 0.01, 2.7, 5.6
sfr_madau = a_madau * (1 + z_values)**b_madau / (1 + ((1 + z_values)/2.9)**c_madau)
sfr_model_A = 0.02 * (1 + z_values)**3 / (1 + ((1 + z_values)/5.0)**4)
sfr_model_B = 0.008 * (1 + z_values)**2.5 * np.exp(-z_values/3.0)

plt.figure(figsize=(12, 7), dpi=200)
plt.scatter(df_obs["Redshift"], df_obs["SFRD"], color="black", marker="o", s=60, label="Obs. Data (Madau+2014)")
plt.plot(z_values, sfr_user, label='User Formula', color='tab:blue', linestyle='-', linewidth=2)
plt.plot(z_values, sfr_madau, label='Madau & Dickinson (2014) Fit', color='tab:orange', linestyle='--', linewidth=2)
plt.plot(z_values, sfr_model_A, label='Fitting Model A', color='tab:green', linestyle='-.', linewidth=2)
plt.plot(z_values, sfr_model_B, label='Fitting Model B', color='tab:red', linestyle=':', linewidth=2)
plt.xlabel('Redshift $z$')
plt.ylabel(r'SFR Density [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]')
plt.title('Cosmic Star Formation History')
plt.yscale('log')
plt.xlim(0, 10)
plt.ylim(1e-3, 1)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('star_formation_rate.png')
plt.show()

############################################
# Element Abundance
############################################

df_elem = pd.read_excel(os.path.join(DATA_DIR, 'Book1.xls'))
df_elem.index = range(1,119)
df_sorted_abundance = df_elem.sort_values(by='Element Abundance', ascending=True).copy()
df_sorted_abundance = df_sorted_abundance[df_sorted_abundance['Element Abundance'] > 0].copy()

plt.figure(figsize=(12, 25))
plt.barh(df_sorted_abundance['Element Symbol'], df_sorted_abundance['Element Abundance'], color='skyblue')
plt.xscale("log")
plt.xlabel("Abundance in Universe (%)")
plt.ylabel("Element Symbol")
plt.title("Elemental Abundance in Universe (Log Scale)")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('element_abundance_bar.png')
plt.show()

top_elements = df_elem[df_elem['Element Abundance'] > 0] \
                .nlargest(10, 'Element Abundance')
plt.figure(figsize=(10, 10))
patches, _ = plt.pie(top_elements['Element Abundance'], startangle=140)
plt.legend(patches, top_elements['Element Symbol'], loc="center left", bbox_to_anchor=(1, 0.5))
plt.title('Top 10 Most Abundant Elements in Universe')
plt.axis('equal')
plt.savefig('top10_element_abundance_pie.png')
plt.show()

############################################
# Star Mass vs Lifespan
############################################

star_names = ["Proxima Centauri", "Sun", "Sirius A", "Betelgeuse", "Rigel"]
star_masses = np.array([0.123, 1, 2.1, 20, 21])
lifespans = 10 * star_masses**-2.5
colors = ["purple", "orange", "blue", "red", "green"]

plt.figure(figsize=(10,6))
for i in range(len(star_names)):
    plt.plot([star_names[i], star_names[i]], [1e-3, lifespans[i]], marker='o', color=colors[i],
             linewidth=3, markersize=8, label=f"{star_names[i]} ({star_masses[i]} M☉)")
plt.yscale("log")
plt.ylabel("Lifespan (Gyr, log scale)")
plt.title("Star Mass vs Lifespan (5 Real Stars)")
plt.legend()
plt.grid(True, ls="--", lw=0.5)
plt.tight_layout()
plt.savefig('star_mass_vs_lifespan.png')
plt.show()

############################################
# Supernova Rate vs Time
############################################

sn_data = """z,Rate_CCSN,Rate_Ia
0.00,0.50e-4,0.10e-4
0.07,1.06e-4,0.20e-4
0.10,1.20e-4,0.25e-4
0.20,1.50e-4,0.35e-4
0.30,2.00e-4,0.50e-4
0.39,3.29e-4,0.65e-4
0.45,3.80e-4,0.75e-4
0.50,4.10e-4,0.85e-4
0.60,5.00e-4,1.29e-4
0.62,5.20e-4,1.29e-4
0.70,5.80e-4,1.40e-4
0.73,6.40e-4,1.50e-4
0.80,6.20e-4,1.60e-4
0.90,6.00e-4,1.70e-4
1.00,5.80e-4,1.80e-4
1.10,5.50e-4,1.90e-4
1.20,5.00e-4,2.00e-4
1.30,4.50e-4,2.10e-4
1.40,4.00e-4,2.20e-4
1.50,3.70e-4,2.30e-4
1.60,3.50e-4,2.40e-4
1.70,3.35e-4,2.50e-4
1.80,3.20e-4,2.60e-4
1.90,3.10e-4,2.70e-4
2.00,3.70e-4,2.80e-4
2.10,3.65e-4,2.90e-4
2.20,3.60e-4,3.00e-4
2.30,3.55e-4,3.10e-4
2.40,3.50e-4,3.20e-4
2.50,3.50e-4,3.30e-4
"""
df_sn = pd.read_csv(io.StringIO(sn_data))

plt.figure(figsize=(10, 6))
plt.plot(df_sn['z'], df_sn['Rate_CCSN'], marker='o', linestyle='-', label='Core-Collapse Supernova Rate')
plt.plot(df_sn['z'], df_sn['Rate_Ia'], marker='x', linestyle='--', label='Type Ia Supernova Rate')
plt.xlabel('Redshift (z)')
plt.ylabel('Supernova Rate (Mpc$^{-3}$ yr$^{-1}$)')
plt.title('Supernova Formation Rate vs. Redshift')
plt.yscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('supernova_rates.png')
plt.show()

plt.figure(figsize=(8,6))
plt.bar(df_sn["z"]-0.01, df_sn["Rate_CCSN"]*1e4, width=0.02, label="Core-Collapse SNe", alpha=0.7)
plt.bar(df_sn["z"]+0.01, df_sn["Rate_Ia"]*1e4, width=0.02, label="Type Ia SNe", alpha=0.7)
plt.xlabel("Redshift (z)", fontsize=12)
plt.ylabel("SN Rate (10⁻⁴ yr⁻¹ Mpc⁻³)", fontsize=12)
plt.title("Supernova Rate vs Cosmic Time (Histogram)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig('supernova_rates_histogram.png')
plt.show()

############################################
# Solar System Formation Timeline
############################################

events = [
    'Formation of the Sun',
    'First solids (CAIs)',
    'Planetesimals',
    'Terrestrial planets (Earth)',
    'Gas giants (Jupiter, Saturn)',
    'Late Heavy Bombardment',
    'Formation of the Moon',
    'Ice giants (Uranus, Neptune)',
    'Migration of giant planets',
    'Kuiper Belt formation'
]
ages = [4.57e9, 4.567e9, 4.56e9, 4.54e9, 4.5e9, 4.0e9, 4.5e9, 4.4e9, 4.0e9, 4.5e9]
ages, events = zip(*sorted(zip(ages, events), reverse=True))
plt.figure(figsize=(12,6))
plt.plot(ages, range(len(ages)), marker='o', color='darkorange')
for i in range(len(ages)):
    plt.text(ages[i], i, "  " + events[i], va='center', fontsize=9)
plt.yticks([])
plt.xlabel("Age (Billion Years Ago)")
plt.title("Timeline of Solar System Formation")
plt.gca().invert_xaxis()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('solar_system_formation_timeline.png')
plt.show()

############################################
# Sun Luminosity vs Time
############################################

t_main_seq = np.linspace(0, 10, 200)
t_red_giant = np.linspace(10, 11, 50)
L_main_seq = 0.7 + 0.8 * (t_main_seq / 10)**1.5
L_red_giant = np.linspace(L_main_seq[-1], 100, len(t_red_giant))
time = np.concatenate((t_main_seq, t_red_giant))
luminosity = np.concatenate((L_main_seq, L_red_giant))
df_star = pd.DataFrame({'Age_from_formation_Gyr': time, 'Luminosity_Lsun': luminosity})

plt.figure(figsize=(12, 6))
plt.plot(df_star['Age_from_formation_Gyr'], df_star['Luminosity_Lsun'], color='orange', linewidth=2, label='Sun-like Star')
present_age = 4.57
plt.axvline(present_age, color='gray', linestyle='--', label='Present Day (~4.57 Gyr)')
plt.scatter(present_age, 1.0, color='red', zorder=5)
plt.text(present_age+0.1, 1.1, 'Present Sun', fontsize=10, color='red')
plt.xlabel('Age from Formation (Gyr)')
plt.ylabel('Luminosity (L / L☉)')
plt.title('Approximate Evolution of a Sun-like Star')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('sun_luminosity_evolution.png')
plt.show()

############################################
# Earth's Major Events Timeline
############################################

df_events = pd.DataFrame({
    'Event': [
        'Formation of Earth', 'Formation of the Moon', 'First Oceans',
        'Oldest Minerals', 'First Life', 'Great Oxygenation', 'First Eukaryotes',
        'Multicellular Life', 'Cambrian Explosion', 'First Land Plants',
        'First Vertebrates', 'First Insects', 'First Amphibians', 'First Reptiles',
        'Permian-Triassic Extinction', 'First Dinosaurs', 'Breakup of Pangea',
        'First Mammals', 'First Birds', 'Cretaceous-Paleogene Extinction',
        'First Primates', 'Homo Sapiens', 'Last Ice Age Ends', 'Agriculture',
        'Industrial Revolution', 'Present Day'
    ],
    'Age (Ma)': [
        4540, 4500, 4400, 4400, 3500, 2400, 1800, 600, 541, 470, 480, 400, 370, 310,
        252, 230, 200, 200, 150, 66, 55, 0.3, 0.0117, 0.01, 0.0002, 0
    ]
})
df_events['Age (Ma)'] = df_events['Age (Ma)'].replace(0, 0.00001)
df_events = df_events.sort_values('Age (Ma)', ascending=False).reset_index(drop=True)

def color_type(event):
    bio = ['Life','Eukaryotes','Multicellular','Insects','Amphibians','Repptiles','Dinosaurs','Mammals','Birds','Primates','Homo']
    geo = ['Earth','Moon','Oceans','Minerals','Oxygenation','Extinction','Breakup', 'Ice Age', 'Agriculture', 'Industrial Revolution', 'Present Day']
    if any(keyword in event for keyword in bio): return 'green'
    elif any(keyword in event for keyword in geo): return 'blue'
    else: return 'red'
df_events['color'] = df_events['Event'].apply(color_type)

plt.figure(figsize=(12, 8))
plt.barh(df_events['Event'], df_events['Age (Ma)'], color=df_events['color'])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel("Age (Million Years Ago, Ma) [log scale]")
plt.ylabel("Event")
plt.title("Timeline of Earth's History & Life Evolution")
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('earth_history_timeline.png')
plt.show()

############################################
# Atmospheric Oxygen % vs Time
############################################

df_o2 = pd.read_excel(os.path.join(DATA_DIR, 'GEOCARB_input_arrays_renamed.xlsx'))
age = df_o2['Age (Ma)']
O2 = df_o2['Atmospheric Oxygen Level (%)']
O2_err = 0.05 * O2

periods = [
    ('Cambrian', 541, 485, '#fde0dd'),
    ('Ordovician', 485, 444, '#fa9fb5'),
    ('Silurian', 444, 419, '#c51b8a'),
    ('Devonian', 419, 359, '#7a0177'),
    ('Carboniferous', 359, 299, '#edf8fb'),
    ('Permian', 299, 252, '#b3cde3'),
    ('Triassic', 252, 201, '#6497b1'),
    ('Jurassic', 201, 145, '#005b96'),
    ('Cretaceous', 145, 66, '#03396c'),
    ('Paleogene', 66, 23, '#ffddc1'),
    ('Neogene', 23, 2.6, '#fbb4ae'),
    ('Quaternary', 2.6, 0, '#b3cde3')
]
plt.figure(figsize=(15,7))
for name, start, end, color in periods:
    plt.axvspan(end, start, color=color, alpha=0.3)
    plt.text((start+end)/2, max(O2)+2, name, ha='center', va='bottom', fontsize=9, rotation=90)
plt.plot(age, O2, color='green', linewidth=2, label='Atmospheric O₂')
plt.fill_between(age, O2-O2_err, O2+O2_err, color='green', alpha=0.2, label='±5% error')
plt.xlabel('Age (Million Years Ago)')
plt.ylabel('Atmospheric Oxygen Level (%)')
plt.title('Atmospheric Oxygen Level Over Phanerozoic Eon')
plt.gca().invert_xaxis()
plt.ylim(0, max(O2)+10)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('atmospheric_oxygen_over_time.png')
plt.show()


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
