#%%
# Setup
import os
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import geopandas as gpd
import warnings 
from concurrent.futures import ThreadPoolExecutor

import functions_general
import functions_Geometry_WSHD

# Suppress warnings, divide by zero warnings
warnings.filterwarnings("ignore") 
main_folder = r"C:\Users\vinji\Python\AHNProfiel_Viewer\AHNProfiel_Dashboard"



#%% 
# Dwarsprofielen ter plaatse genereren
# volgende functie is maar eenmalig wanneer nieuwe dwarsprofielen worden gegenereerd
x_voorland_p = 200
x_achterland_p = 150
x_voorland_b = 150
x_achterland_b = 100
input_folder = os.path.join(main_folder, "01_Input_shapefiles")
savef = os.path.join(main_folder, "02_Intermediate_shapefiles")

# input datafiles primaire kering
file_in = os.path.join(input_folder,"Hecto_Dijkring_Join.shp")
file_klijn = os.path.join(input_folder,"WSHDNormtraject_Base.shp")
file_offlijn = os.path.join(input_folder,"WSHD_prim_kering_2025_5moffbinnen.shp")

dwp_new = functions_Geometry_WSHD.generate_dwp(file_in, file_klijn, file_offlijn, x_achterland_p, x_voorland_p)
dwp_new.to_file(os.path.join(savef,"dwp_2025_prim_kering_hecto.shp"))


# input datafiles boezem links
file_in = os.path.join(input_folder,"WSHD_Boezem_Join.shp")
file_klijn = os.path.join(input_folder,"WSHDNormtraject_Base.shp")
file_offlijn = os.path.join(input_folder,"WSHD_Boezem_5mLinksOffset.shp")

dwp_new = functions_Geometry_WSHD.generate_dwp(file_in, file_klijn, file_offlijn, x_achterland_b, x_voorland_b)
dwp_new.to_file(os.path.join(savef,"dwp_2025_boezem_links_hecto.shp"))


# input datafiles boezem rechts
file_in = os.path.join(input_folder,"WSHD_Boezem_Join.shp")
file_klijn = os.path.join(input_folder,"WSHDNormtraject_Base.shp")
file_offlijn = os.path.join(input_folder,"WSHD_Boezem_5mRechtsOffset.shp")

dwp_new=functions_Geometry_WSHD.generate_dwp(file_in, file_klijn, file_offlijn, x_achterland_b, x_voorland_b)
dwp_new.to_file(os.path.join(savef,"dwp_2025_boezem_rechts_hecto.shp"))


#%% 
# Generate raster to df (and csv) for each dwp

input_folder = os.path.join(main_folder, "02_Intermediate_shapefiles")
savef = os.path.join(main_folder, "03_Bathymetry")
rasterdate_file = os.path.join(main_folder,'data_rws_rasters.txt')

#%%
# input datafiles primaire kering
file_dwp = os.path.join(input_folder,'dwp_2025_prim_kering_hecto.shp')
data: gpd.GeoDataFrame = gpd.read_file(file_dwp).set_crs(28992)

def run_step(i):
    # Eén element verwerken
    functions_Geometry_WSHD.step1_raster_data_to_df(
        data.iloc[i], rasterdate_file, "primair", savef
    )

# ThreadPool gebruiken
with ThreadPoolExecutor(max_workers=8) as executor:  # pas workers aan naar je CPU/I/O
    futures = [executor.submit(run_step, i) for i in range(len(data))]

    # Optioneel: wachten tot alles klaar is
    for f in futures:
        f.result()


#%%
# input datafiles boezem links
file_dwp = os.path.join(input_folder,'dwp_2025_boezem_links_hecto.shp')
data: gpd.GeoDataFrame = gpd.read_file(file_dwp).set_crs(28992)

for i in range(len(data)):
    dfs = functions_Geometry_WSHD.step1_raster_data_to_df(data.iloc[i], rasterdate_file, "boezem_links", savef)

# input datafiles boezem rechts
file_dwp = os.path.join(input_folder,'dwp_2025_boezem_rechts_hecto.shp')
data: gpd.GeoDataFrame = gpd.read_file(file_dwp).set_crs(28992)

for i in range(len(data)):
    dfs = functions_Geometry_WSHD.step1_raster_data_to_df(data.iloc[i], rasterdate_file, "boezem_links", savef)
# %%
