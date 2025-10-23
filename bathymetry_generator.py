import os
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import functions_Geometry_AHN

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Folder configuration
# ---------------------------------------------------------------
main_folder = r"C:\Users\vinji\Python\AHNProfiel_Viewer\AHNProfiel_Dashboard"

# Input and output folders
input_folder = os.path.join(main_folder, "01_Input_shapefiles")
save_intermediate = os.path.join(main_folder, "02_Intermediate_shapefiles")
save_bathy = os.path.join(main_folder, "03_Bathymetry")
rasterdate_file = os.path.join(main_folder, "data_rws_rasters.txt")

# Create necessary folders
os.makedirs(save_intermediate, exist_ok=True)
os.makedirs(save_bathy, exist_ok=True)

# Subfolder for AHN TIFFs
ahn_folder = os.path.join(save_bathy, "ahn_tiffs")
os.makedirs(ahn_folder, exist_ok=True)

# Skip generating DWP shapefiles if already done
skip_dwp_generation = True

# Skip existing CSVs to save time
skip_existing_csv = True

# ---------------------------------------------------------------
# DWP generation settings
# ---------------------------------------------------------------
x_voorland_p = 200
x_achterland_p = 150
x_voorland_b = 150
x_achterland_b = 100

# ---------------------------------------------------------------
# Generate DWP shapefiles
# ---------------------------------------------------------------
if skip_dwp_generation is False:
    # Primair
    file_in = os.path.join(input_folder, "Hecto_Dijkring_Join.shp")
    file_klijn = os.path.join(input_folder, "WSHDNormtraject_Base.shp")
    file_offlijn = os.path.join(input_folder,
                                "WSHD_prim_kering_2025_5moffbinnen.shp")

    print("🧭 Generating DWP Primair...")
    dwp_primair = functions_Geometry_AHN.generate_dwp(
        file_in, file_klijn, file_offlijn, x_achterland_p, x_voorland_p
    )
    file_prim = os.path.join(save_intermediate,
                             "dwp_2025_prim_kering_hecto.shp")
    dwp_primair.to_file(file_prim)
    print(f"✅ Saved: {file_prim}")

    # Boezem links
    file_in = os.path.join(input_folder, "WSHD_Boezem_Join.shp")
    file_klijn = os.path.join(input_folder, "WSHDNormtraject_Base.shp")
    file_offlijn = os.path.join(input_folder,
                                "WSHD_Boezem_5mLinksOffset.shp")

    print("🧭 Generating DWP Boezem Links...")
    dwp_links = functions_Geometry_AHN.generate_dwp(
        file_in, file_klijn, file_offlijn, x_achterland_b, x_voorland_b
    )
    file_links = os.path.join(save_intermediate,
                              "dwp_2025_boezem_links_hecto.shp")
    dwp_links.to_file(file_links)
    print(f"✅ Saved: {file_links}")

    # Boezem rechts
    file_in = os.path.join(input_folder, "WSHD_Boezem_Join.shp")
    file_klijn = os.path.join(input_folder, "WSHDNormtraject_Base.shp")
    file_offlijn = os.path.join(input_folder,
                                "WSHD_Boezem_5mRechtsOffset.shp")

    print("🧭 Generating DWP Boezem Rechts...")
    dwp_rechts = functions_Geometry_AHN.generate_dwp(
        file_in, file_klijn, file_offlijn, x_achterland_b, x_voorland_b
    )
    file_rechts = os.path.join(save_intermediate,
                               "dwp_2025_boezem_rechts_hecto.shp")
    dwp_rechts.to_file(file_rechts)
    print(f"✅ Saved: {file_rechts}")


# ---------------------------------------------------------------
# Worker function (runs in parallel)
# ---------------------------------------------------------------
def process_feature(
        row,
        rasterdate_file,
        savef,
        shape_group,
        ahn_folder,
        skip_existing
        ):

    try:
        import os
        import functions_Geometry_AHN

        # Subfolder for group-specific TIFFs
        tiff_folder = os.path.join(ahn_folder, shape_group)
        os.makedirs(tiff_folder, exist_ok=True)

        # Expected CSV name
        if shape_group == "boezem_rechts":
            name_csv = f"dp_cpt_{row.unique_}_Rechts.csv"
        elif shape_group == "boezem_links":
            name_csv = f"dp_cpt_{row.unique_}_Links.csv"
        elif shape_group == "primair":
            name_csv = f"dp_cpt_{row.unique_}_Primair.csv"
        else:
            name_csv = f"dp_cpt_{row.unique_}_Unknown.csv"

        csv_path = os.path.join(save_bathy, name_csv)

        # Skip if CSV already exists
        if skip_existing and os.path.exists(csv_path):
            return f"⏩ Skipped existing CSV: {name_csv}"

        # Process this DWP
        df, path = functions_Geometry_AHN.raster_data_to_df(
            dwp=row,
            rasterdate_file=rasterdate_file,
            shape_group=shape_group,
            csv_folder=save_bathy,
            tiff_folder=tiff_folder,
        )

        return f"✅ {shape_group}: {getattr(row, 'unique_', '?')}"

    except Exception as e:
        return f"⚠️ {shape_group}: {getattr(row, 'unique_', '?')} -> {e}"


# ---------------------------------------------------------------
# Run a group in parallel
# ---------------------------------------------------------------
def run_parallel_group(file_dwp, shape_group):
    print(f"\n=== Processing group: {shape_group} ===")
    data = gpd.read_file(file_dwp).set_crs(28992)

    if len(data) == 0:
        print(f"⚠️ No features found in {file_dwp}")
        return

    rows = [row for _, row in data.iterrows()]
    results = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_feature, row, rasterdate_file, save_bathy,
                shape_group, ahn_folder, skip_existing_csv
            )
            for row in rows
        ]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append(f"💥 Crash: {e}")

    print("\n".join(results))
    print(f"✅ Finished {len(results)} profiles for {shape_group}.")


# ---------------------------------------------------------------
# Run for all groups
# ---------------------------------------------------------------
if __name__ == "__main__":
    run_parallel_group(os.path.join(
        save_intermediate, "dwp_2025_prim_kering_hecto.shp"), "primair"
                       )
    run_parallel_group(os.path.join(
        save_intermediate, "dwp_2025_boezem_links_hecto.shp"), "boezem_links"
                       )
    run_parallel_group(os.path.join(
        save_intermediate, "dwp_2025_boezem_rechts_hecto.shp"), "boezem_rechts"
                       )
    print("\n🎯 All DWP sets completed successfully.")
