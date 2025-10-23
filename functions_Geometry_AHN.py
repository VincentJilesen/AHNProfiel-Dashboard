import pandas as pd
import numpy as np
import math
import os
import re
import datetime as dt
import geopandas as gpd
import rasterio

from owslib.wcs import WebCoverageService
from shapely.ops import nearest_points

from geopandas import GeoDataFrame
from shapely import LineString, Point
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

"""
Geometric functions:
-   generate line crossing kering, should be based on created
    near points with the keringslijn and an offset buitenwaards (ex. 5m).
    This gives the correct direction of the line. Also as an input
    should be given points from where the dwp's are drawn.
"""


def _angle_between_points(point1, point2):
    """
    determine angle between to geometry points
    (two input point geometrys as input)
    should be in the same coordinate convention.
    """
    # Calculate the differences in coordinates
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    # Calculate the angle in radians
    angle = 90-math.degrees(math.atan2(dy, dx))
    # !   the 90 degrees is difference planar and geodesic convention of angle

    return angle


def _geo_to_pixel(x_geo, y_geo, transform):
    """
    function which translates a array coordinate to geo coordinate
    """
    (x_pixel, y_pixel) = ~transform * (x_geo, y_geo)
    return x_pixel, y_pixel


def _determine_nearest(data_p, data_n):
    """
    determines the nearest geo "line" to geo "point" and gives back
    the coordinates of this point (also which line id) and angle from
    the original point.
    """
    # bepaald dichtsbijzijnde punt op lijn
    nearest = gpd.sjoin_nearest(data_p, data_n, distance_col='dist_')
    # Find the indices of the smallest 'dist_' for each 'unique_'
    nearest = nearest.sort_values(by=['unique_', 'dist_'], ascending=True)
    idx = nearest.groupby('unique_')['dist_'].idxmin()
    filtered_nearest = nearest.loc[idx]

    near = []

    for i in filtered_nearest.iterrows():
        try:  # to avoid errors in the database of the keringlijn...
            temp = []
            temp.append(i[1].unique_)
            temp.append(i[1].CODE)
            # haal lijn op die het dichtstbij ligt
            line = data_n[data_n.CODE == i[1].CODE].geometry.iloc[0]
            # haal punt op die het dichtstbij ligt
            point = data_p[data_p.unique_ == i[1].unique_].geometry.iloc[0]
            # bepaal punt op lijnmo
            # afstand tot lijn
            temp.append(i[1].dist_)
            # punt op lijn (haal x, y op)
            n_point = nearest_points(point, line)[1]
            temp.append(n_point.x)
            temp.append(n_point.y)
            # bepaal hoek tussen punt en near punt
            temp.append(_angle_between_points(point, n_point))
            near.append(temp)
        except Exception:
            print(data_n[data_n.CODE == i[1].CODE])
            continue

    # Remove duplicates based on the first element (unique_) if
    # 'Dijkring_17_0p0' is not unique
    df_near = pd.DataFrame(near, columns=['unique_', 'CODE', 'dist_',
                                          'x', 'y', 'angle'])
    if df_near['unique_'].duplicated().any():
        df_near = df_near.drop_duplicates(subset=['unique_'])
    near = df_near.values.tolist()

    return near


def ahn_per_dwp(dwp, outputfolder=os.getcwd(), filename=None):
    """
    Download AHN WCS raster for a DWP geometry and save as deterministic TIFF.
    Reuses existing file if already downloaded.
    """

    os.makedirs(outputfolder, exist_ok=True)

    # Create a safe deterministic filename
    safe_id = re.sub(r'[^A-Za-z0-9_-]+', '_', str(getattr(dwp, 'unique_', 'dwp')))
    filename = filename or f"ahn_{safe_id}.tiff"
    filepath = os.path.join(outputfolder, filename)

    # ✅ Skip WCS download if TIFF already exists
    if os.path.exists(filepath):
        print(f"♻️  Using existing {filename}")
        return filepath

    print(f"🌐 Downloading new raster: {filename}")

    # WCS source
    url_wcs = ("https://service.pdok.nl/rws/ahn/wcs/v1_0"
               "?request=GetCapabilities&service=WCS")
    wcs = WebCoverageService(url_wcs)

    # Bounding box
    bounds = dwp.geometry.bounds
    bounds_rounded = (
        math.floor(bounds[0]),
        math.floor(bounds[1]),
        math.ceil(bounds[2]),
        math.ceil(bounds[3])
    )

    # Request WCS coverage
    output = wcs.getCoverage(
        identifier=["dtm_05m"],
        format="geotiff",
        crs="EPSG:28992",
        subsets=[
            ("X", bounds_rounded[0], bounds_rounded[2]),
            ("Y", bounds_rounded[1], bounds_rounded[3]),
        ],
        width=(math.ceil(bounds[2]) - math.floor(bounds[0])) / 2,
        height=(math.ceil(bounds[3]) - math.floor(bounds[1])) / 2,
    )

    with open(filepath, "wb") as f:
        f.write(output.read())

    return filepath


def _read_rasterfile(rasterdate_file):
    """
    Read the file which has defined the rasters and date of execution
    input:
    - rasterdate_file : path + filename (.txt file)
    output:
    - list_raster (dict): key = date of raster : value = path+file
    """
    # date_file="dates_rasters_17-2_re_test.txt"
    # currentd_data=r'C:\Users\pauno\Desktop\lokale_map_paul\Eenvoudige_toets2023\data'

    # mappen doorlopen aangegeven in 'dates_rasters.txt', voor zoeken straks
    list_raster = {}
    # mappen doorlopen aangegeven in 'dates_rasters.txt', voor zoeken straks
    # ## -> verwijzen naar juist inputfile? Dit per traject? zit dat
    # in de dwarssdoorsnede verwerkt?
    with open(os.path.join(rasterdate_file)) as peilingen:
        lines = peilingen.readlines()
        for d in range(len(lines)):
            a = []
            a = lines[d].split(';')
            paths = a[3].replace('\n', '')

            date1 = dt.datetime(int(a[0]), int(a[1]),
                                int(a[2].replace('\n', '')))
            date1 = int(date1.timestamp())
            list_raster.update({date1: paths})

    return list_raster


def generate_dwp(cpt, k_lijn, off_lijn, m_buitenw, m_binnenw):
    """
    Function to generate dwp from cpt (points) perpendicular on
    keringlijn (line). Using an offset line with the right orientation
    (toward the rivers in this case, buitenwaards). The offset and lines
    should have the same "CODE" to them. Exact offset does not matter.
    input:
        cpt    = file with point shapes from which lines should be
                    drawn, this should contain an unique_ attribute (.shp)
                    full path
        k_lijn = file with the keringslijn shapefile (lines), this
                    should contain a CODE attribute (.shp)
                    full path
        off_lijn = file with the offset keringslijn shapefile (lines), this
                    should contain a CODE attribute and be orientated
                    "outwards"(.shp)
                    full path
        m_buitenw = in meters how much the dwp should be orientated outwards
                    (toward the river)
        m_binnenw = in meters how much the dwp should be orientated inwards
                    (inland)

    output:
        data = geopandas dataframe with the calculated dwp as
        LineString geoemtries
    """
    # shapefiles inladen
    data = gpd.read_file(cpt).set_crs(28992)
    data.rename(columns={'CODE': 'CODEdijkr'}, inplace=True)
    data['unique_'] = [f"{x}_{y.replace(',', 'p')}" for x, y
                       in zip(data.CODEdijkr, data.NAAM)]
    data.unique_.drop_duplicates(inplace=True)
    data_klijn = gpd.read_file(k_lijn).set_crs(28992)
    data_offlijn = gpd.read_file(off_lijn).set_crs(28992)

    # Bepaal data kruinlijn
    data_k = _determine_nearest(data, data_klijn)
    # Bepaal data offset lijn
    data_off = _determine_nearest(data, data_offlijn)

    # add information of nearby points tot
    data['NEAR_CODE'] = 'not_valid'
    data['NEAR_X'] = np.nan
    data['NEAR_Y'] = np.nan
    data['NEAR_DIST'] = np.nan
    data['NEAR_ANGLE'] = np.nan
    data['NEAR_CODE_'] = 'not_valid'
    data['NEAR_X_OFF'] = np.nan
    data['NEAR_Y_OFF'] = np.nan
    data['NEAR_DIST_'] = np.nan
    data['NEAR_ANG_1'] = np.nan

    for i, j in zip(data_k, data_off):
        # add points on keringlijn
        data.loc[data.unique_ == i[0], 'NEAR_CODE'] = i[1]
        data.loc[data.unique_ == i[0], 'NEAR_DIST'] = i[2]
        data.loc[data.unique_ == i[0], 'NEAR_X'] = i[3]
        data.loc[data.unique_ == i[0], 'NEAR_Y'] = i[4]
        data.loc[data.unique_ == i[0], 'NEAR_ANGLE'] = i[5]
        # add points on offset keringslijn
        data.loc[data.unique_ == j[0], 'NEAR_CODE_'] = j[1]
        data.loc[data.unique_ == j[0], 'NEAR_DIST_'] = j[2]
        data.loc[data.unique_ == j[0], 'NEAR_X_OFF'] = j[3]
        data.loc[data.unique_ == j[0], 'NEAR_Y_OFF'] = j[4]
        data.loc[data.unique_ == j[0], 'NEAR_ANG_1'] = j[5]

    # assign new point geometry based on NEAR_X and NEAR_Y
    # so it starts on the keringslijn
    point_ = []
    for idx, r in data.iterrows():
        point_.append(Point(r.NEAR_X, r.NEAR_Y))

    data = data.set_geometry(point_)

    geometry_ = []
    for idx, r in data.iterrows():
        # punt buiten keringslijn met klein hoek
        if (r.NEAR_DIST < r.NEAR_DIST_) and (abs(r.NEAR_ANGLE-r.NEAR_ANG_1)
                                             < 90):
            x1 = r.geometry.x+m_binnenw*math.sin(math.radians(r.NEAR_ANGLE
                                                              - 180))
            y1 = r.geometry.y+m_binnenw*math.cos(math.radians(r.NEAR_ANGLE
                                                              - 180))
            x2 = r.geometry.x+m_buitenw*math.sin(math.radians(r.NEAR_ANGLE))
            y2 = r.geometry.y+m_buitenw*math.cos(math.radians(r.NEAR_ANGLE))
            geometry_.append(LineString([(x1, y1), (x2, y2)]))
            continue
        if (r.NEAR_DIST > r.NEAR_DIST_) and (abs(r.NEAR_ANGLE-r.NEAR_ANG_1)
                                             < 90):
            x1 = r.geometry.x+m_binnenw*math.sin(math.radians(r.NEAR_ANGLE))
            y1 = r.geometry.y+m_binnenw*math.cos(math.radians(r.NEAR_ANGLE))
            x2 = r.geometry.x+m_buitenw*math.sin(math.radians(r.NEAR_ANGLE
                                                              - 180))
            y2 = r.geometry.y+m_buitenw*math.cos(math.radians(r.NEAR_ANGLE
                                                              - 180))
            geometry_.append(LineString([(x1, y1), (x2, y2)]))
            continue
        if abs(r.NEAR_ANGLE-r.NEAR_ANG_1) >= 90:
            x1 = r.geometry.x+m_binnenw*math.sin(math.radians(r.NEAR_ANGLE))
            y1 = r.geometry.y+m_binnenw*math.cos(math.radians(r.NEAR_ANGLE))
            x2 = r.geometry.x+m_buitenw*math.sin(math.radians(r.NEAR_ANG_1))
            y2 = r.geometry.y+m_buitenw*math.cos(math.radians(r.NEAR_ANG_1))
            geometry_.append(LineString([(x1, y1), (x2, y2)]))
            continue

    data = data.set_geometry(geometry_)

    return data


def raster_data_to_df(
        dwp: GeoDataFrame,
        rasterdate_file,
        shape_group: str,
        csv_folder: str,
        tiff_folder: str,
        ):
    """
    Extract bathymetry data for one DWP and save to CSV.
    savef_ is the folder for AHN TIFFs; CSVs are written one folder up.
    """

    # --- Download the AHN raster for this DWP
    ahn_path = ahn_per_dwp(dwp, tiff_folder)

    # --- Open the AHN raster and extract profile points
    with rasterio.open(ahn_path) as src_ahn:
        ahn_transform = src_ahn.transform
        point_distance = ahn_transform.a
        polyline_geom = dwp.geometry
        polyline_length = polyline_geom.length

        # Create points along the line at AHN resolution
        points = [
            polyline_geom.interpolate(i * point_distance)
            for i in range(0, int(polyline_length / point_distance))
        ]
        points_gdf = gpd.GeoDataFrame(geometry=points)

        # Get AHN elevation for each point
        pixel_values_ahn = []
        band = src_ahn.read(1)
        for x_geo, y_geo in zip(points_gdf.geometry.x, points_gdf.geometry.y):
            x_pixel, y_pixel = _geo_to_pixel(x_geo, y_geo, ahn_transform)
            try:
                pixel_value_ahn = band[int(y_pixel), int(x_pixel)]
            except Exception:
                pixel_value_ahn = np.inf
            pixel_values_ahn.append(pixel_value_ahn)

    # --- Build DataFrame
    df = pd.DataFrame({
        "xRD": round(points_gdf.geometry.x, 4),
        "yRD": round(points_gdf.geometry.y, 4),
        "ahn": pixel_values_ahn
    })

    # --- Read bathymetry rasters and add them as columns
    list_raster = _read_rasterfile(rasterdate_file)
    for idx, (timestamp, raster_path) in enumerate(list_raster.items()):
        try:
            with rasterio.open(os.path.normpath(raster_path)) as src:
                transform = src.transform
                band = src.read(1)
                pixel_values = []
                for x_geo, y_geo in zip(points_gdf.geometry.x,
                                        points_gdf.geometry.y):
                    x_pixel, y_pixel = _geo_to_pixel(x_geo, y_geo, transform)
                    try:
                        pixel_value = band[int(y_pixel), int(x_pixel)]
                    except Exception:
                        pixel_value = np.inf
                    pixel_values.append(pixel_value)

            tdt = dt.datetime.fromtimestamp(timestamp)
            df[f"{tdt.year}-{tdt.month}-{tdt.day}"] = pixel_values
        except Exception as e:
            print(f"⚠️ Could not read raster {raster_path}: {e}")

    # --- Clean up invalid values
    df[df > 1E6] = np.inf
    df[df < -100] = np.inf

    # --- Add distance from crest (simplified, same as before)
    pt = Point(dwp.NEAR_X, dwp.NEAR_Y)
    sindex = points_gdf.sindex
    nearest_idx = int(sindex.nearest(pt)[1])
    hulp_values = np.empty(len(points))
    hulp_values[:] = np.nan
    hulp_values[0:nearest_idx] = -1
    hulp_values[nearest_idx] = 0
    hulp_values[nearest_idx+1::] = 1
    nul_values = [
        round(np.sqrt((x - pt.x)**2 + (y - pt.y)**2), 2) * s
        for (x, y, s) in zip(df.xRD, df.yRD, hulp_values)
    ]
    df["meters"] = nul_values

    # --- Define CSV name
    if shape_group == "boezem_rechts":
        name_csv = f"dp_cpt_{dwp.unique_}_Rechts.csv"
    elif shape_group == "boezem_links":
        name_csv = f"dp_cpt_{dwp.unique_}_Links.csv"
    elif shape_group == "primair":
        name_csv = f"dp_cpt_{dwp.unique_}_Primair.csv"
    else:
        name_csv = f"dp_cpt_{dwp.unique_}_Unknown.csv"

    # --- Save CSV in explicit CSV folder (03_Bathymetry) ---
    csv_path = os.path.join(csv_folder, name_csv)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")
    return df, csv_path
