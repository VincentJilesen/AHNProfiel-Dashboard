import pandas as pd
import numpy as np 
import os
import datetime as dt
import geopandas as gpd
from geopandas import GeoDataFrame
import shapely as shpl
from shapely import LineString, Point
from shapely.ops import nearest_points
from owslib.wcs import WebCoverageService
from osgeo import gdal
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings

#import geolib_plus as glp
#from geolib_plus.bro_xml_cpt import BroXmlCpt
"""
Geometric functions:
-   generate line crossing kering, should be based on created near points with 
    the keringslijn and an offset buitenwaards (ex. 5m). This gives the correct
    direction of the line. Also as an input should be given points from where the
    dwp's are drawn.
"""
gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG") # to overcome errors EPSG
gdal.DontUseExceptions()

warnings.filterwarnings("ignore", category=DeprecationWarning) # no warnings..
warnings.filterwarnings("ignore", category=UserWarning) # no warnings..

def angle_between_points(point1, point2):
    """
    determine angle between to geometry points (two input point geometrys as input)
    should be in the same coordinate convention.
    """
    # Calculate the differences in coordinates
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    # Calculate the angle in radians
    angle = 90-math.degrees(math.atan2(dy, dx)) 
    #!   the 90 degrees is difference planar and geodesic convention of angle

    return angle

def geo_to_pixel(x_geo, y_geo, transform):
    """
    function which translates a array coordinate to geo coordinate
    """
    x_pixel = int((x_geo - transform[0]) / transform[1])
    y_pixel = int((y_geo - transform[3]) / transform[5])
    return x_pixel, y_pixel

def determine_nearest(data_p, data_n):
    """
    determines the nearest geo "line" to geo "point" and gives back 
    the coordinates of this point (also which line id) and angle from 
    the original point.
    """
    # bepaald dichtsbijzijnde punt op lijn
    nearest = gpd.sjoin_nearest(data_p,data_n, distance_col='dist_')
    # Find the indices of the smallest 'dist_' for each 'unique_'
    nearest = nearest.sort_values(by=['unique_','dist_'],ascending=True)
    idx = nearest.groupby('unique_')['dist_'].idxmin()
    filtered_nearest = nearest.loc[idx]

    near=[]
    
    for i in filtered_nearest.iterrows():
        try: # to avoid errors in the database of the keringlijn...
            temp=[]
            temp.append(i[1].unique_)
            temp.append(i[1].CODE)
            # haal lijn op die het dichtstbij ligt
            line=data_n[data_n.CODE==i[1].CODE].geometry.iloc[0]
            # haal punt op die het dichtstbij ligt
            point=data_p[data_p.unique_==i[1].unique_].geometry.iloc[0]
            # bepaal punt op lijnmo
            ## afstand tot lijn
            temp.append(i[1].dist_) 
            ## punt op lijn (haal x, y op)
            n_point=nearest_points(point,line)[1]
            temp.append(n_point.x)
            temp.append(n_point.y)
            ## bepaal hoek tussen punt en near punt
            temp.append(angle_between_points(point,n_point))
            near.append(temp)
        except:
            print(data_n[data_n.CODE==i[1].CODE])
            continue
    
    # Remove duplicates based on the first element (unique_) if 'Dijkring_17_0p0' is not unique
    df_near = pd.DataFrame(near, columns=['unique_', 'CODE', 'dist_', 'x', 'y', 'angle'])
    if df_near['unique_'].duplicated().any():
        df_near = df_near.drop_duplicates(subset=['unique_'])
    near = df_near.values.tolist()
        
    return near

def generate_dwp(cpt, k_lijn, off_lijn, m_buitenw, m_binnenw):
    """
    Function to generate dwp from cpt (points) perpendicular on keringlijn (line). 
    Using an offset line with the right orientation (toward the rivers in this case, buitenwaards).
    The offset and lines should have the same "CODE" to them. Exact offset does not matter.
    input:
        cpt    = file with point shapes from which lines should be 
                    drawn, this should contain an unique_ attribute (.shp)
                    full path
        k_lijn = file with the keringslijn shapefile (lines), this 
                    should contain a CODE attribute (.shp)
                    full path
        off_lijn = file with the offset keringslijn shapefile (lines), this 
                    should contain a CODE attribute and be orientated "outwards"(.shp)
                    full path
        m_buitenw = in meters how much the dwp should be orientated outwards 
                    (toward the river)
        m_binnenw = in meters how much the dwp should be orientated inwards 
                    (inland)

    output:
        data = geopandas dataframe with the calculated dwp as LineString geoemtries
        
    """
    # shapefiles inladen
    data=gpd.read_file(cpt).set_crs(28992)
    data.rename(columns={'CODE': 'CODEdijkr'}, inplace=True)
    data['unique_']=[f"{x}_{y.replace(',','p')}" for x,y in zip(data.CODEdijkr, data.NAAM)]
    data.unique_.drop_duplicates(inplace=True)
    data_klijn=gpd.read_file(k_lijn).set_crs(28992)
    data_offlijn=gpd.read_file(off_lijn).set_crs(28992)

    # Bepaal data kruinlijn
    data_k=determine_nearest(data, data_klijn)
    # Bepaal data offset lijn 
    data_off=determine_nearest(data, data_offlijn)

    # add information of nearby points tot 
    data['NEAR_CODE'],data['NEAR_X'],data['NEAR_Y'],data['NEAR_DIST'],data['NEAR_ANGLE']='not_valid',np.nan,np.nan,np.nan,np.nan
    data['NEAR_CODE_'],data['NEAR_X_OFF'],data['NEAR_Y_OFF'],data['NEAR_DIST_'],data['NEAR_ANG_1']='not_valid',np.nan,np.nan,np.nan,np.nan

    for i,j in zip(data_k, data_off):
        #add points on keringlijn
        data.loc[data.unique_==i[0],'NEAR_CODE']=i[1]
        data.loc[data.unique_==i[0],'NEAR_DIST']=i[2]
        data.loc[data.unique_==i[0],'NEAR_X']=i[3]
        data.loc[data.unique_==i[0],'NEAR_Y']=i[4]
        data.loc[data.unique_==i[0],'NEAR_ANGLE']=i[5]
        #add points on offset keringslijn
        data.loc[data.unique_==j[0],'NEAR_CODE_']=j[1]
        data.loc[data.unique_==j[0],'NEAR_DIST_']=j[2]
        data.loc[data.unique_==j[0],'NEAR_X_OFF']=j[3]
        data.loc[data.unique_==j[0],'NEAR_Y_OFF']=j[4]
        data.loc[data.unique_==j[0],'NEAR_ANG_1']=j[5]

    # assign new point geometry based on NEAR_X and NEAR_Y so it starts on the keringslijn
    point_=[]
    for idx, r in data.iterrows():
        point_.append(Point(r.NEAR_X, r.NEAR_Y))

    data=data.set_geometry(point_)
    
    geometry_=[]
    for idx, r in data.iterrows():
        #punt buiten keringslijn met klein hoek 
        if (r.NEAR_DIST < r.NEAR_DIST_) and (abs(r.NEAR_ANGLE-r.NEAR_ANG_1)<90):
            x1=r.geometry.x+m_binnenw*math.sin(math.radians(r.NEAR_ANGLE-180))
            y1=r.geometry.y+m_binnenw*math.cos(math.radians(r.NEAR_ANGLE-180))
            x2=r.geometry.x+m_buitenw*math.sin(math.radians(r.NEAR_ANGLE))
            y2=r.geometry.y+m_buitenw*math.cos(math.radians(r.NEAR_ANGLE))
            geometry_.append(LineString([(x1,y1), (x2,y2)]))
            continue
        if (r.NEAR_DIST > r.NEAR_DIST_) and (abs(r.NEAR_ANGLE-r.NEAR_ANG_1)<90):
            x1=r.geometry.x+m_binnenw*math.sin(math.radians(r.NEAR_ANGLE))
            y1=r.geometry.y+m_binnenw*math.cos(math.radians(r.NEAR_ANGLE))
            x2=r.geometry.x+m_buitenw*math.sin(math.radians(r.NEAR_ANGLE-180))
            y2=r.geometry.y+m_buitenw*math.cos(math.radians(r.NEAR_ANGLE-180))
            geometry_.append(LineString([(x1,y1), (x2,y2)]))
            continue
        if abs(r.NEAR_ANGLE-r.NEAR_ANG_1)>=90:
            x1=r.geometry.x+m_binnenw*math.sin(math.radians(r.NEAR_ANGLE))
            y1=r.geometry.y+m_binnenw*math.cos(math.radians(r.NEAR_ANGLE))
            x2=r.geometry.x+m_buitenw*math.sin(math.radians(r.NEAR_ANG_1))
            y2=r.geometry.y+m_buitenw*math.cos(math.radians(r.NEAR_ANG_1))
            geometry_.append(LineString([(x1,y1), (x2,y2)]))
            continue

    data=data.set_geometry(geometry_)

    return data

def ahn_per_dwp(dwp: GeoDataFrame, ouputfolder=os.getcwd(), filename='wcs_ahn_temp.tiff'):
    """
    To retrieve the ahn height map (wcs) per dwp (line element). This will be
    saved in a temp file for later use
    input:
    - dwp          : the geometry of the dwarsprofiel (line, can be other geometry type).
    - outputfolder  : the folder where the temp file is stored (optional)
    - filename      : name of ahn rasterfile (optional)
    output:
    - tiff file with the ahn raster for the dwp
    - returns path+file 
    """
    # actuele ahn (if a different ahn is needed, this link should be changed)
   
    url_wcs=r'https://service.pdok.nl/rws/ahn/wcs/v1_0?request=GetCapabilities&service=WCS'
   
    #load wcs in model
    wcs = WebCoverageService(url_wcs)

    bounds=dwp.geometry.bounds
    bounds_rounded=(
                math.floor(bounds[0]),
                math.floor(bounds[1]),
                math.ceil(bounds[2]),
                math.ceil(bounds[3])
            )
    wcs = WebCoverageService(url_wcs)       
    output = wcs.getCoverage(
                            identifier=['dtm_05m'],
                            format='geotiff',
                            crs='EPSG:28992',
                            subsets = [('X', bounds_rounded[0], bounds_rounded[2]), ('Y', bounds_rounded[1], bounds_rounded[3])],
                            width=(math.ceil(bounds[2])-math.floor(bounds[0]))/2,
                            height=(math.ceil(bounds[3])-math.floor(bounds[1]))/2,
                        )
    
    try:
        if filename.split('.')[-1]!='tiff':
            filename = str(filename.split('.')[0]+'.tiff')
    except:
        filename=filename

    with open(os.path.join(ouputfolder, filename), 'wb') as ahn:
        ahn.write(output.read())

    return os.path.join(ouputfolder, filename)

def read_rasterfile(rasterdate_file):
    """
    Read the file which has defined the rasters and date of execution
    input:
    - rasterdate_file : path + filename (.txt file)
    output:
    - list_raster (dict): key = date of raster : value = path+file
    """
    #date_file="dates_rasters_17-2_re_test.txt"
    #currentd_data=r'C:\Users\pauno\Desktop\lokale_map_paul\Eenvoudige_toets2023\data'

    #mappen doorlopen aangegeven in 'dates_rasters.txt', voor zoeken straks
    list_raster={}
    #mappen doorlopen aangegeven in 'dates_rasters.txt', voor zoeken straks 
    ### -> verwijzen naar juist inputfile? Dit per traject? zit dat in de dwarssdoorsnede verwerkt?
    with open(os.path.join(rasterdate_file)) as peilingen:
        lines=peilingen.readlines()
        for d in range(len(lines)): 
            a=[]
            a=lines[d].split(';')
            paths=a[3].replace('\n','')

            date1=dt.datetime(int(a[0]),int(a[1]),int(a[2].replace('\n','')))
            date1=int(date1.timestamp())
            list_raster.update({date1 : paths})

    return list_raster

def step1_raster_data_to_df(dwp: GeoDataFrame, # use function generate_dwp(cpt, k_lijn, off_lijn, m_buitenw, m_binnenw)
                            rasterdate_file,
                            shape_group:str, 
                            savef_=None,
                            ken_punten=None, # puntenshape per profiele met punten op buitentaludvlak
                            ):
    """
    Function to place bathymetry data into a dataframe (and save as csv)
    - dwp   : 
    """
    # # make seperate folder for bathymetry files
    # try:
    #     os.mkdir(os.path.join(savef,'bathymetry'))
    #     savef_=os.path.join(savef,'bathymetry')
    # except:
    #     savef_=os.path.join(savef,'bathymetry')
    
    # lees AHN tiff bestand in via functie ahn_per_dwp
    src_ahn=gdal.Open(ahn_per_dwp(dwp, savef_))
    # ophalen transformatei stelsel van raster (resolutie etc)
    ahn_transform=src_ahn.GetGeoTransform()
    # ophalen coordinate stelsel raster
    ahn_crs=src_ahn.GetProjection()
    
    #### leeg dataframe voor wegschrijven data.
    df=pd.DataFrame()
    # Get the first geometry in the GeoDataFrame
    polyline_geom=dwp.geometry
    # Get the length of the polyline
    polyline_length=polyline_geom.length
    # distance between points on line (same as resolution raster ahn)
    #       this is normally 0,5m; our underwater bathymetry data is also this resolution
    #       be carefull if this is not your case.
    point_distance=ahn_transform[1]
    
    # generate points on line (temp input for height reading function)
    points=[]

    for distance_along_polyline in range(0, int(polyline_length/point_distance), 1):
            point = polyline_geom.interpolate(distance_along_polyline*point_distance)
            points.append(point)

    points_gdf = gpd.GeoDataFrame(geometry=points)

    # get AHN height for all points on dwp
    pixel_values_ahn=[]
    for x_geo, y_geo in zip(points_gdf.geometry.x, points_gdf.geometry.y):
        x_pixel, y_pixel = geo_to_pixel(x_geo, y_geo, ahn_transform)
        pixel_value_ahn = src_ahn.ReadAsArray(x_pixel, y_pixel, 1, 1)[0, 0]
        pixel_values_ahn.append(pixel_value_ahn)
            
    df['xRD']= round(points_gdf.geometry.x,4)
    df['yRD']= round(points_gdf.geometry.y,4)
    df['ahn']= pixel_values_ahn

    # get height points per underwaterbathymetry and add each to df
    #for v in range(len(dat1)): # per raster
    list_raster=read_rasterfile(rasterdate_file)
    nr=0
    for k,v in list_raster.items():      
        # Iterate through each point and get the pixel value
        src=gdal.Open(os.path.normpath(v))
        transform=src.GetGeoTransform()
        crs=src.GetProjection()
        globals()[f'pixel_values{nr}'] = []
        for x_geo, y_geo in zip(points_gdf.geometry.x, points_gdf.geometry.y):
            x_pixel, y_pixel = geo_to_pixel(x_geo, y_geo, transform)
            try:
                pixel_value = src.ReadAsArray(x_pixel, y_pixel, 1, 1)[0, 0]
            except Exception as e:
                #print(f"Error reading pixel value at ({x_geo}, {y_geo}): {e}")
                pixel_value = np.inf
            globals()[f'pixel_values{nr}'].append(pixel_value)
    
        tdt=dt.datetime.fromtimestamp(k)
        df[f'{tdt.year}-{tdt.month}-{tdt.day}'] = globals()[f'pixel_values{nr}']
        nr=+1 # add based on length list_raster

    # cleanup NoData values dataframe for further processing
    df[df>1E+6]=np.Inf
    df[df<-100]=np.Inf
    # genereer spatial index van punten op dwp, dit om losse punten te referen
        # dit zijn bijvoorbeeld de positie van de kruinlijn 
        # of andere (ex insteek rivier etc)
    sindex = points_gdf.sindex # spatial index

    # geometrisch punt welke moet worden opgezocht (als 0 punt)
        # dwp.NEAR = snijpunt met dwp met kruinlijn
        # andere later toevoegen...
    pt=Point(dwp.NEAR_X,dwp.NEAR_Y)
    # kermerkende punten (dataframe) op basis van profielvlak; punten zijn al met interesect functie arcgis gemaakt.
    # df['ken_punten']=np.nan # lege column voor ken_punten

    # if ken_punten is not None:
    #     # ken_punten is een geopandas dataframe met punt geometries
    #     # deze moeten worden toegevoegd aan de dataframe
    #     for i in range(len(ken_punten)):
    #         # bepaal punt op dwp
    #         pt_ken=Point(ken_punten.geometry.x.iloc[i], ken_punten.geometry.y.iloc[i])
    #         # bepaal index van punt op dwp
    #         nearest_idx_ken = int(sindex.nearest(pt_ken)[1])
    #         # voeg toe aan dataframe
    #         df.loc[nearest_idx_ken,'ken_punten']=1

    # dichtsbijzijnden punt op dwp vinden
    nearest_idx = int(sindex.nearest(pt)[1])

    #create empty voor column voor absolute afstand binnen- naar buitenwaards vanaf gegeven punt
    hulp_values = np.empty(int(polyline_length/point_distance))
    hulp_values[:] = np.nan

    hulp_values[0:nearest_idx] = -1 # negatieve afstand binnenwaards
    hulp_values[nearest_idx] = 0 # kruin
    hulp_values[nearest_idx+1::] = 1 # positive afstand buitenwaards
    
    #bereken absolute afstand vanaf punt (niet nearest idx x en y; het punt ligt op de lijn)
    nul_values = []
    null_values = np.empty(int(polyline_length/point_distance))
    null_values = [round((np.sqrt(abs(xrd-pt.bounds[0])**2+abs(yrd-pt.bounds[1])**2)),2) for xrd,yrd in zip(df.xRD, df.yRD)]        
    for xtt,ytt in zip(null_values,hulp_values):
        nul_values.append(xtt * ytt)

    # toevoegen absolute afstanden column vanaf gegeven punt aan dataframe    
    df['meters'] = nul_values

    # # toevoegen ahn hoogte aan btalud_punt
    # df['btalud_punt'] = [x*y for x,y in zip(df['ken_punten'],df['ahn'])]

    #verwijder lege datum columns (waar er geen data is vanuit gegeven rasters)
    for l,ll in df.items():
        if df[f'{l}'].isnull().values.all():
            df.drop([f'{l}'],inplace=True, axis=1)

    # columns met lodingen
    df_idx=df.loc[:,df.columns.str.contains(r'(\d{4}-\d{1,2}-\d{1,2})')]
    # alle andere columns
    df_rest=df.loc[:,~df.columns.str.contains(r'(\d{4}-\d{1,2}-\d{1,2})')]
    
    # als er overlappende rasters zijn met dezelfde datum middelen en 1 col vormen
    for name, group in df_idx.T.groupby(by=df_idx.columns):
        if len(group.T.columns)>1:
            df_idx[f'{name}']=[np.mean(x[1]) for x in group.T.iterrows()]   
        else:
            continue

    # make sure the columns are chronological
    cols_as_date = [dt.datetime.strptime(x,'%Y-%m-%d') for x in df_idx.columns]
    df_idx = df_idx[[f'{x.year}-{x.month}-{x.day}' for x in sorted(cols_as_date)]]

    # weer bij elkaar voegen van de dataframe
    df=df_idx.join(df_rest)

    # opslaan data als CSV
    folder=os.path.split(ahn_per_dwp(dwp, savef_))[0]
    
    if shape_group == "boezem_rechts":
        name_csv=f'dp_cpt_{dwp.unique_}_Rechts.csv'
    elif shape_group ==  "boezem_links":
        name_csv=f'dp_cpt_{dwp.unique_}_Links.csv'
    elif shape_group == "primair":
        name_csv = f"dp_cpt_{dwp.unique_}_Primair.csv" 
    print(dwp.unique_)
    
    path_to_csv=str(folder+name_csv)
    df.to_csv(os.path.join(folder,name_csv))
    
    return df, path_to_csv

def plot_dwp(path_to_csv, rasterdate_file, save_plot=False):
    # read csv with data per dwp
    dwp_df=pd.read_csv(os.path.normpath(path_to_csv))
    name_plot=os.path.split(path_to_csv)[-1].replace('dp_cpt_','').replace('.csv','')

    # get all columns with bathymetry data
    df=dwp_df.loc[:,dwp_df.columns.str.contains(r'(\d{4}-\d{1,2}-\d{1,2})')]
    # drop columns with only np.inf values
    df=df.loc[:,~df.isin([np.inf]).all()]

    list_raster=read_rasterfile(rasterdate_file)

    fig, ax = plt.subplots(figsize=(50,10))
    ax.set_title(f'Dwarsdoorsnede plot raai CPT {name_plot}',fontsize=20)
    #generate legend
    legenda=[]
    [legenda.append(o) for o in df]
    legenda.append('AHN')
    #generate labels
    label1=[]
    for z in df.columns:
        z_=int(dt.datetime.strptime(z, "%Y-%m-%d").timestamp())
        for x,t in list_raster.items():
            if z_ == int(x):
                tdt=dt.datetime.fromtimestamp(x)
                label1.append(f'{tdt.year}-{tdt.month}-{tdt.day}'+": "+t.split("\\")[-1].replace('_re.tif',''))
   
    ax.set_xlabel('Afstand tot vaargeul [m]')
    ax.set_ylabel('Gemeten diepte [mNAP]')
    ax.set_xticks(np.arange(min(dwp_df['meters']),max(dwp_df['meters']),20))
    ax.set_yticks(np.arange(-20,10,1)) 
    np.random.seed(1) # zodat elke figuur dezelfde kleuren pallet aanhoudt

    # for z in enumerate(df.join(dwp_df['ahn'])): # per raster 
    #     col = (np.random.random(), np.random.random(), np.random.random()) 
    #     ax.plot(dwp_df['meters'], dwp_df[f'{z[1]}'], color=col, label=label1[z[0]])
    for g in enumerate(df): # per raster 
        col = (np.random.random(), np.random.random(), np.random.random()) 
        ax.plot(dwp_df['meters'], df[f'{g[1]}'], color=col, label=label1[g[0]])
    # add ahn data
    ax.plot(dwp_df['meters'], dwp_df['ahn'], color=(0, 0, 0), label='AHN')
    
    legend1=ax.legend(loc='lower left')
    plt.gca().add_artist(legend1) # so a second one can be added 

    ax.plot([0,0],[-20,9], color='g')
    bbox = {'fc': 'green', 'pad':3}
    ax.text(0.5,6.5,'Kruinlijn WSHD',color='white',rotation=0,bbox=bbox)
    
    # # plotten van buitenkruin en buiten(ste)teen
    # ax.plot(dwp_df['meters'], dwp_df['btalud_punt'],color='b', marker='o', label='Buitentaludpunt')

    ax.grid()
    if save_plot==True:
        fig.savefig(path_to_csv.replace('.csv','.png'),  dpi=300, bbox_inches='tight')
    else:
        pass
    plt.close()

    return fig

def zvg_vs_bathy(dwp: GeoDataFrame, csv_cpt: str, csv_bathy: str, rasterdate_file: str, min_ld: float=0, save_plot=False):
    """
    Function to plot the zvg data (cpt) and bathymetry data (dwp) in one figure

    needs checking?
    """
    #load csv cpt data
    cpt=pd.read_csv(csv_cpt, index_col=0)
    cpt=cpt[['mNAP', 'D_r', 'class_zvg', 'laagdikte']]
    #load csv dwp bathymetry data
    bath=pd.read_csv(csv_bathy, index_col=0)

    # haal alleen bathymetry columns op
    bath2=bath.loc[:,bath.columns.str.contains(r'(\d{4}-\d{1,2}-\d{1,2})')]
    
    # haal alleen positieve waardes lengte richting ahn op en voeg toe
    bath2['ahn_pos']=[x[1]['ahn'] if x[1]['meters']>=0 else np.inf for x in bath.iterrows()] # add ahn_pos]
    
    # maak column aan die de laagste waarde per afstand geven
    # Find the lowest bathymetry value for each 'meters' value
    low_value=[]
    for x in bath2.iterrows():
        if all(x[1].values==np.inf):
            low_value.append(np.inf)
            continue
        if x[1].min()>=low_value[-1]:
            low_value.append(low_value[-1])
            continue
        else:
            low_value.append(x[1].min())
            
    bath2['low_bath'] = low_value
    bath['low_bath'] = low_value
    
    # drop the ahn_pos column; otherwise always lowest value...
    bath2.drop(columns=['ahn_pos'], inplace=True)
    # load matplotlib fig and continue editting "plot_dwp function"
    fig=plot_dwp(csv_bathy, rasterdate_file, save_plot=False)

    dict_zvg={
            'SC':"#993366",
            'TC':"#97071d",
            'CC':"#685fba",
            'SD':"#758719",
            'TD':"#CD853f",
            'CD':"#cccc00",
            'CCS':"#de2a36",
        }

    if (dwp.NEAR_DIST < dwp.NEAR_DIST_) and (abs(dwp.NEAR_ANGLE-dwp.NEAR_ANG_1)<90):
        side=-1
    else:
        side=1

    def find_non_empty_groups(df, column_name):
        values = df[column_name].values
        groups = []
        start_idx = None
        for i in range(len(values)):
            if not pd.isna(values[i]):
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    groups.append([start_idx, i - 1])
                    start_idx = None
        if start_idx is not None:
            groups.append([start_idx, len(values) - 1])
        return groups

    # Apply the function
    groups = find_non_empty_groups(cpt, 'laagdikte')

    depth=[]
    for i in groups:

        nearest_idx_min=(bath2['low_bath'] - cpt.mNAP.iloc[i[0]]).abs().idxmin()
        nearest_idx_max=(bath2['low_bath'] - cpt.mNAP.iloc[i[1]]).abs().idxmin()
        if cpt.laagdikte.iloc[i[1]]>=min_ld:
            depth.append(
                {'class_zvg': cpt.class_zvg.iloc[i[1]],
                'start_d_cpt': cpt.mNAP.iloc[i[0]], 
                'end_d_cpt': cpt.mNAP.iloc[i[1]],
                'ld_d_cpt': cpt.laagdikte.iloc[i[1]],
                'total_d_cpt': abs(cpt.mNAP.iloc[i[1]]-cpt.mNAP.iloc[i[0]]),
                'depth_s_d': bath2.low_bath.iloc[nearest_idx_min],
                'depth_e_d': bath2.low_bath.iloc[nearest_idx_max],
                'dist_s_d': bath.meters.iloc[nearest_idx_min],
                'dist_e_d': bath.meters.iloc[nearest_idx_max]},
            )
        else:
            continue
        
    for i in depth:
        vertices = np.array([
                            [dwp.NEAR_DIST*side, i['start_d_cpt']], 
                            [i['dist_s_d'], i['start_d_cpt']], 
                            [i['dist_e_d'], i['end_d_cpt']], 
                            [dwp.NEAR_DIST*side, i['end_d_cpt']], 
                            ])

        # Create and add the filled polygon
        face_c=dict_zvg[i['class_zvg']]
        polygon = plt.Polygon(
                            vertices, 
                            closed=True, 
                            fill=True, 
                            edgecolor='none', 
                            facecolor=face_c)
        fig.gca().add_patch(polygon)

    # plot CPT in figure
    
    fig.gca().plot([dwp.NEAR_DIST*side,dwp.NEAR_DIST*side],[-20,9], color='r', lw=1, ls='--')	
    bbox = {'fc': 'white', 'pad':3}

    fig.gca().text(dwp.NEAR_DIST*side,8,f'{dwp.unique_}',color='red',rotation=0,bbox=bbox)
       
    # Create custom legend handles
    handles = [plt.Line2D([0], [0], color=color, lw=5) for color in dict_zvg.values()]
    labels = list(dict_zvg.keys())

    # Add the custom legend
    fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.9, 0.85), title=f'ZV gevoelige lagen >{min_ld} m')
    
    if save_plot==True:
        fig.savefig(csv_bathy.replace('.csv','_zvg.png'),bbox_inches='tight')
    else:
        pass
    plt.close()

    return depth, bath, fig

def vlzv_ET(dwp: GeoDataFrame, 
                csv_cpt: str, 
                csv_bathy: str, 
                rasterdate_file: str, 
                min_ld: float=0, 
                radius: float=2.5, # is standaard bij deze toets
                min_height: float=7.5, # minimale verschilhoek voor knikpunt zoeken (karakteristieke punten)
                max_height: float=45, # maximale verschilhoek voor knikpunt zoeken (karakteristieke punten) 
                save_plot=False):

    toets=[]
    # inport bare figure for plotting E_1 results
    depth, bath, fig = zvg_vs_bathy(dwp, csv_cpt, csv_bathy, rasterdate_file, min_ld=min_ld, save_plot=False)
    
    
    radius=2.5 # radius of search (for average angle calculation)
    kpunt=[]

    # k punten buitentalud ophalen uit bath
    bath_filt = bath[bath['ken_punten'].notna()]# filter rijen alleen de buitentaludpunten
    
    # Filter bath to only rows where 'ken_punten' is not NaN
    if len(bath_filt)==1:
        # if only one point is found, add it to the kpunt list
        x=bath_filt.meters.iloc[0]
        y=bath_filt.low_bath.iloc[0]
        kpunt.append({
            'id0' : bath.meters[bath.meters==x].index[0],
            'meters': x,
            'low_bath': y,
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
            })
    elif len(bath_filt)>=2:
    # if more than two points are found
        x=bath_filt.meters.iloc[0] # firt point
        y=bath_filt.low_bath.iloc[0]
        kpunt.append({
            'id0' : bath.meters[bath.meters==x].index[0],
            'meters': x,
            'low_bath': y,
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
            })
        x=bath_filt.meters.iloc[1] # second point
        y=bath_filt.low_bath.iloc[1]
        kpunt.append({
            'id0' : bath.meters[bath.meters==x].index[0],
            'meters': x,
            'low_bath': y,
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
            })
    else:
        kpunt.append({
            'id0' : bath.meters[bath.meters==0].index[0],
            'meters': x,
            'low_bath': y,
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
            }) 

# # -1 m punt
    llws=-1 # aangenomen LLWS benedenrivieren
    idx_llws = (bath.low_bath - llws).abs().idxmin() # index van laagste punt rond LLWS
    # change second point to extend with an angle to llws
    if len(kpunt)==1:
        # calculate the angle between the first and second point
        angle_b = math.atan2(-1,4)
    
        # change the second point to extend with an angle to llws
        x_new = kpunt[0]['meters'] + (-(kpunt[0]['low_bath']-bath.low_bath.iloc[idx_llws])/math.tan(angle_b))
        
        y_new = llws
        kpunt[1] = ({
            'id0': 999,
            'meters': x_new,
            'low_bath': y_new,
            'y_diff': 0,
            'angle_b': angle_b
            })
    if len(kpunt)>1:
        # calculate the angle between the first and second point
        angle_b = math.atan2(kpunt[1]['low_bath'] - kpunt[0]['low_bath'], 
                             kpunt[1]['meters'] - kpunt[0]['meters'])
        
        # change the second point to extend with an angle to llws
        x_new = kpunt[0]['meters'] + (-(kpunt[0]['low_bath']-bath.low_bath.iloc[idx_llws])/math.tan(angle_b))
        print(x_new, ' x_new wanneer 2 punten' )
        y_new = bath.low_bath.iloc[idx_llws]
        kpunt[1] = ({
            'id0': 999,
            'meters': x_new,
            'low_bath': y_new,
            'y_diff': 0,
            'angle_b': angle_b
            })

    # Find the index of the point with low_bath closest to llws
    kpunt.append({
        'id0': idx_llws,
        'meters': bath.meters.iloc[idx_llws],
        'low_bath': bath.low_bath.iloc[idx_llws],
        'y_diff': 0,
        'angle_b': np.inf, 
        'angle_a': np.inf, 
        'angle_diff': np.inf,
    })
    
    radius=2.5 # radius of search (maximum distance!)

### find the lowest chr. point on the bathymetry line
    kpunt_bot=[]
    # take cell 311 as example
    for x,y in zip(bath.meters ,bath.low_bath):
        if y == np.inf: # skip if point has no value
            kpunt_bot.append({
            'id0' : bath.meters[bath.meters==x].index[0],
            'meters': x,
            'low_bath': y,
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
            })
            continue
        try:
            dist = np.sqrt((x - bath.meters)**2 + (y - bath.low_bath)**2)
            id0=dist[dist==0].index[0]
            # find closest point before
            dist_before=dist[:id0]
            idb = (dist_before- radius).abs().idxmin()
            # find closest point after
            dist_after=dist[id0+1:]
            ida = (dist_after- radius).abs().idxmin()

            # determine angle before and after
            a_before= math.degrees(math.atan2(bath.low_bath.iloc[id0]-bath.low_bath.iloc[idb],
                                    bath.meters.iloc[id0]-bath.meters.iloc[idb]))
            a_after = math.degrees(math.atan2(bath.low_bath.iloc[ida]-bath.low_bath.iloc[id0],
                                    bath.meters.iloc[ida]-bath.meters.iloc[id0]))                             
            a_diff= abs(a_after-a_before)

            y_diff=abs(bath.low_bath.iloc[idb] - bath.low_bath.iloc[ida])
        except Exception as e:
            pass
        if y_diff==np.inf: # skip if difference is too large
            kpunt_bot.append({
                'id0' : bath.meters[bath.meters==x].index[0],
                'meters': x,
                'low_bath': y,
                'y_diff': 0,
                'angle_b': np.inf, 
                'angle_a': np.inf, 
                'angle_diff': np.inf,
                })
        else:
            kpunt_bot.append({
                'id0' : id0,
                'meters': x,
                'low_bath': y,
                'y_diff': y_diff,
                'angle_b': a_before, 
                'angle_a': a_after, 
                'angle_diff':a_diff,
                })

# laagste punt OF onderkant zvg laag`
# vergelijk diepste zvg laag met laagste punt
    #     1. als laagste punt dieper is dan laagste zvg laag, neem laagste zvg laag
    #     2. als laagste zvg laag dieper is dan laagste punt, neem laagste punt
    try:
        nearest_zvgl=(bath['low_bath'] - depth[-1]['depth_e_d']).abs().idxmin()
        if bath.low_bath.iloc[kpunt_bot[-1]['id0']] < bath.low_bath.iloc[nearest_zvgl]:
            kpunt.append({
            'id0': nearest_zvgl,
            'meters': bath.meters.iloc[nearest_zvgl],
            'low_bath': bath.low_bath.iloc[nearest_zvgl],
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
        })
        else:
            kpunt.append({
            'id0': kpunt_bot[-1]['id0'],
            'meters': bath.meters.iloc[kpunt_bot[-1]['id0']],
            'low_bath': bath.low_bath.iloc[kpunt_bot[-1]['id0']],
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
        })
    except:
        kpunt.append({
            'id0': kpunt_bot[-1]['id0'],
            'meters': bath.meters.iloc[kpunt_bot[-1]['id0']],
            'low_bath': bath.low_bath.iloc[kpunt_bot[-1]['id0']],
            'y_diff': 0,
            'angle_b': np.inf, 
            'angle_a': np.inf, 
            'angle_diff': np.inf,
        })
    
    
# dijkpunten (eerste 2; buitenkruin; teen dijk; insteek rivier)

    x_=[x['meters'] for x in kpunt]
    y_=[x['low_bath'] for x in kpunt]

    fig.gca().plot(x_,y_, "o-", markersize=10, color='#FF13F0', label='c_points')

##############Programmeren Eenvoudige toets#####################
    try:
        # bepalen schematisering huidige situatie
        H_geul = abs(y_[-1]-y_[-2]) #hoogte onderwatertalud
        X_geul = abs(x_[-1]-x_[-2]) #breedte onderwatertalud
        d_h_onder = abs(y_[2]-llws) # hoogte bovenkant voorland tot LLWS /OLW??
        alpha_r = math.atan2(H_geul, X_geul) # IN RADIANS

        h_dijk = abs(y_[0]-y_[1]) # hoogte dijk
        x_dijk = abs(x_[0]-x_[1]) # breedte dijk
        a_boven = math.atan2(h_dijk, x_dijk) # IN RADIANS
        b_voorland = abs(x_[1]-x_[-2]) # breedte voorland

        a_q_boven =math.atan2(2*h_dijk, b_voorland+2*h_dijk*(1/math.tan(a_boven))) # IN RADIANS
        # fictieve hoogte onderwatertalud
        
        H_r=H_geul+d_h_onder+2*h_dijk*((1/math.tan(alpha_r))/(1/math.tan(a_q_boven)))
        
        # bepalen signaleringsprofiel
        z_invloed=4*h_dijk # aangenomen 4*h_dijk als invloedzone
        M_invloed=2*H_geul # marge zv
        # first point = teen dijk
        xs_=[x_[1]]    
        ys_=[y_[1]]
        # second point = teen dijk + z_invloed
        xs_.append(x_[1]+z_invloed)           
        ys_.append(y_[1])
        # third point = teen dijk + z_invloed + M_invloed
        xs_.append(x_[1]+z_invloed+M_invloed)
        ys_.append(y_[1])
        # fourth point = teen dijk + z_invloed + M_invloed + s_talud
        a_talud=math.atan2(1,15) #1:15 talud
        l_talud=abs(y_[-1]-y_[1])/math.sin(a_talud) # lengte talud	
        xs_.append(xs_[2]+l_talud*math.cos(a_talud)) 
        ys_.append(ys_[-1]-abs(y_[-1]-y_[1]))

        fig.gca().plot(xs_,ys_, "o--", markersize=10, color='#0ee623', label='s_points')

    ##### E_1 
            # LET OP geen rekeningen gehouden met bestorting...
        # controleer op 1/3 H_hoogte of punt op werkerlijk profiel > punt op signaleringsprofiel anders Faal
        # vind X coordinaat van punt op signaleringsprofiel en werkelijk profiel
        #     xs en ys zoeken:
        ys_1deel3=y_[-1]+(1/3)*H_geul
        xs_1deel3=xs_[-1]-(1/3)*H_geul*math.tan(math.pi/2-a_talud)
        #     x en y zoeken:
        y_1deel3=y_[-1]+(1/3)*H_geul
        x_1deel3=x_[-1]-(1/3)*H_geul*math.tan(math.pi/2-alpha_r)

        if x_1deel3<xs_1deel3:
            E_1='Faal'
        else:
            E_1='Voldoet'

        fig.gca().plot(
            [x_1deel3, xs_1deel3],
            [y_1deel3, ys_1deel3],
            'o',
            markersize=12,
            color='#296cd9')

        # Plot text op figuur om te kenmerken + resultaat toets
        bbox = {'fc': 'white', 'pad':3}
        fig.gca().text((2/3)*max(bath.meters),7,f'Eenvoudige toets E_1 resultaat = {E_1}',color='red',fontsize=16,bbox=bbox)
        
        if save_plot==True:
            fig.savefig(csv_bathy.replace('.csv','_E_1.png'),bbox_inches='tight')
        else:
            pass
        plt.close()

    ##### E_2    
        # opnieuw standaard figuur inlezen
        depth, bath, fig = zvg_vs_bathy(dwp, csv_cpt, csv_bathy, rasterdate_file, min_ld=min_ld, save_plot=False)
        
        #% plot low_bath lijn
        fig.gca().plot(
            bath.meters,
            bath.low_bath, 
            color='black', 
            lw=3, 
            ls='--')

        # Helling is niet steiler dan 1:4 over 5m hoogte
        a_e_2_toets=math.atan2(1,4) # 1:4 talud
        depth_e_2_check=5 # 5m hoogte check E_2
        E_2_talud=[]
        # controleer niet verder dan het laagste punt
        x_min=min(bath.meters[bath.low_bath==min(bath.low_bath)])

        # from depth LLWS -2,5m start to check
        e_2_depth=y_[-2]-2.5    # y_[-2] is al gecheckt dat het goede startpunt is 
                                # geulinsteek (eerste punt check mogelijk)
        for x,y in zip(bath.meters ,bath.low_bath):
            if (y == np.inf) or (y>e_2_depth) or (x>x_min):
                E_2_talud.append({
                    'x_E_2': x,
                    'y_E_2': y,
                    'angle_5m': np.inf,
                    'check_angle': np.inf,
                    'check_E_2': np.inf
                    })
                continue
            else:
                id_y=bath[bath.low_bath==y].index[0]
                id_y_before=(bath.low_bath-(y+depth_e_2_check/2)).abs().idxmin()
                id_y_after=(bath.low_bath-(y-depth_e_2_check/2)).abs().idxmin()
                before_x=bath.meters.iloc[id_y_before]
                before_y=bath.low_bath.iloc[id_y_before]
                after_x=bath.meters.iloc[id_y_after]
                after_y=bath.low_bath.iloc[id_y_after]    
                angle= math.degrees(math.atan2(after_y-before_y,
                                    after_x-before_x))
            E_2_talud.append({
                'x_E_2': x,
                'y_E_2': y,
                'angle_5m': angle,
                'check_angle': angle+math.degrees(a_e_2_toets),
                'check_E_2': (angle+math.degrees(a_e_2_toets))<0
                })
                
        E_2_check=[x['check_E_2'] for x in E_2_talud]
        #### Plot punten met 1:4 talud gem. over 5m
        data_e_2_talud=pd.DataFrame(E_2_talud)
        # first filter True waardes
        data_e_2_talud_true=data_e_2_talud[data_e_2_talud.check_E_2==True]
        #hulpcolomn
        data_e_2_talud_true['hulp']=(data_e_2_talud_true['x_E_2'].diff() != 0.5).cumsum()
        # plot lijnen waar gemiddeld talud 1:4 is
        for gr_id, gr_df in data_e_2_talud_true.groupby(by='hulp'):
            fig.gca().plot(
                gr_df.x_E_2,
                gr_df.y_E_2, 
                color='red',
                lw=5 
                )

        if any([x['check_E_2'] for x in E_2_talud]): # als er tenminste één True voorkomst dan faal 
            E_2='Faal'
        else:
            E_2='Voldoet'

        # Plot text op figuur om te kenmerken + resultaat toets
        bbox = {'fc': 'white', 'pad':3}
        fig.gca().text((1/4)*max(bath.meters),7,f'Eenvoudige toets E_2 resultaat = {E_2}',color='red',fontsize=16,bbox=bbox)
        
        if save_plot==True:
            fig.savefig(csv_bathy.replace('.csv','_E_2.png'),bbox_inches='tight')
        else:
            pass
        plt.close()
    
    ##### E_3
        """
        Wanneer 1 van de volgende 2 criteria aan de hand is, is zettingsvloeiing mogelijk:
        1. H_r --> cot alpha_r <= (H_r/24)^(1/3)
        2. bresvloeiing (kan niet controleren met d50 / d15) wel op vookomen helling
        zie tabel: 
        """
        # opnieuw standaard figuur inlezen
        depth, bath, fig = zvg_vs_bathy(dwp, csv_cpt, csv_bathy, rasterdate_file, min_ld=0.25, save_plot=False)

        #% plot low_bath lijn
        fig.gca().plot(
            bath.meters,
            bath.low_bath, 
            color='black', 
            lw=3, 
            ls='--')

        check_e_3_1=(1/math.tan(alpha_r))<=7*(H_r/24)**(1/3)

        check_e_3_2_tabel={
            -5: -math.atan2(1,2),
            -10: -math.atan2(1,2.5),
            -15: -math.atan2(1,3),
            -20: -math.atan2(1,3.5),
            -25: -math.atan2(1,4),
            -30: -math.atan2(1,4.7),
            -35: -math.atan2(1,5.4),
            -40: -math.atan2(1,6),
        }

        E_3_talud=[]

        for i, x in enumerate(zip(bath.meters ,bath.low_bath)):
            if (x[1] == np.inf) or (x[1]>y_[-2]) or (x[0]>x_min): # skip if point has no value or above water
                E_3_talud.append({
                    'id0' : i,
                    'meters': x[0],
                    'low_bath': x[1],
                    'chk_bath': np.inf,
                    'angle_p': np.inf,
                    'angle_c': np.inf,
                    'chk_pc_a': np.inf,
                    })
            if x[1]<y_[-2] and (x[0]<=x_min):
                # hoek uitrekeningen (tangent -1 en +1)
                alpha_e3=math.atan2(bath.low_bath.iloc[i+1]-bath.low_bath.iloc[i-1],
                                        bath.meters.iloc[i+1]-bath.meters.iloc[i-1])
                
                if (alpha_e3==np.inf) or (alpha_e3==math.pi/2) or (alpha_e3==math.pi): # skip if difference is too large
                            E_3_talud.append({
                                'id0' : i,
                                'meters': x[0],
                                'low_bath': x[1],
                                'chk_bath': np.inf,
                                'angle_p': np.inf,
                                'angle_c': np.inf,
                                'chk_pc_a': np.inf,
                                })
                else:
                    # which value to check against
                    for d,chk in check_e_3_2_tabel.items():
                        if d+5>x[1]-y_[-2]>=d: # because the chk values start counting from the geulinsteek   
                            E_3_talud.append({
                                'id0' : i,
                                'meters': x[0],
                                'low_bath': x[1],
                                'chk_bath': x[1]-y_[-2],
                                'angle_p': math.degrees(alpha_e3),
                                'angle_c': math.degrees(chk),
                                'chk_pc_a': math.degrees(alpha_e3)<math.degrees(chk)
                                })

            data_e_3_talud=pd.DataFrame(E_3_talud)
            # first filter True waardes
            data_e_3_talud_true=data_e_3_talud[data_e_3_talud.chk_pc_a==True]
            #hulpcolomn
            data_e_3_talud_true['hulp']=(data_e_3_talud_true['meters'].diff() != 0.5).cumsum()
            # plot lijnen waar gemiddeld talud 1:4 is
            for gr_id, gr_df in data_e_3_talud_true.groupby(by='hulp'):
                fig.gca().plot(
                    gr_df.meters,
                    gr_df.low_bath, 
                    color='red',
                    lw=5 
                    )

        if check_e_3_1==False:
            E_3_1='Faal'
            if not(any([x['chk_pc_a'] for x in E_3_talud])): # als er tenminste één True voorkomst dan faal 
                E_3_2='Voldoet'
            else:
                E_3_2='Faal'
        elif check_e_3_1==True:
            E_3_1='Voldoet'
            E_3_2='n.v.t.'
        
        # Plot text op figuur om te kenmerken + resultaat toets
        bbox = {'fc': 'white', 'pad':3}
        fig.gca().text((1/4)*max(bath.meters),7,f'Eenvoudige toets E_3_2 resultaat = {E_3_2}',color='red',fontsize=16,bbox=bbox)
        
        if save_plot==True:
            fig.savefig(csv_bathy.replace('.csv','_E_3.png'),bbox_inches='tight')
        else:
            pass
        plt.close()
    except:
        E_1='n.v.t.'
        E_2='n.v.t.'
        E_3_1='n.v.t.'
        E_3_2='n.v.t.'
        xs_1deel3=np.nan
        x_1deel3=np.nan
        
        if save_plot==True:
            fig.savefig(csv_bathy.replace('.csv','_error.png'),bbox_inches='tight')
        else:
            pass
        plt.close()
       
    toets.append({
        'P_1deel3_x': x_1deel3,
        'S_1deel3_y': xs_1deel3,
        'Toets_E_1': E_1,
        'Toets_E_2': E_2,
        'Toets_E_3_1': E_3_1,
        'Toets_E_3_2': E_3_2,
    })

    return toets