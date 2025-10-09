from dash import (Dash, html, dcc, Input, Output,
                  State, dash_table, callback, callback_context)
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from typing import cast, Optional, Tuple

# ------------------ Config ------------------
MAP_HEIGHT = 600
PROFILE_HEIGHT = 420
DEFAULT_ZOOM = 15   # 16–19 is meestal prettig

boezem_side = 'links'

app = Dash(external_stylesheets=[dbc.themes.CERULEAN])


# ------------------ Helpers ------------------
def ensure_unique(
        df: pd.DataFrame,
        boezem: Optional[bool] = False
        ) -> pd.DataFrame:
    """Zorg dat er een kolom 'unique_' is (CODE + '_' + NAAM).
       Normaliseert CODE-veld (CODEdijkr / CODE1 / CODE_1 -> CODE)."""

    if 'CODE' not in df.columns:
        if 'CODEdijkr' in df.columns:
            df = df.rename(columns={'CODEdijkr': 'CODE'})
        elif 'CODE1' in df.columns:
            df = df.rename(columns={'CODE1': 'CODE'})
        elif 'CODE_1' in df.columns:
            df = df.rename(columns={'CODE_1': 'CODE'})
        else:
            df['CODE'] = ""

    if boezem:
        df["SORT_CODE"] = (
            df["CODE"].astype(str)
            + df["NAAM"].astype(str).str.split("_").str[-1].str.split("-")
            .str[:-1].str.join("-")
        )
    else:
        df['SORT_CODE'] = df['CODE']

    if 'NAAM' not in df.columns:
        raise KeyError("Kolom 'NAAM' ontbreekt voor 'unique_' constructie.")

    if 'unique_' not in df.columns:
        code = df['CODE'].astype(str).str.strip()
        naam = df['NAAM'].astype(str).str.strip()
        naam = naam.astype(str).str.replace(',', 'p')
        df['unique_'] = code + '_' + naam

    return df


def lines_to_latlon(geom: LineString | MultiLineString) -> list:
    out = []
    if isinstance(geom, LineString):
        x, y = geom.xy; out.append((list(y), list(x)))
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            x, y = g.xy; out.append((list(y), list(x)))
    return out


# ------------------ Data: punten combineren ------------------
# Primaire kering (eerste punten set)
gdf = gpd.read_file(
    './02_Intermediate_shapefiles/Hecto_Dijkring_Join.shp'
    ).to_crs(epsg=4326)
gdf = ensure_unique(gdf)
gdf['lon'] = gdf.geometry.x
gdf['lat'] = gdf.geometry.y
gdf['hover'] = gdf['unique_'].astype(str)
gdf['source'] = 'primair'

# Boezem kering (tweede punten set)
gdf2 = gpd.read_file(
    './02_Intermediate_shapefiles/WSHD_Boezem_Join.shp'
    ).to_crs(epsg=4326)
gdf2 = ensure_unique(gdf2, boezem=True)
gdf2['lon'] = gdf2.geometry.x
gdf2['lat'] = gdf2.geometry.y
gdf2['hover'] = gdf2['unique_'].astype(str)
gdf2['source'] = 'boezem'

# Combineer -> één set punten
pts = pd.concat([gdf, gdf2], ignore_index=True)

# Natuurlijke sortering op unique_,
# en bij dubbele unique_ de 'primair' prefereren
pts['source_order'] = pts['source'].map({'primair': 0, 'boezem': 1}).fillna(2)


# Sorteer punten op hectometering
pts_sorted = (
    pts.sort_values(
        by=['source_order', 'SORT_CODE', 'NAAM'],
        key=lambda col: (
            pd.to_numeric(
                col.str.replace("-", "_")
                   .str.split('_').str[-1]         # pak laatste deel
                   .str.replace('p', '.')          # 12p5 -> 12.5
                   .str.replace(',', '.'),         # 33,4 -> 33.4
                errors='coerce'                     # niet-parsebaar -> NaN
            )
            if col.name == 'NAAM' else col
        )
    )
    .drop_duplicates('unique_', keep='first').reset_index(drop=True)
)


N = len(pts_sorted)
unique_to_idx = {u: i for i, u in enumerate(pts_sorted['unique_'])}

# ------------------ Data: lijnen ------------------
line_prim = gpd.read_file(
    './02_Intermediate_shapefiles/dwp_2025_prim_kering_hecto.shp'
    ).to_crs(epsg=4326)
line_prim = ensure_unique(line_prim)

line_boez_links = gpd.read_file(
    './02_Intermediate_shapefiles/dwp_2025_boezem_links_hecto.shp'
    ).to_crs(epsg=4326)
line_boez_links = ensure_unique(line_boez_links)

line_boez_rechts = gpd.read_file(
    './02_Intermediate_shapefiles/dwp_2025_boezem_rechts_hecto.shp'
    ).to_crs(epsg=4326)
line_boez_rechts = ensure_unique(line_boez_rechts)


# ------------------ Figure maker ------------------
def make_map_with_highlight(
        idx: int,
        zoom: int,
        show_basis: bool,
        show_boezem: bool,
        show_prim: bool,
        show_boez_lines: bool
        ) -> go.Figure:
    """Kaart met één puntenlaag (gekleurd op 'source')
       en alleen de lijnen bij de selectie.
       Zichtbaarheid wordt gestuurd via booleans."""
    idx = int(idx) % N
    row = pts_sorted.iloc[idx]

    # Filter punten op checkbox-visibility
    points_df = pts_sorted.copy()
    if not show_basis:
        points_df = points_df[points_df['source'] != 'primair']
    if not show_boezem:
        points_df = points_df[points_df['source'] != 'boezem']

    # Maak figuur met de (eventueel gefilterde) punten
    fig: go.Figure
    fig = px.scatter_map(
        points_df,
        lat='lat', lon='lon',
        hover_name='hover',
        color='source',     # aparte trace per bron die nog zichtbaar is
        zoom=zoom,
        center={'lat': row.lat, 'lon': row.lon},
        height=MAP_HEIGHT
    )

    # Update the makers met de juiste kleuren.
    trace: go.Scatter
    scatter_traces = cast(tuple[go.Scatter, ...], fig.data)
    for trace in scatter_traces:
        if trace['name'] == 'primair':
            trace['marker'] = marker=dict(size=12, color="#2563eb")

        if trace['name'] == 'boezem':
            trace['marker'] = marker=dict(size=10, color="#5c8cf3")

    fig.data = scatter_traces

    fig.update_layout(
        map=dict(style="open-street-map",
                 center=dict(lat=row.lat, lon=row.lon), zoom=zoom),
        margin=dict(r=0, t=0, l=0, b=0),
        clickmode='event+select',
        hovermode='closest',
        showlegend=True,
        height=MAP_HEIGHT,
        uirevision=None
    )

    # Duidelijke extra highlight-marker op de selectie (altijd tonen)
    fig.add_trace(go.Scattermap(
        lat=[row.lat], lon=[row.lon],
        mode="markers",
        name="Geselecteerd",
        marker=dict(size=20, symbol="circle", color="#ef4444"),
        text=[row['hover']],
        hoverinfo="text"
    ))

    # Lijnen bij de selectie, afhankelijk van toggles
    drew_any = False
    if show_prim:
        sel_prim = line_prim[line_prim['unique_'] == row['unique_']]
        if not sel_prim.empty:
            for geom in sel_prim.geometry:
                for lat_list, lon_list in lines_to_latlon(geom):
                    fig.add_trace(go.Scattermap(
                        lat=lat_list, lon=lon_list,
                        mode="lines",
                        name="Lijn (primair)",
                        line=dict(width=3, color="#111827")
                    ))
                    drew_any = True
    if show_boez_lines:
        if boezem_side == 'links':
            sel_boez = line_boez_links[
                line_boez_links['unique_'] == row['unique_']
                ]
        else:
            sel_boez = line_boez_rechts[
                line_boez_rechts['unique_'] == row['unique_']
                ]
        if not sel_boez.empty:
            for geom in sel_boez.geometry:
                for lat_list, lon_list in lines_to_latlon(geom):
                    fig.add_trace(go.Scattermap(
                        lat=lat_list, lon=lon_list,
                        mode="lines",
                        name="Lijn (boezem)",
                        line=dict(width=3, color="#111827")
                    ))
                    drew_any = True

    if not drew_any and (show_prim or show_boez_lines):
        fig.update_layout(
            title_text=f"Geen lijnen gevonden voor unique_ = {row['unique_']}"
            )

    return fig


# ------------------ Layout ------------------
app.layout = html.Div(
    [
        dcc.Store(id='idx-store', data=0),  # actieve index

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            'Hectometrering', style={'textAlign': 'center'}
                            ),

                        # Layer toggles
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Checklist(
                                        id='layer-toggle',
                                        options=[
                                            {'label': 'Punten (primair)',
                                             'value': 'primair'},
                                            {'label': 'Punten (boezem)',
                                             'value': 'boezem'},
                                            {'label': 'Lijn (primair)',
                                             'value': 'line_prim'},
                                            {'label': 'Lijn (boezem)',
                                             'value': 'line_boez'},
                                        ],
                                        value=['primair', 'boezem',
                                               'line_prim', 'line_boez'],
                                        inline=True,
                                        inputStyle={'margin-right': '6px'},
                                        labelStyle={'margin-right': '16px'}
                                    ),
                                    width=12
                                )
                            ],
                            className="mb-2"
                        ),

                        dcc.Graph(
                            id='map',
                            style={'height': f'{MAP_HEIGHT}px'}
                        ),
                    ],
                    width=9
                ),
                dbc.Col(
                    [
                        dbc.Button('Copy', id='copy-button', color="primary",
                                   className='me-1'),
                        dash_table.DataTable(
                            id='hover-table',
                            page_size=500,
                            virtualization=True,
                            style_table={'height': f'{MAP_HEIGHT}px',
                                         'overflowY': 'auto'},
                        ),
                    ],
                    width=3
                ),
            ],
            className="g-2"
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            'AHN Dwarsprofiel', style={'textAlign': 'center'}
                            ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "◀ Vorige", id="prev-btn",
                                    color="secondary", className="me-2"
                                    ),
                                dbc.Button(
                                    "Volgende ▶", id="next-btn",
                                    color="secondary"
                                    ),
                                dbc.Button(
                                    "Boezem Links", id="switch-btn",
                                    color="primary"
                                    ),
                            ],
                            className="mb-2",
                        ),
                        dcc.Graph(id='profile-fig',
                                  style={'height': f'{PROFILE_HEIGHT}px'}),
                        dcc.RangeSlider(-200, 150, 10, value=[-200, 150],
                                        id="profile-rangeslider",
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True
                                            },
                                        allowCross=False,
                                        )
                    ],
                    width=12
                )
            ]
        ),
    ],
    style={'padding': '12px'}
)


# ------------------ Callbacks ------------------
@callback(
    Output('idx-store', 'data'),
    Output('map', 'figure'),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    Input('map', 'clickData'),
    Input('layer-toggle', 'value'),   # <— toggles
    State('idx-store', 'data'),
    prevent_initial_call=False
)
def navigate_points(prev_clicks, next_clicks, clickData, layers, idx):
    """Werk actieve index bij o.b.v. knoppen/klik
    en herteken kaart o.b.v. toggles."""
    # visibility flags
    show_basis = 'primair' in layers if layers is not None else True
    show_boezem = 'boezem' in layers if layers is not None else True
    show_prim = 'line_prim' in layers if layers is not None else True
    show_boez_lines = 'line_boez' in layers if layers is not None else True

    ctx = callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'map' and clickData:
            p = clickData['points'][0]
            htext = p.get('hovertext') or p.get('text')
            if htext and htext in unique_to_idx:
                idx = unique_to_idx[htext]
        elif trigger == 'prev-btn':
            idx = (int(idx or 0) - 1) % N
        elif trigger == 'next-btn':
            idx = (int(idx or 0) + 1) % N
        else:
            # toggles veranderd: index behouden
            idx = int(idx or 0) % N
    else:
        idx = int(idx or 0) % N  # initial draw

    fig = make_map_with_highlight(idx, DEFAULT_ZOOM, show_basis,
                                  show_boezem, show_prim, show_boez_lines)
    return idx, fig


@callback(
    Output('hover-table', 'columns'),
    Output('hover-table', 'data'),
    Output('profile-fig', 'figure'),
    Input('idx-store', 'data'),
    Input('profile-rangeslider', 'value'),
    prevent_initial_call=False
)
def update_detail(idx, rangevalue: Tuple[int, int]):
    """Update tabel en profiel o.b.v. de actieve index (pts_sorted)."""
    import os
    import glob
    import pandas as pd
    import plotly.express as px

    global boezem_side  # gebruikt om uit meerdere matches te kiezen

    # Zorg voor veilige defaults
    table_cols = [{'name': c, 'id': c} for c in ['x', 'z']]
    table_data = []
    empty_fig = px.scatter(title="Geen data beschikbaar")
    empty_fig.update_layout(margin=dict(r=0, t=40, l=0, b=0),
                            height=PROFILE_HEIGHT, autosize=False)

    try:
        idx = int(idx or 0) % N
    except Exception:
        return table_cols, table_data, empty_fig

    dfs_list = []
    file_names = []

    # helper: lees csv met fallback op ; en komma-decimaal
    # Zoekt ook naar waarden uit de lodingen
    def read_bathy_csv(path):

        # Probeer standaard, daarna Europees CSV-formaat
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=';', decimal=',')

        if 'xRD' not in df.columns:
            raise ValueError(
                f"Kolom 'xRD' niet gevonden in {os.path.basename(path)}"
                )

        # Alle kolommen strikt vóór xRD (xRD uitgesloten)
        idx_xrd = df.columns.get_loc('xRD')
        hoogtekolommen = list(df.columns[1:idx_xrd])

        # Verwachtte bronkolommen
        if 'meters' not in df.columns:
            raise ValueError(
                f"Kolom 'meters' niet gevonden in {os.path.basename(path)}"
                )
        if 'ahn' not in df.columns:
            raise ValueError(
                f"Kolom 'ahn' niet gevonden in {os.path.basename(path)}"
                )

        # Maak numeriek en vervang ±inf door NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # x uit meters
        df['x'] = pd.to_numeric(df['meters'], errors='coerce')

        # z primair uit ahn
        df['z'] = pd.to_numeric(df['ahn'], errors='coerce')

        # Fallback: eerste geldige waarde uit kolommen vóór xRD,
        # exclusief meters/ahn
        fallback_kolommen = [c for c in hoogtekolommen
                             if c not in ('meters', 'ahn')]
        if fallback_kolommen:
            # Zet fallback-kolommen numeriek
            df[fallback_kolommen] = df[fallback_kolommen].apply(
                pd.to_numeric, errors='coerce'
            )
            mask = df['z'].isna()
            if mask.any():
                df.loc[mask, 'z'] = (
                    df.loc[mask, fallback_kolommen]
                    .bfill(axis=1)
                    .iloc[:, 0]
                )

        # Geef alleen x en z terug
        return df[['x', 'z']]

    # verzamel rijen en dataframes voor idx-1, idx, idx+1
    for offset in [-1, 0, 1]:
        j = idx + offset
        if j < 0 or j >= len(pts_sorted):
            continue  # buiten bereik: overslaan

        row = pts_sorted.iloc[j]
        file_name_comp = f"{row.CODE}_{str(row.NAAM).replace(',', 'p')}"
        matches = sorted(glob.glob(os.path.join(
            './03_Bathymetry', f"dp_cpt_{file_name_comp}_*.csv"))
                         )

        if not matches:
            # Als er helemaal niets is voor de centrale index, toon melding
            if offset == 0:
                msg_fig = px.scatter(
                    title=f"Geen CSV gevonden voor: {file_name_comp}"
                    )
                msg_fig.update_layout(
                    margin=dict(r=0, t=40, l=0, b=0), height=PROFILE_HEIGHT,
                    autosize=False
                    )
                return table_cols, table_data, msg_fig
            # voor buurprofielen gewoon overslaan
            continue

        # Kies CSV op basis van boezemzijde
        # (gecontroleerd, niet blind op index 1)
        pick = 0
        if boezem_side == 'rechts' and len(matches) > 1:
            pick = 1
        csv_path = matches[pick]

        try:
            df = read_bathy_csv(csv_path)
        except Exception as e:
            if offset == 0:
                bad = px.scatter(
                    title="Probleem met CSV: "
                    + f"{os.path.basename(csv_path)} — {e}"
                    )
                bad.update_layout(margin=dict(r=0, t=40, l=0, b=0),
                                  height=PROFILE_HEIGHT, autosize=False)
                return table_cols, table_data, bad
            else:
                continue

        # markeer profielbron: bestandsnaam + offset
        label = f"{file_name_comp}"
        df = df[['x', 'z']].copy()
        df['Profiel'] = label
        dfs_list.append(df)
        file_names.append(file_name_comp)

        # Voor de centrale rij vul de tabel
        if offset == 0:
            if "Primair" in csv_path:
                table_data = df[['x', 'z']].loc[
                    rangevalue[0]*2 + 400:
                    rangevalue[1]*2+401].to_dict('records')
            else:
                table_data = df[['x', 'z']].loc[
                    rangevalue[0]*2 + 300:
                    rangevalue[1]*2+300].to_dict('records')

    if not dfs_list:
        return table_cols, table_data, empty_fig

    dfs = pd.concat(dfs_list, ignore_index=True)

    prof = px.line(
        dfs, x='x', y='z',
        color='Profiel',  # aparte lijn per profiel
        markers=True,
        labels={'x': 'Afstand (m)', 'z': 'AHN (m)', 'Profiel': 'Profiel'}
    )

    prof.update_layout(
        xaxis_title="Afstand (m)",
        yaxis_title="AHN (m)",
        xaxis=dict(range=[rangevalue[0], rangevalue[1]]),
        margin=dict(r=0, t=40, l=0, b=0),
        height=PROFILE_HEIGHT,
        autosize=False,
        legend_title_text="Profielen",
    )
    prof.update_traces(hovertemplate="x=%{x}<br>z=%{y}<extra></extra>")

    return table_cols, table_data, prof


@callback(
    Output('copy-button', 'color'),
    Input('copy-button', 'n_clicks'),
    State('hover-table', 'data'),
    prevent_initial_call=True
)
def copy_table(n_clicks, data):
    if not n_clicks or not data:
        return 'primary'
    t_data = pd.DataFrame.from_dict(data)
    if {'x', 'z'}.issubset(t_data.columns):
        t_data[['x', 'z']].to_clipboard(index=False)
        return 'secondary'
    return 'primary'


@callback(
    Output("switch-btn", 'children'),
    Input('switch-btn', 'n_clicks'),
    Input('switch-btn', 'children'),
    prevent_initial_call=True,
)
def boezem_switch(n_clicks, children):
    global boezem_side
    ctx = callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'switch-btn' and children == 'Boezem Links':
            boezem_side = 'rechts'
            return 'Boezem Rechts'
        elif trigger == 'switch-btn' and children == 'Boezem Rechts':
            boezem_side = 'links'
            return 'Boezem Links'


# ------------------ Main ------------------
if __name__ == '__main__':
    app.run(debug=True)
