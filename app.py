"""ReclaimSat - satellite monitoring of mine waste reclamation.
Case study: SSGPO (Sokolov-Sarbai, ERG), Rudny, Kazakhstan.
Data: Landsat 5/7/8/9 summer NDVI 1990-2026 and Sentinel-2 NDVI 2026, processed in Google Earth Engine.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

DATA = Path(__file__).parent / "data"
REF = "reference_steppe"

# Indicative thresholds for the Recovery Index (RI = NDVI dump / NDVI natural steppe)
RECOVERED = 0.85
RECOVERING = 0.60

st.set_page_config(page_title="ReclaimSat", layout="wide")


# ---------------- Data ----------------
@st.cache_data
def load_series():
    df = pd.read_csv(DATA / "ndvi_timeseries.csv")
    wide = df.pivot(index="year", columns="name", values="mean").sort_index()
    ref = wide[REF]
    dumps = [c for c in wide.columns if c != REF]
    ri = wide[dumps].div(ref, axis=0)
    return wide, ri, dumps


def slope_per_decade(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 3:
        return np.nan
    return float(np.polyfit(s.index.values, s.values, 1)[0] * 10)


def status(ri_now: float) -> str:
    if ri_now >= RECOVERED:
        return "Recovered"
    if ri_now >= RECOVERING:
        return "Recovering"
    return "Not recovering"


wide, ri, dumps = load_series()
labels_path = DATA / "dump_labels.json"
labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
label = lambda d: labels.get(d, d)

first_years = ri.index[ri.index <= 2000]
last_years = ri.index[ri.index >= 2022]

rows = []
for d in dumps:
    ri_early = ri.loc[first_years, d].mean()
    ri_now = ri.loc[last_years, d].mean()
    rows.append({
        "Object": label(d),
        "id": d,
        "NDVI 2022-26": wide.loc[last_years, d].mean(),
        "RI 1990-2000": ri_early,
        "RI 2022-26": ri_now,
        "RI trend 2014-26 /decade": slope_per_decade(ri.loc[ri.index >= 2014, d]),
        "Status": status(ri_now),
    })
summary = pd.DataFrame(rows)
# Priority: lowest current RI and weakest recent trend first
summary["Priority score"] = (1 - summary["RI 2022-26"]) - summary["RI trend 2014-26 /decade"].fillna(0)
summary = summary.sort_values("Priority score", ascending=False).reset_index(drop=True)
summary.insert(0, "Priority", range(1, len(summary) + 1))

# ---------------- Header ----------------
st.title("ReclaimSat")
st.markdown(
    "**Satellite monitoring of mine waste reclamation** - SSGPO iron ore mines (ERG), Rudny, Kazakhstan. "
    "37 years of Landsat data (1990-2026) compared against natural steppe."
)

worst = summary.iloc[0]
best = summary.sort_values("RI 2022-26", ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Objects monitored", len(dumps))
c2.metric("Summers with data (1990-2026)", len(wide))
c3.metric(f"Lowest RI: {worst['Object']}", f"{worst['RI 2022-26']:.2f}",
          f"{worst['RI 2022-26'] - worst['RI 1990-2000']:+.2f} since 1990s")
c4.metric(f"Highest RI: {best['Object']}", f"{best['RI 2022-26']:.2f}",
          f"{best['RI 2022-26'] - best['RI 1990-2000']:+.2f} since 1990s")

st.caption(
    "Recovery Index (RI) = summer NDVI of a dump / NDVI of natural steppe in the same year. "
    "Dividing by the steppe removes the effect of wet and dry years. RI = 1 means vegetation like the steppe."
)

# ---------------- Ranking ----------------
st.subheader("Reclamation priority")
fmt = {"NDVI 2022-26": "{:.2f}", "RI 1990-2000": "{:.2f}", "RI 2022-26": "{:.2f}",
       "RI trend 2014-26 /decade": "{:+.3f}", "Priority score": "{:.2f}"}


def color_status(v):
    return {"Recovered": "background-color:#d9f2d9; color:#14532d; font-weight:600",
            "Recovering": "background-color:#fff3c4; color:#713f12; font-weight:600",
            "Not recovering": "background-color:#f8d0cc; color:#7f1d1d; font-weight:600"}.get(v, "")


st.dataframe(
    summary.drop(columns=["id"]).style.format(fmt).map(color_status, subset=["Status"]),
    hide_index=True, width="stretch",
)
st.caption(f"Status thresholds (indicative): Recovered RI >= {RECOVERED}, Recovering {RECOVERING}-{RECOVERED}, "
           f"Not recovering < {RECOVERING}.")

# ---------------- Year selection (chart click -> map) ----------------
YEARS_DIR = DATA / "years"
years_meta = json.loads((YEARS_DIR / "years.json").read_text()) if (YEARS_DIR / "years.json").exists() else None
year_list = sorted(int(y) for y in years_meta["years"]) if years_meta else []
LAYER_S2 = "Sentinel-2 NDVI 2026 (10 m)"
LAYER_TREND = "NDVI trend 1990-2026 (Landsat 30 m)"
LAYER_YEAR = "By year, relative to steppe (Landsat 30 m)"
if year_list and "map_year" not in st.session_state:
    st.session_state["map_year"] = year_list[-1]


def take_click(event, key):
    """Apply a click on a chart point once: switch the map to that year."""
    pts = getattr(getattr(event, "selection", None), "points", None) or []
    sig = tuple(sorted({int(round(p["x"])) for p in pts if "x" in p}))
    if sig and st.session_state.get(f"_last_{key}") != sig and year_list:
        y = min(year_list, key=lambda v: abs(v - sig[0]))
        st.session_state["map_year"] = y
        st.session_state["layer"] = LAYER_YEAR
    st.session_state[f"_last_{key}"] = sig


sel_year = st.session_state.get("map_year") if st.session_state.get("layer") == LAYER_YEAR else None

# ---------------- Charts ----------------
if year_list:
    st.caption("Tip: click a point on a chart to see that year on the map below.")
palette = ["#2563eb", "#dc2626", "#f59e0b", "#16a34a", "#7c3aed", "#0891b2", "#be185d"]
left, right = st.columns(2)

with left:
    st.subheader("Summer NDVI")
    fig = go.Figure()
    for i, d in enumerate(dumps):
        fig.add_trace(go.Scatter(x=wide.index, y=wide[d], name=label(d), mode="lines+markers",
                                 line=dict(color=palette[i % len(palette)], width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=wide.index, y=wide[REF], name="Natural steppe", mode="lines+markers",
                             line=dict(color="#6b7280", width=2.5, dash="dash"), marker=dict(size=4)))
    fig.update_layout(height=420, yaxis_title="NDVI", xaxis_title="Year", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.2))
    if sel_year:
        fig.add_vline(x=sel_year, line_width=1, line_dash="dot", line_color="#333333")
    ev1 = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="ndvi_chart")
    take_click(ev1, "ndvi_chart")

with right:
    st.subheader("Recovery Index (3-year rolling mean)")
    fig2 = go.Figure()
    fig2.add_hrect(y0=RECOVERED, y1=1.1, fillcolor="#d9f2d9", opacity=0.5, line_width=0)
    fig2.add_hrect(y0=RECOVERING, y1=RECOVERED, fillcolor="#fff3c4", opacity=0.5, line_width=0)
    fig2.add_hrect(y0=0, y1=RECOVERING, fillcolor="#f8d0cc", opacity=0.4, line_width=0)
    for i, d in enumerate(dumps):
        roll = ri[d].rolling(3, min_periods=2, center=True).mean()
        fig2.add_trace(go.Scatter(x=ri.index, y=roll, name=label(d), mode="lines",
                                  line=dict(color=palette[i % len(palette)], width=2.5)))
        fig2.add_trace(go.Scatter(x=ri.index, y=ri[d], mode="markers", showlegend=False,
                                  marker=dict(color=palette[i % len(palette)], size=4, opacity=0.4)))
    fig2.update_layout(height=420, yaxis_title="RI (dump / steppe)", xaxis_title="Year",
                       yaxis_range=[0, 1.1], margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=-0.2))
    if sel_year:
        fig2.add_vline(x=sel_year, line_width=1, line_dash="dot", line_color="#333333")
    ev2 = st.plotly_chart(fig2, width="stretch", on_select="rerun", selection_mode="points", key="ri_chart")
    take_click(ev2, "ri_chart")

# ---------------- Map ----------------
st.subheader("Map")
meta = json.loads((DATA / "rasters.json").read_text())
layer_opts = [LAYER_S2, LAYER_TREND] + ([LAYER_YEAR] if year_list else [])
layer = st.radio("Layer", layer_opts, horizontal=True, key="layer")
if layer == LAYER_YEAR:
    year = st.select_slider("Year", options=year_list, key="map_year")
    ym = years_meta["years"][str(year)]

m = folium.Map(location=[53.0227, 63.1417], zoom_start=12, tiles=None, control_scale=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite",
).add_to(m)
if layer == LAYER_YEAR:
    folium.raster_layers.ImageOverlay(str(YEARS_DIR / f"ri_{year}.png"), bounds=years_meta["bounds"],
                                      opacity=0.8, name=f"Relative NDVI {year}").add_to(m)
    legend = (f"{year}: vegetation relative to natural steppe that summer (brown 0 = bare, teal 1 = like the steppe). "
              f"Image coverage {ym['coverage_pct']}%"
              + ("; gaps or stripes are missing Landsat data." if ym["coverage_pct"] < 95 or 2003 <= year <= 2012 else "."))
elif layer == LAYER_S2:
    folium.raster_layers.ImageOverlay(str(DATA / "s2_ndvi_2026.png"), bounds=meta["s2"]["bounds"],
                                      opacity=0.75, name="NDVI 2026").add_to(m)
    legend = "NDVI 0 (bare, brown) to 0.6 (green vegetation, teal)"
else:
    folium.raster_layers.ImageOverlay(str(DATA / "ndvi_trend.png"), bounds=meta["trend"]["bounds"],
                                      opacity=0.75, name="NDVI trend").add_to(m)
    legend = "NDVI change per year: red = losing vegetation (-0.01/yr), green = gaining (+0.01/yr)"

dumps_geo = DATA / "dumps.geojson"
if dumps_geo.exists():
    gj = json.loads(dumps_geo.read_text())
    ri_now = dict(zip(summary["id"], summary["RI 2022-26"]))
    stat = dict(zip(summary["id"], summary["Status"]))
    col = {"Recovered": "#16a34a", "Recovering": "#f59e0b", "Not recovering": "#dc2626"}
    for f in gj["features"]:
        n = f["properties"].get("name")
        f["properties"]["label"] = label(n)
        f["properties"]["RI"] = f"{ri_now.get(n, float('nan')):.2f}"
        f["properties"]["status"] = stat.get(n, "")
    folium.GeoJson(
        gj, name="Dumps",
        style_function=lambda f: {"color": col.get(f["properties"]["status"], "yellow"),
                                  "weight": 3, "fillOpacity": 0.05},
        tooltip=folium.GeoJsonTooltip(fields=["label", "RI", "status"], aliases=["Object", "RI 2022-26", "Status"]),
    ).add_to(m)
folium.LayerControl().add_to(m)
map_html = m.get_root().render()
if hasattr(st, "iframe"):
    st.iframe(map_html, height=540)
else:  # older Streamlit
    components.html(map_html, height=540)
st.caption(legend)

# ---------------- Method ----------------
with st.expander("Method and limitations"):
    st.markdown(f"""
**Data.** Landsat 5, 7, 8, 9 Collection 2 Level-2 surface reflectance, June-July, cloud/shadow/snow masked,
annual median NDVI. Landsat 5/7 harmonized to Landsat 8 (Roy et al., 2016). Sentinel-2 L2A for the 2026 map.
Processing in Google Earth Engine.

**Reference steppe.** Pixels 8-18 km from the mine, classed as grassland in ESA WorldCover 2021,
more than 1.5 km from the dumps, with low year-to-year NDVI variability in 2014-2026 (to exclude cropland and fallow).
500 random pixels.

**Limitations.**
- Satellite NDVI shows *how green* a dump is, not *why*: natural revegetation and company reclamation look the same.
- Grassland in 2021 does not prove the land was never ploughed (Virgin Lands campaign, 1950s).
- 30 m Landsat pixels mix dump tops with slopes and roads at polygon edges.
- No Landsat summer data for {", ".join(str(y) for y in sorted(set(range(1990, 2027)) - set(wide.index)))}.
- Status thresholds are indicative, not regulatory.
""")
