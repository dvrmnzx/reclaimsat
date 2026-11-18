"""ReclaimSat - satellite monitoring of mine waste reclamation.
Case study: SSGPO (Sokolov-Sarbai, ERG), Rudny, Kazakhstan.
Data: Landsat 5/7/8/9 summer NDVI 1990-2026 and Sentinel-2 NDVI 2026, processed in Google Earth Engine.

Design: the palette is the NDVI colour ramp itself (bare ground brown -> steppe teal),
so the same colours mean the same thing in the text, the recovery strip and the map.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
REF = "reference_steppe"

# Indicative thresholds for the Recovery Index (RI = NDVI dump / NDVI natural steppe)
RECOVERED = 0.85
RECOVERING = 0.60

# Palette taken from the BrBG ramp used on the maps
BARE = "#8C510A"      # bare ground
SAND = "#BF812D"      # sparse cover
PALE = "#F6E8C3"
MINT = "#C7EAE5"
STEPPE = "#01665E"    # like the steppe
INK = "#12322E"
MUTED = "#4F6662"
CANVAS = "#F7F9F8"
RAMP = [[0.0, "#543005"], [0.18, BARE], [0.36, SAND], [0.5, "#DFC27D"], [0.62, PALE],
        [0.74, MINT], [0.86, "#80CDC1"], [0.93, "#35978F"], [1.0, STEPPE]]
STATUS_COLOR = {"Not recovering": BARE, "Recovering": SAND, "Recovered": STEPPE}

st.set_page_config(page_title="ReclaimSat - mine waste recovery from space", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');
html, body, [class*="st-"], .stMarkdown, p, li, label, input, button, table {{
  font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif !important;
}}
h1, h2, h3 {{ font-family: 'Archivo', Arial, sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
.wordmark {{ font-weight: 800; font-stretch: 118%; font-size: 1.05rem; color: {STEPPE}; margin-bottom: 1.6rem; }}
.wordmark span {{ color: {MUTED}; font-weight: 400; font-stretch: 100%; }}
.lede {{ font-size: clamp(2rem, 4.2vw, 3.35rem); line-height: 1.08; font-weight: 750; font-stretch: 108%;
         color: {INK}; max-width: 22ch; margin: 0 0 1.1rem 0; }}
.lede .bare {{ color: {BARE}; }}
.lede .green {{ color: {STEPPE}; }}
.sub {{ font-size: 1.12rem; line-height: 1.55; color: {MUTED}; max-width: 62ch; margin-bottom: 2.2rem; }}
.section {{ font-weight: 750; font-stretch: 110%; font-size: 1.45rem; color: {INK}; margin: 2.6rem 0 0.3rem 0; }}
.note {{ color: {MUTED}; font-size: 0.98rem; max-width: 70ch; line-height: 1.5; }}
.ramp {{ height: 10px; border-radius: 5px; max-width: 420px; margin: 0.2rem 0 0.25rem 0;
         background: linear-gradient(90deg, #543005, {BARE}, {SAND}, #DFC27D, {PALE}, {MINT}, #80CDC1, #35978F, {STEPPE}); }}
.ramp-labels {{ display: flex; justify-content: space-between; max-width: 420px; color: {MUTED}; font-size: 0.9rem; }}
.obj {{ padding: 0.7rem 0 0.75rem 0; border-bottom: 1px solid #DCE5E2; }}
.obj b {{ color: {INK}; font-size: 1.02rem; }}
.obj .val {{ font-weight: 750; font-stretch: 112%; font-size: 1.6rem; line-height: 1.1; }}
.obj .st {{ font-size: 0.95rem; }}
footer, #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


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


def status(v: float) -> str:
    if v >= RECOVERED:
        return "Recovered"
    if v >= RECOVERING:
        return "Recovering"
    return "Not recovering"


wide, ri, dumps = load_series()
labels_path = DATA / "dump_labels.json"
labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
label = lambda d: labels.get(d, d)
short = lambda d: label(d).split("(")[0].strip()

first_years = ri.index[ri.index <= 2000]
last_years = ri.index[ri.index >= 2022]

rows = []
for d in dumps:
    ri_now = ri.loc[last_years, d].mean()
    rows.append({
        "id": d,
        "Object": short(d),
        "Status": status(ri_now),
        "RI 2022-26": ri_now,
        "RI 1990-2000": ri.loc[first_years, d].mean(),
        "Trend 2014-26, per decade": slope_per_decade(ri.loc[ri.index >= 2014, d]),
        "RI since 1990": ri[d].dropna().round(3).tolist(),
    })
summary = pd.DataFrame(rows)
summary["score"] = (1 - summary["RI 2022-26"]) - summary["Trend 2014-26, per decade"].fillna(0)
summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
summary.insert(0, "Priority", range(1, len(summary) + 1))
order = list(summary["id"])  # most urgent first

# Yearly maps: data/years/ (preferred), else data/, else the repo root
YEARS_DIR = next((d for d in (DATA / "years", DATA, ROOT) if (d / "years.json").exists()), DATA / "years")
years_meta = json.loads((YEARS_DIR / "years.json").read_text()) if (YEARS_DIR / "years.json").exists() else None
year_list = sorted(int(y) for y in years_meta["years"]) if years_meta else list(map(int, ri.index))
if "map_year" not in st.session_state:
    st.session_state["map_year"] = year_list[-1]


def take_click(event, key):
    """Apply a click on the strip once: move the map to that year."""
    pts = getattr(getattr(event, "selection", None), "points", None) or []
    sig = tuple(sorted({int(round(p["x"])) for p in pts if "x" in p}))
    if sig and st.session_state.get(f"_last_{key}") != sig:
        st.session_state["map_year"] = min(year_list, key=lambda v: abs(v - sig[0]))
        st.session_state["layer"] = LAYER_YEAR
    st.session_state[f"_last_{key}"] = sig


LAYER_YEAR = "One summer, compared with the steppe"
LAYER_S2 = "Summer 2026 in detail (10 m)"
LAYER_TREND = "Change since 1990"

# ---------------- Opening ----------------
worst = summary.iloc[0]
others = summary.iloc[1:]
best_two = summary.sort_values("RI 2022-26", ascending=False).head(2)
st.markdown('<div class="wordmark">ReclaimSat <span>· SSGPO mines, Rudny, Kazakhstan</span></div>',
            unsafe_allow_html=True)
st.markdown(
    f'<p class="lede">After {len(wide)} summers, the <span class="bare">{worst["Object"].lower()}</span> '
    f'is still a quarter as green as the steppe. The old rock dumps have <span class="green">nearly caught up</span>.</p>'
    if worst["RI 2022-26"] < 0.35 else
    f'<p class="lede">How green are the SSGPO waste dumps compared with the steppe around them?</p>',
    unsafe_allow_html=True)
st.markdown(
    f'<p class="sub">Landsat has photographed these mines every summer since 1990. For each dump we compare its '
    f'vegetation with untouched steppe in the same year, so dry and wet years cancel out. '
    f'1.0 means "as green as the steppe". In 2022-26 the {worst["Object"].lower()} scores '
    f'{worst["RI 2022-26"]:.2f}; {short(best_two.iloc[0]["id"]).lower()} scores {best_two.iloc[0]["RI 2022-26"]:.2f}.</p>',
    unsafe_allow_html=True)

# ---------------- Recovery strip (the one memorable element) ----------------
st.markdown('<div class="section">Every dump, every summer</div>', unsafe_allow_html=True)
st.markdown('<p class="note">One square per summer. Brown is bare ground, teal is as green as the steppe. '
            'Click a square to open that summer on the map.</p>', unsafe_allow_html=True)

xs, ys, cs, hov = [], [], [], []
for d in order[::-1]:
    for yr, v in ri[d].dropna().items():
        xs.append(int(yr)); ys.append(short(d)); cs.append(float(v))
        hov.append(f"{short(d)}<br>{int(yr)}: {v:.2f} of the steppe")
sel = st.session_state.get("map_year")
strip = go.Figure(go.Scatter(
    x=xs, y=ys, mode="markers", text=hov, hoverinfo="text",
    marker=dict(symbol="square", size=22, color=cs, cmin=0, cmax=1.1, colorscale=RAMP,
                line=dict(width=0)),
))
strip.add_vline(x=sel, line_width=2, line_color=INK)
strip.update_layout(
    height=72 + 46 * len(dumps), margin=dict(l=10, r=10, t=10, b=30),
    plot_bgcolor=CANVAS, paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Archivo, Arial, sans-serif", color=INK, size=14),
    xaxis=dict(showgrid=False, zeroline=False, dtick=5, range=[1989, 2027], fixedrange=True),
    yaxis=dict(showgrid=False, fixedrange=True), dragmode=False,
)
ev = st.plotly_chart(strip, width="stretch", on_select="rerun", selection_mode="points", key="strip",
                     config={"displayModeBar": False})
take_click(ev, "strip")
st.markdown('<div class="ramp"></div><div class="ramp-labels"><span>0 bare ground</span>'
            '<span>0.6</span><span>1.0 like the steppe</span></div>', unsafe_allow_html=True)

# ---------------- Map + per-object readout ----------------
st.markdown('<div class="section">On the map</div>', unsafe_allow_html=True)
layer_opts = [LAYER_YEAR, LAYER_S2, LAYER_TREND] if years_meta else [LAYER_S2, LAYER_TREND]
layer = st.radio("What to show", layer_opts, horizontal=True, key="layer", label_visibility="collapsed")
year = None
if layer == LAYER_YEAR:
    year = st.select_slider("Summer", options=year_list, key="map_year")

left, right = st.columns([5, 2], gap="large")
meta = json.loads((DATA / "rasters.json").read_text())
m = folium.Map(location=[53.0227, 63.1417], zoom_start=12, tiles=None, control_scale=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite photo",
).add_to(m)
if layer == LAYER_YEAR:
    ym = years_meta["years"][str(year)]
    folium.raster_layers.ImageOverlay(str(YEARS_DIR / f"ri_{year}.png"), bounds=years_meta["bounds"],
                                      opacity=0.82, name=f"Summer {year}").add_to(m)
    legend = (f"Summer {year}, Landsat 30 m. Same colours as the strip above. Image coverage {ym['coverage_pct']}%"
              + ("; gaps or stripes are missing satellite data." if ym["coverage_pct"] < 95 or 2003 <= year <= 2012
                 else "."))
elif layer == LAYER_S2:
    folium.raster_layers.ImageOverlay(str(DATA / "s2_ndvi_2026.png"), bounds=meta["s2"]["bounds"],
                                      opacity=0.78, name="Summer 2026").add_to(m)
    legend = "Sentinel-2, summer 2026, 10 m. Brown is bare ground, teal is vegetation."
else:
    folium.raster_layers.ImageOverlay(str(DATA / "ndvi_trend.png"), bounds=meta["trend"]["bounds"],
                                      opacity=0.78, name="Change since 1990").add_to(m)
    legend = ("Change in greenness per year, 1990-2026. Green: vegetation came back. "
              "Red: vegetation was lost, mostly where pits and dumps grew.")

dumps_geo = DATA / "dumps.geojson"
if dumps_geo.exists():
    gj = json.loads(dumps_geo.read_text())
    stat = dict(zip(summary["id"], summary["Status"]))
    now = dict(zip(summary["id"], summary["RI 2022-26"]))
    for f in gj["features"]:
        n = f["properties"].get("name")
        f["properties"]["label"] = short(n)
        f["properties"]["RI"] = f"{now.get(n, float('nan')):.2f} of the steppe (2022-26)"
        f["properties"]["status"] = stat.get(n, "")
    folium.GeoJson(
        gj, name="Monitored dumps",
        style_function=lambda f: {"color": "#FFFFFF", "weight": 5, "fillOpacity": 0},
    ).add_to(m)
    folium.GeoJson(
        gj, name="Status",
        style_function=lambda f: {"color": STATUS_COLOR.get(f["properties"]["status"], INK),
                                  "weight": 2.5, "fillOpacity": 0.06},
        tooltip=folium.GeoJsonTooltip(fields=["label", "RI", "status"], aliases=["", "", ""]),
    ).add_to(m)
folium.LayerControl(collapsed=True).add_to(m)

with left:
    html = m.get_root().render()
    if hasattr(st, "iframe"):
        st.iframe(html, height=560)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=560)
    st.markdown(f'<p class="note">{legend}</p>', unsafe_allow_html=True)

with right:
    yr = year if year else int(ri.index.max())
    head = f"Summer {yr}" if year else "Latest summer"
    st.markdown(f'<div class="section" style="margin-top:0">{head}</div>', unsafe_allow_html=True)
    for d in order:
        v = ri.at[yr, d] if yr in ri.index and pd.notna(ri.at[yr, d]) else None
        avg = summary.loc[summary["id"] == d, "RI 2022-26"].iloc[0]
        st_ = status(avg)
        val = f"{v:.2f}" if v is not None else "no data"
        st.markdown(
            f'<div class="obj"><b>{short(d)}</b><br>'
            f'<span class="val" style="color:{STATUS_COLOR[st_]}">{val}</span> '
            f'<span class="note">of the steppe</span><br>'
            f'<span class="st" style="color:{STATUS_COLOR[st_]}">{st_} (2022-26 average {avg:.2f})</span></div>',
            unsafe_allow_html=True)

# ---------------- Priority list ----------------
st.markdown('<div class="section">Where to spend reclamation money first</div>', unsafe_allow_html=True)
st.markdown(f'<p class="note">Ranked by how far each dump is from the steppe today and how slowly it is closing the gap. '
            f'Thresholds are indicative: recovered from {RECOVERED}, recovering from {RECOVERING}.</p>',
            unsafe_allow_html=True)
table = summary[["Priority", "Object", "Status", "RI 2022-26", "RI 1990-2000",
                 "Trend 2014-26, per decade", "RI since 1990"]]
st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={
        "Priority": st.column_config.NumberColumn(width="small"),
        "RI 2022-26": st.column_config.NumberColumn(format="%.2f"),
        "RI 1990-2000": st.column_config.NumberColumn(format="%.2f"),
        "Trend 2014-26, per decade": st.column_config.NumberColumn(format="%+.3f"),
        "RI since 1990": st.column_config.LineChartColumn(y_min=0, y_max=1.1, width="medium"),
    },
)

# ---------------- Why the steppe comparison ----------------
with st.expander("Why compare with the steppe? Raw greenness, year by year"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wide.index, y=wide[REF], name="Untouched steppe", mode="lines",
                             line=dict(color=MUTED, width=3, dash="dot")))
    shades = [BARE, SAND, "#35978F", STEPPE]
    for i, d in enumerate(order):
        fig.add_trace(go.Scatter(x=wide.index, y=wide[d], name=short(d), mode="lines+markers",
                                 line=dict(color=shades[i % len(shades)], width=2), marker=dict(size=4)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Summer NDVI",
                      plot_bgcolor=CANVAS, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Archivo, Arial, sans-serif", color=INK),
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<p class="note">In dry summers (2010, 2012, 2019, 2021) everything turns less green, the steppe '
                'included. Dividing each dump by the steppe removes that swing, which is what the strip and the '
                'map above show.</p>', unsafe_allow_html=True)

with st.expander("Method and limitations"):
    missing = ", ".join(str(y) for y in sorted(set(range(1990, 2027)) - set(int(y) for y in wide.index)))
    st.markdown(f"""
**Data.** Landsat 5, 7, 8 and 9 surface reflectance, June-July, clouds, shadows and snow removed, one median image
per summer. Landsat 5 and 7 adjusted to match Landsat 8 (Roy et al., 2016). Sentinel-2 for the 2026 detail map.
Processed in Google Earth Engine.

**The steppe benchmark.** 500 random pixels 8-18 km from the mine, mapped as grassland by ESA WorldCover 2021,
more than 1.5 km from the dumps, and with steady greenness in 2014-2026, which leaves out ploughed fields.

**Limits.** Greenness shows how much vegetation there is, not why: natural regrowth and company reclamation look the
same from space. 30 m pixels mix dump tops with slopes and roads along the edges. Grassland today does not prove the
land was never ploughed. No summer images for {missing}.
""")
    st.markdown("Code and data: [github.com/dvrmnzx/reclaimsat](https://github.com/dvrmnzx/reclaimsat)")
