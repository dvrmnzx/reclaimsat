"""ReclaimSat - satellite monitoring of mine waste reclamation.
Case study: SSGPO (Sokolov-Sarbai, ERG), Rudny, Kazakhstan.
Data: Landsat 5/7/8/9 summer NDVI 1990-2026 and Sentinel-2 NDVI 2026, processed in Google Earth Engine.

Design: the palette is the NDVI colour ramp itself (bare ground brown -> steppe teal),
so the same colours mean the same thing in the text, the cards, the recovery strip and the map.
The page reads top to bottom as a story: the problem, how we measure it, the answer,
the evidence (strip, map), and what to do. Numbers and method sit in expanders at the end.

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
LINE = "#DCE5E2"
RAMP = [[0.0, "#543005"], [0.18, BARE], [0.36, SAND], [0.5, "#DFC27D"], [0.62, PALE],
        [0.74, MINT], [0.86, "#80CDC1"], [0.93, "#35978F"], [1.0, STEPPE]]
STATUS_COLOR = {"Not recovering": BARE, "Recovering": SAND, "Recovered": STEPPE}
# The same three statuses in everyday words
STATUS_PLAIN = {"Not recovering": "Needs help (not recovering)",
                "Recovering": "Grass is coming back (recovering)",
                "Recovered": "Almost like the steppe (recovered)"}
# What each kind of dump is, for readers who have never seen a mine
KIND = {"tailings": "Fine sand and mud left over after the iron is taken out of the rock.",
        "dump": "A hill of rock that was dug away to reach the iron ore."}

st.set_page_config(page_title="ReclaimSat - mine waste recovery from space", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');
html, body, [class*="st-"], .stMarkdown, p, li, label, input, button, table {{
  font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif !important;
}}
/* keep Streamlit's icon font, otherwise icons render as their names ("_arrow_right") */
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important;
}}
h1, h2, h3 {{ font-family: 'Archivo', Arial, sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}
.block-container {{ padding: 4.75rem 3rem 4rem 3rem; max-width: 100%; }}
@media (max-width: 640px) {{ .block-container {{ padding-left: 1rem; padding-right: 1rem; }} }}

.wordmark {{ font-weight: 800; font-stretch: 118%; font-size: 1.05rem; color: {STEPPE}; margin-bottom: 1.6rem; }}
.wordmark span {{ color: {MUTED}; font-weight: 400; font-stretch: 100%; }}
/* .stMarkdown p.x beats Streamlit's own "p {{ font-size: inherit }}" rule inside markdown blocks */
.stMarkdown p.lede, .lede {{ font-size: clamp(1.5rem, 2.5vw, 2.25rem); line-height: 1.12; font-weight: 750;
         font-stretch: 108%; color: {INK}; max-width: 30ch; margin: 0 0 1.2rem 0; }}
.lede .bare {{ color: {BARE}; }}
.lede .green {{ color: {STEPPE}; }}
.stMarkdown p.sub, .sub {{ font-size: 1.2rem; line-height: 1.55; color: {MUTED}; max-width: 56ch; margin-bottom: 0; }}

.section {{ font-weight: 750; font-stretch: 110%; font-size: 1.6rem; color: {INK}; margin: 3.6rem 0 0.35rem 0; }}
/* hero: SDG chips and the rounded NDVI image */
.sdgs {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.2rem; }}
.sdg {{ font-size: 0.9rem; font-weight: 600; color: {STEPPE}; background: {MINT}66; border: 1px solid {MINT};
        padding: 0.3rem 0.75rem; border-radius: 999px; }}
[data-testid="stImage"] img {{ border-radius: 16px; }}
.stMarkdown p.cap {{ font-size: 0.9rem; margin-top: 0.4rem; }}
.stMarkdown p.note, .note {{ color: {MUTED}; font-size: 1rem; max-width: 70ch; line-height: 1.5; }}

/* how we measure it: one row of three steps joined by thin arrows */
.flow {{ display: flex; align-items: flex-start; gap: 1.2rem; margin-top: 0.9rem; }}
.fstep {{ flex: 1 1 0; }}
.fstep h4 {{ margin: 0 0 0.3rem 0; padding: 0; font-size: 1.12rem; font-weight: 750; color: {INK}; }}
.fstep .fn {{ display: inline-block; min-width: 1.6rem; height: 1.6rem; line-height: 1.6rem; text-align: center;
              border-radius: 50%; background: {STEPPE}; color: #FFFFFF; font-size: 0.9rem; margin-right: 0.5rem; }}
.fstep p {{ margin: 0 0 0 2.1rem; color: {MUTED}; font-size: 0.98rem; line-height: 1.5; }}
.farrow {{ flex: 0 0 56px; height: 1px; background: {MUTED}; position: relative; margin-top: 0.8rem; }}
.farrow::after {{ content: ""; position: absolute; right: 0; top: -4px; border: 4.5px solid transparent;
                  border-left: 7px solid {MUTED}; border-right: 0; }}
.ridef {{ margin: 1.1rem 0 0 0; color: {INK}; font-size: 1rem; }}
@media (max-width: 760px) {{
  .flow {{ flex-direction: column; gap: 0.2rem; }}
  .farrow {{ flex: 0 0 22px; width: 1px; height: 22px; margin: 0 0 0 0.8rem; }}
  .farrow::after {{ right: auto; top: auto; bottom: 0; left: -4px; border: 4.5px solid transparent;
                    border-top: 7px solid {MUTED}; border-bottom: 0; }}
}}

/* the four dumps today */
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; margin-top: 1rem; }}
.card {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; padding: 1.15rem 1.25rem 1.1rem;
         border-top-width: 6px; }}
.card .name {{ font-weight: 750; font-size: 1.12rem; color: {INK}; }}
.card .what {{ color: {MUTED}; font-size: 0.92rem; line-height: 1.4; min-height: 2.6em; margin: 0.2rem 0 0.8rem; }}
.card .big {{ font-weight: 800; font-stretch: 112%; font-size: 2.3rem; line-height: 1; }}
.card .words {{ color: {INK}; font-size: 0.98rem; margin: 0.3rem 0 0.7rem; }}
.bar {{ position: relative; height: 12px; background: #EEF3F1; border-radius: 6px; }}
.bar .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }}
.bar .goal {{ position: absolute; top: -5px; bottom: -5px; width: 2px; background: {INK}; }}
.bar-labels {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: {MUTED}; margin: 0.25rem 0 0.8rem; }}
.pill {{ display: inline-block; font-weight: 700; font-size: 0.9rem; padding: 0.2rem 0.6rem; border-radius: 999px; }}
.trend {{ display: block; margin-top: 0.45rem; color: {MUTED}; font-size: 0.92rem; }}

.ramp {{ height: 10px; border-radius: 5px; max-width: 420px; margin: 0.2rem 0 0.25rem 0;
         background: linear-gradient(90deg, #543005, {BARE}, {SAND}, #DFC27D, {PALE}, {MINT}, #80CDC1, #35978F, {STEPPE}); }}
.ramp-labels {{ display: flex; justify-content: space-between; max-width: 420px; color: {MUTED}; font-size: 0.9rem; }}

/* map readout: one chip per dump for the chosen summer */
.chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.2rem 0 0.6rem; }}
.chip {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 10px; padding: 0.45rem 0.75rem;
         font-size: 0.95rem; color: {INK}; }}
.chip b {{ font-size: 1.1rem; }}

/* what to fix first */
.rank {{ display: grid; grid-template-columns: 2.6rem 1fr; gap: 0.2rem 0.8rem; align-items: baseline;
         padding: 0.85rem 0; border-bottom: 1px solid {LINE}; max-width: 760px; }}
.rank .no {{ font-weight: 800; font-stretch: 115%; font-size: 1.8rem; line-height: 1; }}
.rank .t {{ font-weight: 750; font-size: 1.1rem; color: {INK}; }}
.rank .why {{ grid-column: 2; color: {MUTED}; font-size: 0.98rem; }}

.glossary dt {{ font-weight: 750; color: {INK}; margin-top: 0.6rem; }}
.glossary dd {{ margin: 0.1rem 0 0 0; color: {MUTED}; max-width: 70ch; }}
footer, #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


def html(block: str):
    """Render a block of our own HTML (cards, steps) with the page styles above."""
    if hasattr(st, "html"):
        st.html(block)
    else:
        st.markdown(block, unsafe_allow_html=True)


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


def as_green(v: float) -> str:
    """Recovery Index in words a child can follow."""
    for lim, words in [(0.2, "Less than a fifth"), (0.3, "About a quarter"), (0.4, "About a third"),
                       (0.55, "About half"), (0.7, "About two thirds"), (0.85, "About three quarters"),
                       (0.95, "Almost")]:
        if v < lim:
            return f"{words} as green as the steppe"
    return "As green as the steppe"


def trend_words(t: float) -> str:
    if pd.isna(t):
        return "Not enough summers to tell"
    if t >= 0.08:
        return "Getting greener quickly since 2014"
    if t >= 0.03:
        return "Slowly getting greener since 2014"
    if t > -0.03:
        return "Hardly changing since 2014"
    return "Getting less green since 2014"


wide, ri, dumps = load_series()
labels_path = DATA / "dump_labels.json"
labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
label = lambda d: labels.get(d, d)
short = lambda d: label(d).split("(")[0].strip()
# lowercase only the first letter inside sentences, so "Tailings facility N" -> "tailings facility N"
inline = lambda s: s[:1].lower() + s[1:]
kind = lambda d: KIND["tailings"] if "tailings" in short(d).lower() else KIND["dump"]

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
by_id = summary.set_index("id")

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


LAYER_YEAR = "One summer"
LAYER_S2 = "Summer 2026, close-up"
LAYER_TREND = "What changed since 1990"

# ---------------- 1. Opening ----------------
worst = summary.iloc[0]
others = summary.iloc[1:]
NUM = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
num = lambda n: NUM.get(int(n), str(int(n)))
caught = int((others["RI 2022-26"] >= RECOVERING).sum())
gap_years = sorted(set(range(1990, int(wide.index.max()) + 1)) - set(int(y) for y in wide.index))
gaps_txt = ", ".join(str(y) for y in gap_years)
if caught == len(others):
    caught_txt = f'All {num(len(others))} other dumps have nearly caught up.'
elif caught:
    caught_txt = f'{num(caught).capitalize()} of the {num(len(others))} other dumps have nearly caught up.'
else:
    caught_txt = f'None of the {num(len(others))} other dumps has caught up yet.'
span_years = int(wide.index.max()) - 1990  # 36 -> "30+ years"
worst_name = "tailings facility" if "tailings" in worst["Object"].lower() else inline(worst["Object"])

st.markdown('<div class="wordmark">ReclaimSat <span>· SSGPO iron mines, Rudny, Kazakhstan</span></div>',
            unsafe_allow_html=True)
hero_text, hero_img = st.columns([1, 1], gap="large", vertical_alignment="center")
with hero_text:
    st.markdown(
        f'<p class="lede">After {span_years // 10 * 10}+ years, the {worst_name} is still a quarter as green '
        f'as the steppe.</p>'
        if worst["RI 2022-26"] < 0.35 else
        '<p class="lede">How green are the SSGPO waste dumps compared with the steppe around them?</p>',
        unsafe_allow_html=True)
    st.markdown(
        f'<p class="sub">{caught_txt} Digging out iron ore leaves behind huge hills of rock and sand. '
        'Can grass grow back on them by itself, or do they need help? '
        'Satellites have photographed these mines every summer since 1990, so we can check.</p>',
        unsafe_allow_html=True)
    html('<div class="sdgs"><span class="sdg">SDG 15 · Life on Land</span>'
         '<span class="sdg">SDG 12 · Responsible production</span></div>')
with hero_img:
    hero_path = DATA / "hero.webp"
    if hero_path.exists():
        st.image(str(hero_path), width="stretch")
        st.markdown('<p class="note cap">Sentinel-2, summer 2026. Brown = bare ground, teal = vegetation.</p>',
                    unsafe_allow_html=True)

# ---------------- 2. How we measure it ----------------
st.markdown('<div class="section">How we measure it</div>', unsafe_allow_html=True)
html("""
<div class="flow">
  <div class="fstep"><h4><span class="fn">1</span>A photo every summer</h4>
    <p>Landsat satellites photograph the mines every June and July, since 1990.</p></div>
  <div class="farrow" aria-hidden="true"></div>
  <div class="fstep"><h4><span class="fn">2</span>Measure the green</h4>
    <p>Living plants reflect sunlight in a special way, so the photo shows how much grass grows.</p></div>
  <div class="farrow" aria-hidden="true"></div>
  <div class="fstep"><h4><span class="fn">3</span>Compare with wild steppe</h4>
    <p>Each dump is compared with untouched grassland in the same summer, so a dry year does not fool us.</p></div>
</div>
<p class="ridef"><b>We call this the Recovery Index (RI): 1.0 = as green as the steppe, 0 = bare ground.</b></p>
""")

# ---------------- 3. The answer: four cards ----------------
st.markdown('<div class="section">The four dumps today</div>',
            unsafe_allow_html=True)
st.markdown('<p class="note">Average of the last five summers (2022-26). The black tick on each bar is 1.0, '
            'the green of the wild steppe.</p>', unsafe_allow_html=True)
SCALE = 1.1  # bars run from 0 to 1.1 so the 1.0 tick is visible
cards = []
for d in order:
    r = by_id.loc[d]
    v, s = float(r["RI 2022-26"]), r["Status"]
    c = STATUS_COLOR[s]
    cards.append(
        f'<div class="card" style="border-top-color:{c}">'
        f'<div class="name">{short(d)}</div><div class="what">{kind(d)}</div>'
        f'<div class="big" style="color:{c}">{v:.2f}</div>'
        f'<div class="words">{as_green(v)}</div>'
        f'<div class="bar"><div class="fill" style="width:{min(v / SCALE, 1) * 100:.1f}%;background:{c}"></div>'
        f'<div class="goal" style="left:{100 / SCALE:.1f}%"></div></div>'
        f'<div class="bar-labels"><span>0 bare</span><span>1.0 steppe</span></div>'
        f'<span class="pill" style="background:{c}1A;color:{c}">{STATUS_PLAIN[s]}</span>'
        f'<span class="trend">{trend_words(r["Trend 2014-26, per decade"])}</span></div>')
html('<div class="cards">' + "".join(cards) + "</div>")

# ---------------- 4. Evidence: recovery strip ----------------
st.markdown('<div class="section">Every dump, every summer</div>',
            unsafe_allow_html=True)
st.markdown('<p class="note">Each square is one summer. Brown = bare ground, teal = as green as the steppe. '
            'Click a square to see that summer on the map below. '
            f'Gaps: no summer images in {gaps_txt}. The dark line marks the summer shown on the map.</p>',
            unsafe_allow_html=True)

xs, ys, cs, hov = [], [], [], []
for d in order[::-1]:
    for yr, v in ri[d].dropna().items():
        xs.append(int(yr)); ys.append(short(d)); cs.append(float(v))
        hov.append(f"{short(d)}<br>Summer {int(yr)}: {v:.2f}<br>{as_green(float(v))}")
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

# ---------------- 5. Map ----------------
st.markdown('<div class="section">See it on the map</div>', unsafe_allow_html=True)
layer_opts = [LAYER_YEAR, LAYER_S2, LAYER_TREND] if years_meta else [LAYER_S2, LAYER_TREND]
layer = st.radio("What to show", layer_opts, horizontal=True, key="layer", label_visibility="collapsed")
year = None
if layer == LAYER_YEAR:
    year = st.select_slider("Summer", options=year_list, key="map_year")

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
    legend = "Sentinel-2, summer 2026, 10 m: sharper than Landsat. Brown is bare ground, teal is vegetation."
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
        f["properties"]["status"] = STATUS_PLAIN.get(stat.get(n, ""), "")
    folium.GeoJson(
        gj, name="Monitored dumps",
        style_function=lambda f: {"color": "#FFFFFF", "weight": 5, "fillOpacity": 0},
    ).add_to(m)
    plain_to_color = {STATUS_PLAIN[k]: v for k, v in STATUS_COLOR.items()}
    folium.GeoJson(
        gj, name="Status",
        style_function=lambda f: {"color": plain_to_color.get(f["properties"]["status"], INK),
                                  "weight": 2.5, "fillOpacity": 0.06},
        tooltip=folium.GeoJsonTooltip(fields=["label", "RI", "status"], aliases=["", "", ""]),
    ).add_to(m)
folium.LayerControl(collapsed=True).add_to(m)

# one chip per dump for the summer on the map
yr = year if year else int(ri.index.max())
chips = []
for d in order:
    v = ri.at[yr, d] if yr in ri.index and pd.notna(ri.at[yr, d]) else None
    c = STATUS_COLOR[by_id.loc[d, "Status"]]
    chips.append(f'<div class="chip">{short(d)} &nbsp;<b style="color:{c}">'
                 f'{f"{v:.2f}" if v is not None else "no data"}</b></div>')
st.markdown(f'<p class="note" style="margin-bottom:0.2rem"><b>Summer {yr}</b>, compared with the steppe:</p>',
            unsafe_allow_html=True)
html('<div class="chips">' + "".join(chips) + "</div>")

map_html = m.get_root().render()
if hasattr(st, "iframe"):
    st.iframe(map_html, height=560)
else:
    import streamlit.components.v1 as components
    components.html(map_html, height=560)
st.markdown(f'<p class="note">{legend}</p>', unsafe_allow_html=True)
st.markdown('<p class="note">Outlined: the four dumps in this analysis; outline colour = status. '
            'Other brown areas (pits, the town, pond shores) are bare ground but are not analysed.</p>',
            unsafe_allow_html=True)

# ---------------- 6. What to do ----------------
st.markdown('<div class="section">Which dump needs help first</div>',
            unsafe_allow_html=True)
st.markdown('<p class="note">Ranked by how far each dump is from the steppe today and how slowly it is catching up.</p>',
            unsafe_allow_html=True)
ranks = []
for _, r in summary.iterrows():
    c = STATUS_COLOR[r["Status"]]
    ranks.append(f'<div class="rank"><div class="no" style="color:{c}">{r["Priority"]}</div>'
                 f'<div class="t">{r["Object"]}</div>'
                 f'<div class="why">{as_green(r["RI 2022-26"])} ({r["RI 2022-26"]:.2f}). '
                 f'{trend_words(r["Trend 2014-26, per decade"])}.</div></div>')
html("".join(ranks))

# ---------------- 7. Details for those who want them ----------------
st.markdown('<div class="section" style="font-size:1.25rem">More details</div>', unsafe_allow_html=True)

with st.expander("All the numbers"):
    st.markdown(f'<p class="note">Thresholds are indicative: recovered from {RECOVERED}, recovering from {RECOVERING}. '
                f'Change per 10 years: +0.1 is clear regrowth, about 0 is no change.</p>',
                unsafe_allow_html=True)
    table = summary[["Priority", "Object", "Status", "RI 2022-26", "RI 1990-2000",
                     "Trend 2014-26, per decade", "RI since 1990"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Priority": st.column_config.NumberColumn("#", width="small"),
            "RI 2022-26": st.column_config.NumberColumn("Recovery index 2022-26", format="%.2f"),
            "RI 1990-2000": st.column_config.NumberColumn("Recovery index 1990-2000", format="%.2f"),
            "Trend 2014-26, per decade": st.column_config.NumberColumn(
                "Change per 10 years (2014-26)", format="%+.3f",
                help="+0.1 is clear regrowth, about 0 is no change"),
            "RI since 1990": st.column_config.LineChartColumn("Recovery index since 1990", y_min=0, y_max=1.1),
        },
    )

with st.expander("Words used on this page"):
    html(f"""<dl class="glossary">
<dt>Steppe</dt><dd>The wide, flat grassland that covers this part of Kazakhstan.</dd>
<dt>Waste dump</dt><dd>{KIND["dump"]}</dd>
<dt>Tailings facility</dt><dd>{KIND["tailings"]}</dd>
<dt>Reclamation</dt><dd>Covering a dump with soil and planting grass so the land becomes useful again.</dd>
<dt>NDVI</dt><dd>A satellite measure of green vegetation: about 0 is bare ground, 0.5 and above is dense grass.</dd>
<dt>Recovery Index (RI)</dt><dd>A dump's NDVI divided by the steppe's NDVI in the same summer.
1.0 = as green as the steppe, 0 = bare ground.</dd>
</dl>""")

with st.expander("Why compare with the steppe? Raw greenness, year by year"):
    st.markdown('<p class="note">NDVI is a satellite measure of green vegetation: about 0 is bare ground, '
                '0.5 and above is dense grass.</p>', unsafe_allow_html=True)
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
land was never ploughed. The small old dump NW is about 300 m across, so edge pixels may raise its score.
No summer images for {missing}.
""")
    st.markdown("Code and data: [github.com/dvrmnzx/reclaimsat](https://github.com/dvrmnzx/reclaimsat)")
