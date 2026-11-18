# ReclaimSat

Satellite monitoring of mine waste reclamation. Case study: SSGPO (ERG), Rudny, Kazakhstan.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. The map needs internet (Leaflet and satellite tiles load online).

## Files
- `app.py` - Streamlit app
- `data/ndvi_timeseries.csv` - GEE export (summer NDVI per dump + reference steppe, 1990-2026)
- `data/s2_ndvi_2026.png`, `data/ndvi_trend.png`, `data/rasters.json` - map overlays made from GEE GeoTIFFs
- `data/dump_labels.json` - display names for dumps (edit freely)
- `data/dumps.geojson` - optional, dump outlines from GEE export `dumps_ssgpo`
- `gee_reclaimsat_ssgpo.js` - Google Earth Engine script
- `prep_rasters.py` - converts GEE GeoTIFFs to overlays: `python prep_rasters.py <folder_with_tifs> data`
  (needs rasterio, matplotlib, pillow; expects ndvi_s2_ssgpo_2026.tif and ndvi_trend_ssgpo.tif)

## Deploy (free)
Push this folder to a public GitHub repo, then share.streamlit.io -> New app -> app.py.
