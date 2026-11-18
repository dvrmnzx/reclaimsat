import json, sys
import numpy as np, rasterio
from rasterio.enums import Resampling
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PIL import Image

def read(src, max_w=1300):
    with rasterio.open(src) as r:
        scale = min(1, max_w / r.width)
        h, w = int(r.height*scale), int(r.width*scale)
        return r.read(1, out_shape=(h, w), resampling=Resampling.average).astype('float32'), r.bounds

def water_mask(shape, up):
    with rasterio.open(f'{up}/ndvi_s2_ssgpo_2026.tif') as r:
        a = r.read(1, out_shape=shape, resampling=Resampling.average)
    return a < 0  # open water has negative NDVI

def render(src, out_png, vmin, vmax, cmap, max_w=1300):
    with rasterio.open(src) as r:
        scale = min(1, max_w / r.width)
        h, w = int(r.height*scale), int(r.width*scale)
        a = r.read(1, out_shape=(h, w), resampling=Resampling.average).astype('float32')
        b = r.bounds
    rgba = colormaps[cmap](Normalize(vmin, vmax, clip=True)(a))
    wm = water_mask(a.shape, UP)
    rgba[..., 3] = np.where(np.isfinite(a) & ~wm, 0.85, 0)
    Image.fromarray((rgba*255).astype('uint8')).save(out_png, optimize=True)
    return [[b.bottom, b.left], [b.top, b.right]], float(np.nanpercentile(a, 2)), float(np.nanpercentile(a, 98))

up = sys.argv[1]; out = sys.argv[2]; UP = up
meta = {}
meta['s2'] = render(f'{up}/ndvi_s2_ssgpo_2026.tif', f'{out}/s2_ndvi_2026.png', 0, 0.6, 'BrBG')
meta['trend'] = render(f'{up}/ndvi_trend_ssgpo.tif', f'{out}/ndvi_trend.png', -0.01, 0.01, 'RdYlGn')
json.dump({k: {'bounds': v[0], 'p2': v[1], 'p98': v[2]} for k, v in meta.items()}, open(f'{out}/rasters.json', 'w'), indent=1)
print(json.dumps(meta, indent=1))
