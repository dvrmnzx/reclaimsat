// ReclaimSat - NDVI monitoring of mine waste dumps
// Site: SSGPO (Sokolov-Sarbai, ERG), Rudny, Kostanay region, Kazakhstan
// Run in Google Earth Engine Code Editor: https://code.earthengine.google.com
//
// BEFORE RUNNING - draw two geometry layers with the Code Editor drawing tools:
//   1) "dumps"     - import as FeatureCollection. Draw 4-6 waste dumps / tailings
//                    of different ages in ONE layer. Names dump_1, dump_2... are set automatically
//                    in drawing order - write down which number is which dump.
//   2) "reference" - OPTIONAL, only for REFERENCE_MODE = 'manual'. Import as Geometry.
//                    By default the reference steppe is selected automatically (see below).
// Rename the imports exactly "dumps" and "reference" (top of the script).

// ---------------- Settings ----------------
var CENTER = ee.Geometry.Point([63.1417, 53.0227]); // SSGPO, Global Energy Monitor (WGS84)
var AOI = CENTER.buffer(15000).bounds();
var START_YEAR = 1990;
var END_YEAR = 2026;
var MONTH_START = 6;  // June
var MONTH_END = 7;    // July - peak green in Kostanay steppe

Map.setCenter(63.1417, 53.0227, 12); // client-side, no server call
Map.setOptions('SATELLITE');

// ---------------- Landsat 5/7/8/9 (Collection 2, Level 2) ----------------
function maskLandsat(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
    .and(qa.bitwiseAnd(1 << 3).eq(0))        // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))        // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0));       // snow
  return img.updateMask(clear);
}

// Landsat 5, 7: red = B3, nir = B4.
// Cross-sensor harmonization ETM+ -> OLI (Roy et al. 2016, Remote Sensing of Environment,
// OLS coefficients as used in the Earth Engine Landsat harmonization tutorial).
// Without it, the switch to Landsat 8 in 2013 can create an artificial NDVI step.
var HARMONIZE = true;
function ndviTM(img) {
  var sr = img.select(['SR_B3', 'SR_B4']).multiply(0.0000275).add(-0.2);
  var red = sr.select('SR_B3');
  var nir = sr.select('SR_B4');
  if (HARMONIZE) {
    red = red.multiply(0.9047).add(0.0061);
    nir = nir.multiply(0.8462).add(0.0412);
  }
  return nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    .set('system:time_start', img.get('system:time_start'));
}

function ndviOLI(img) { // Landsat 8, 9: red = B4, nir = B5
  var sr = img.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
  return sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    .set('system:time_start', img.get('system:time_start'));
}

function landsat(id, ndviFn) {
  return ee.ImageCollection(id)
    .filterBounds(AOI)
    .filter(ee.Filter.calendarRange(MONTH_START, MONTH_END, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 60))
    .map(maskLandsat)
    .map(ndviFn);
}

var ls = landsat('LANDSAT/LT05/C02/T1_L2', ndviTM)
  .merge(landsat('LANDSAT/LE07/C02/T1_L2', ndviTM))
  .merge(landsat('LANDSAT/LC08/C02/T1_L2', ndviOLI))
  .merge(landsat('LANDSAT/LC09/C02/T1_L2', ndviOLI));

// ---------------- Annual summer composites ----------------
var years = ee.List.sequence(START_YEAR, END_YEAR);
var annual = ee.ImageCollection.fromImages(years.map(function (y) {
  var c = ls.filter(ee.Filter.calendarRange(y, y, 'year'));
  return c.median().set({
    'year': y,
    'n_scenes': c.size(),
    'system:time_start': ee.Date.fromYMD(y, 7, 1).millis()
  });
})).filter(ee.Filter.gt('n_scenes', 0)); // drop years with no data

print('Years with data:', annual.aggregate_array('year'));

// ---------------- NDVI trend (slope per year) ----------------
var trend = annual.map(function (img) {
  return ee.Image.constant(ee.Number(img.get('year'))).float().rename('t')
    .addBands(img.select('NDVI'));
}).reduce(ee.Reducer.linearFit()); // bands: scale (slope/yr), offset

// ---------------- Sentinel-2: current high-res NDVI ----------------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(END_YEAR + '-06-01', END_YEAR + '-07-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function (img) {
    var scl = img.select('SCL');
    var clear = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
    return img.updateMask(clear);
  })
  .median();
var s2ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');

// ---------------- Map layers ----------------
var ndviVis = {min: 0, max: 0.6, palette: ['8c510a', 'd8b365', 'f6e8c3', 'c7eae5', '5ab4ac', '01665e']};
var slopeVis = {min: -0.01, max: 0.01, palette: ['b2182b', 'f7f7f7', '1a9850']};

Map.addLayer(s2ndvi.clip(AOI), ndviVis, 'NDVI Sentinel-2 ' + END_YEAR);
Map.addLayer(trend.select('scale').clip(AOI), slopeVis, 'NDVI trend ' + START_YEAR + '-' + END_YEAR + ' (per year)', false);

// ---------------- Dumps ----------------
// Auto-name dumps in drawing order: dump_1, dump_2, ...
var dumpList = dumps.toList(dumps.size());
var dumpsNamed = ee.FeatureCollection(
  ee.List.sequence(0, dumps.size().subtract(1)).map(function (i) {
    return ee.Feature(dumpList.get(i))
      .set('name', ee.String('dump_').cat(ee.Number(i).add(1).format('%d')));
  })
);
Map.addLayer(dumpsNamed, {color: 'yellow'}, 'Dumps');

// ---------------- Reference steppe ----------------
// 'auto'   - all pixels that are grassland (ESA WorldCover 2021), 8-18 km from the mine,
//            >1.5 km from drawn dumps, and with low year-to-year NDVI variability
//            in 2014-2026 (one sensor family). Cropland / fallow rotation has high variability.
// 'manual' - use the drawn polygon "reference".
var REFERENCE_MODE = 'auto';
var SD_MAX = 0.08;   // max std. dev. of annual NDVI; lower = stricter. Tune after looking at the map.
var RING_IN = 8000, RING_OUT = 18000;

var ring = CENTER.buffer(RING_OUT).difference(CENTER.buffer(RING_IN));
var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var grass = worldcover.eq(30);
var ndviSd = annual.filter(ee.Filter.gte('year', 2014)).select('NDVI')
  .reduce(ee.Reducer.stdDev()).rename('NDVI_sd');
var nearDumps = ee.Image(0).paint(dumpsNamed.geometry().buffer(1500), 1); // 1 = within 1.5 km of a dump
var steppeMask = grass.and(ndviSd.lt(SD_MAX)).and(nearDumps.not())
  .clip(ring).selfMask();

Map.addLayer(ndviSd.clip(ring), {min: 0, max: 0.15, palette: ['1a9850', 'ffffbf', 'd73027']},
  'NDVI year-to-year SD 2014-2026 (red = cropland-like)', false);
Map.addLayer(steppeMask, {palette: ['00ff66']}, 'Auto reference steppe pixels');
Map.addLayer(ee.FeatureCollection([ee.Feature(ring)]).style({color: '00ff66', fillColor: '00000000', width: 1}),
  {}, 'Reference ring 8-18 km');

// Diagnostics (turn on only when tuning SD_MAX; several heavy requests at once can hit
// 'Too many concurrent aggregations')
var DIAGNOSTICS = false;
if (DIAGNOSTICS) {
print('Auto reference area, km2:', ee.Number(steppeMask.multiply(ee.Image.pixelArea())
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: ring, scale: 60, maxPixels: 1e10, tileScale: 4})
  .values().get(0)).divide(1e6));

// Diagnostics: how much grassland is in the ring, and typical NDVI variability there
print('Grassland in ring (WorldCover), km2:', ee.Number(grass.selfMask().multiply(ee.Image.pixelArea())
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: ring, scale: 60, maxPixels: 1e10, tileScale: 4})
  .values().get(0)).divide(1e6));
print('NDVI SD percentiles on grassland (p10, p25, p50):', ndviSd.updateMask(grass)
  .reduceRegion({reducer: ee.Reducer.percentile([10, 25, 50]), geometry: ring, scale: 60, maxPixels: 1e10, tileScale: 4}));
}

// To save memory, the reference is a fixed random sample of up to 500 steppe pixels
// (computed once), not the full mask re-evaluated for every year.
var refPoints = steppeMask.rename('m').sample({
  region: ring, scale: 30, numPixels: 20000, seed: 42, geometries: true, tileScale: 4
}).limit(500);
if (DIAGNOSTICS) print('Reference sample points:', refPoints.size());
Map.addLayer(refPoints, {color: 'ff00ff'}, 'Reference sample points', false);

// ---------------- Time series table: dumps + reference ----------------
// All years are stacked into ONE multi-band image, so the whole series needs only
// two aggregations (dumps + reference) instead of one per year.
var yearList = annual.aggregate_array('year');
var bandNames = yearList.map(function (y) {
  return ee.String('y').cat(ee.Number(y).toInt().format('%d'));
});
var stack = annual.select('NDVI').toBands().rename(bandNames);

var dumpStats = stack.reduceRegions({collection: dumpsNamed, reducer: ee.Reducer.mean(), scale: 30});
var refGeom = (REFERENCE_MODE === 'manual') ? reference : refPoints.geometry();
var refStats = stack.reduceRegion({reducer: ee.Reducer.mean(), geometry: refGeom, scale: 30, maxPixels: 1e10});

var table = ee.FeatureCollection(yearList.map(function (y) {
  var key = ee.String('y').cat(ee.Number(y).toInt().format('%d'));
  var d = dumpStats.map(function (f) {
    return ee.Feature(null, {name: f.get('name'), year: ee.Number(y).toInt(), mean: f.get(key)});
  });
  var r = ee.Feature(null, {name: 'reference_steppe', year: ee.Number(y).toInt(), mean: refStats.get(key)});
  return d.merge(ee.FeatureCollection([r]));
})).flatten();

print(ui.Chart.feature.groups(table, 'year', 'mean', 'name').setOptions({
  title: 'Summer NDVI: waste dumps vs natural steppe (Landsat 30 m, reference: ' + REFERENCE_MODE + ')',
  vAxis: {title: 'NDVI'},
  hAxis: {title: 'Year', format: '####'},
  lineWidth: 1,
  pointSize: 3
}));

// Mean NDVI trend per dump (slope per year)
print('NDVI trend per dump (slope/yr):', trend.select('scale').reduceRegions({
  collection: dumpsNamed, reducer: ee.Reducer.mean(), scale: 30
}));

// ---------------- Exports for the Streamlit app ----------------
Export.table.toDrive({
  collection: table,
  description: 'ndvi_timeseries_ssgpo',
  fileFormat: 'CSV',
  selectors: ['name', 'year', 'mean']
});

Export.image.toDrive({
  image: s2ndvi.clip(AOI),
  description: 'ndvi_s2_ssgpo_' + END_YEAR,
  region: AOI,
  scale: 10,
  maxPixels: 1e10
});

Export.image.toDrive({
  image: trend.select('scale').clip(AOI),
  description: 'ndvi_trend_ssgpo',
  region: AOI,
  scale: 30,
  maxPixels: 1e10
});

// Dump polygons for the map in the Streamlit app (save as data/dumps.geojson)
Export.table.toDrive({
  collection: dumpsNamed,
  description: 'dumps_ssgpo',
  fileFormat: 'GeoJSON'
});
