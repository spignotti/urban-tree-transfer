// Sentinel-2 preview script for Berlin and Leipzig (July 2022).
// Matches logic in src/urban_tree_transfer/data_processing/sentinel.py.

var year = 2022;
var month = 7;
var bufferM = 500;
var scale = 10;
var crs = "EPSG:25833";
var driveFolder = "sentinel2_processing_stage";
var exportToDrive = false;

var spectralBands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"];
var vegetationIndices = [
  "NDVI",
  "EVI",
  "GNDVI",
  "NDre1",
  "NDVIre",
  "CIre",
  "IRECI",
  "RTVIcore",
  "NDWI",
  "MSI",
  "NDII",
  "kNDVI",
  "VARI",
];

function addVegetationIndices(image) {
  var b2 = image.select("B2").toFloat();
  var b3 = image.select("B3").toFloat();
  var b4 = image.select("B4").toFloat();
  var b5 = image.select("B5").toFloat();
  var b6 = image.select("B6").toFloat();
  var b7 = image.select("B7").toFloat();
  var b8 = image.select("B8").toFloat();
  var b11 = image.select("B11").toFloat();
  var b12 = image.select("B12").toFloat();

  var b2s = b2.divide(10000.0);
  var b4s = b4.divide(10000.0);
  var b8s = b8.divide(10000.0);

  var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI");
  var gndvi = b8.subtract(b3).divide(b8.add(b3)).rename("GNDVI");
  var evi = b8s
    .subtract(b4s)
    .multiply(2.5)
    .divide(b8s.add(b4s.multiply(6)).subtract(b2s.multiply(7.5)).add(1))
    .rename("EVI");
  var vari = b3.subtract(b4).divide(b3.add(b4).subtract(b2)).rename("VARI");
  var ndre1 = b8.subtract(b5).divide(b8.add(b5)).rename("NDre1");
  var ndvire = b8.subtract(b6).divide(b8.add(b6)).rename("NDVIre");
  var cire = b8.divide(b5).subtract(1).rename("CIre");
  var ireci = b7.subtract(b4).divide(b5.divide(b6)).rename("IRECI");
  var rtvicore = b8
    .subtract(b5)
    .multiply(100)
    .subtract(b8.subtract(b3).multiply(10))
    .rename("RTVIcore");
  var ndwi = b8.subtract(b11).divide(b8.add(b11)).rename("NDWI");
  var msi = b11.divide(b8).rename("MSI");
  var ndii = b8.subtract(b12).divide(b8.add(b12)).rename("NDII");
  var kndvi = b8.subtract(b4).divide(b8.add(b4)).pow(2).tanh().rename("kNDVI");

  return image.addBands([
    ndvi,
    gndvi,
    evi,
    vari,
    ndre1,
    ndvire,
    cire,
    ireci,
    rtvicore,
    ndwi,
    msi,
    ndii,
    kndvi,
  ]);
}

function maskScl(image) {
  var scl = image.select("SCL");
  var mask = scl.eq(4).or(scl.eq(5));
  return image.updateMask(mask);
}

function clampBands(image) {
  return image.clamp(0, 10000);
}

var admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
  ee.Filter.eq("ADM0_NAME", "Germany"),
);
var admin1 = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
  ee.Filter.eq("ADM0_NAME", "Germany"),
);

function getCityGeometry(cityName) {
  var admin2City = admin2.filter(ee.Filter.eq("ADM2_NAME", cityName)).first();
  var admin1City = admin1.filter(ee.Filter.eq("ADM1_NAME", cityName)).first();
  var geom = ee.Algorithms.If(
    admin2City,
    ee.Feature(admin2City).geometry(),
    ee.Feature(admin1City).geometry(),
  );
  return ee.Geometry(geom);
}

function buildComposite(cityName, geom) {
  var startDate = ee.Date.fromYMD(year, month, 1);
  var endDate = startDate.advance(1, "month");
  var region = geom.buffer(bufferM);

  var collection = ee
    .ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate(startDate, endDate)
    .map(maskScl)
    .map(clampBands)
    .map(addVegetationIndices);

  var median = collection.median().clip(region);
  var outBands = spectralBands.concat(vegetationIndices);
  return median.select(outBands).toFloat();
}

var berlinGeom = getCityGeometry("Berlin");
var leipzigGeom = getCityGeometry("Leipzig");

var berlinComposite = buildComposite("Berlin", berlinGeom);
var leipzigComposite = buildComposite("Leipzig", leipzigGeom);

Map.centerObject(berlinGeom, 8);
Map.addLayer(
  berlinComposite.select(["B4", "B3", "B2"]),
  { min: 0, max: 3000 },
  "Berlin RGB",
);
Map.addLayer(
  berlinComposite.select("NDVI"),
  { min: -0.2, max: 0.9, palette: ["8c510a", "f6e8c3", "4d9221"] },
  "Berlin NDVI",
);

Map.addLayer(
  leipzigComposite.select(["B4", "B3", "B2"]),
  { min: 0, max: 3000 },
  "Leipzig RGB",
);
Map.addLayer(
  leipzigComposite.select("NDVI"),
  { min: -0.2, max: 0.9, palette: ["8c510a", "f6e8c3", "4d9221"] },
  "Leipzig NDVI",
);

if (exportToDrive) {
  Export.image.toDrive({
    image: berlinComposite,
    description: "S2_Berlin_2022_07_median",
    folder: driveFolder,
    region: berlinGeom.buffer(bufferM),
    scale: scale,
    crs: crs,
    maxPixels: 1e10,
  });

  Export.image.toDrive({
    image: leipzigComposite,
    description: "S2_Leipzig_2022_07_median",
    folder: driveFolder,
    region: leipzigGeom.buffer(bufferM),
    scale: scale,
    crs: crs,
    maxPixels: 1e10,
  });
}
