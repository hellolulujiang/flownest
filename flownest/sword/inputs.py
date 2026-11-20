"""The inputs of the SWORD crosswalk and where the work files go, set once by configure() before any step runs.

SW     folder of the SWORD v17c GeoPackages sword_{af,as,eu,na,oc,sa}_v17c.gpkg (build 0.0.15, Zenodo 23870877)
FN     folder of the FlowNest release (MERIT Hydro 90 m, 1 km2 base)
UPA    MERIT Hydro upstream area (km2), on the release grid
WTH    MERIT Hydro river width (m), on the release grid
WORK   folder of the work files of this run: FACE (systems and basins), BALL (all basins, also those under 1 km2)
"""
import os
import numpy as np

SW = FN = UPA = WTH = WORK = FACE = BALL = None
ARGS = None   # the arguments of configure(), handed to worker processes
CONTINENTS = ('AF', 'AS', 'EU', 'NA', 'OC', 'SA')


def configure(sword_gpkg_dir, release_dir, upa_tif, wth_tif, work_dir):
    """Set the inputs and the work folder (made here if missing)."""
    global SW, FN, UPA, WTH, WORK, FACE, BALL, ARGS
    ARGS = (sword_gpkg_dir, release_dir, upa_tif, wth_tif, work_dir)
    SW, FN, WORK = (os.path.join(os.path.abspath(p), '') for p in (sword_gpkg_dir, release_dir, work_dir))
    UPA, WTH = os.path.abspath(upa_tif), os.path.abspath(wth_tif)
    FACE, BALL = WORK + 'face/', WORK + 'all_basins.parquet'
    for c in CONTINENTS:
        gpkg_path(c)
    for f in (FN + 'metadata.json', UPA, WTH):
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    os.makedirs(FACE, exist_ok=True)


def gpkg_path(cont, folder=None):
    """Path of one continent's SWORD v17c GeoPackage, sword_{cont}_v17c.gpkg with the continent code in any case, as the
    file is named on disk (so that the same code runs on file systems that tell upper from lower case)."""
    folder = folder or SW
    on_disk = {n.lower(): n for n in os.listdir(folder)}
    name = on_disk.get(f'sword_{cont.lower()}_v17c.gpkg')
    if name is None:
        raise FileNotFoundError(f'no SWORD v17c GeoPackage for {cont} in {folder}')
    return os.path.join(folder, name)


class Grid:
    """A raster of the whole globe on the release grid, read by windows of pixels: width, height, gt (x0, dx, 0, y0, 0, dy) and
    read(x0, y0, w, h), which refuses a window off the grid."""

    def __init__(self, path):
        import rasterio
        self.src = rasterio.open(path)
        self.width, self.height = self.src.width, self.src.height
        self.gt = self.src.transform.to_gdal()

    def read(self, x0, y0, w, h):
        from rasterio.windows import Window
        x0, y0, w, h = int(x0), int(y0), int(w), int(h)
        if not (0 <= x0 and 0 <= y0 and w > 0 and h > 0 and x0 + w <= self.width and y0 + h <= self.height):
            raise ValueError(f'window {x0}, {y0}, {w} x {h} is off the grid {self.width} x {self.height}')
        return self.src.read(1, window=Window(x0, y0, w, h))


def same_grid(a, b):
    """Whether two Grids share the geotransform."""
    return np.allclose(a.gt, b.gt)
