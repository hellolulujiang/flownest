# FlowNest user guide

[Introduction](../README.md).


## Overview

FlowNest numbers the channel pixels of a D8 flow-direction grid depth first, as described in the FlowNest paper
(Jiang et al., submitted to Water Resources Research). The channel pixels upstream of any channel pixel are then one range
of integers, and each reach, path, basin and group of basins holds one range of channel pixels and one range of reaches.
A signed raster gives each hillslope pixel minus the number of the channel pixel it first drains to, so the land upstream
of a pixel is selected by the same range.

This 4 × 4 MERIT Hydro window at the Dungeness mouth (zero-based rows 44214:44218, columns 68244:68248) was numbered at 1 km²:

```text
D8:                    Signed raster:
[[  0   0   0   0]      [[ 1  0  0  0]
 [ 64 128  64 128]       [ 2  0  0  0]
 [ 64  16 128 128]       [ 3 -3  0  0]
 [ 64  32  16 128]]      [-3  4 -4  0]]
upstream of 1: range(1, 5)
```

Here `1 <= abs(v) < 5` selects the seven pixels upstream of pixel 1, including itself. Positive values identify channel pixels;
negative values identify hillslope pixels draining first to that channel pixel; 0 reaches no numbered channel within this window.

FullNest, the numbering of MERIT Hydro made with FlowNest, is in two records:
[global files at 1 km²](https://doi.org/10.5281/zenodo.24393217) and
[coarser thresholds and crosswalks](https://doi.org/10.5281/zenodo.24393244).

Documentation and the other FullHydro tools: <https://fullhydro.org/tools/>

Author: Lulu Jiang (<https://lulujiang.me>)


## Install

```sh
git clone https://github.com/hellolulujiang/flownest.git
cd flownest
pip install .                 # numpy and numba: number a grid
pip install '.[release]'      # adds pyarrow and rasterio: read a release
pip install '.[crosswalks]'   # adds pyogrio, geopandas and shapely too: place gauges and MERIT-Basins on a release
pip install '.[test]'         # pytest and all of the above
```

Python 3.9 or later. Version 1.0.0 gives the same numbers as FullNest `20261113_195421`; see [Checks against the release](#checks-against-the-release).


## Number a grid

With same-shaped arrays `d8` and `upa` (upstream area in km²):

```python
from flownest import number_grid, verify
n = number_grid(d8, upa, threshold=1.0, nodata=247)
p = int(n.raster[n.raster > 0][0])
upstream = n.upstream(p)  # range(p, p + n.upstream_channel_pixel_count[p - n.first])
mask = (abs(n.raster) >= upstream.start) & (abs(n.raster) < upstream.stop)
assert all(count == 0 for count in verify(n, d8, upa, 1.0, nodata=247).values())
```

The default is `nodata=255`; pass `nodata=247` for MERIT Hydro (247 = sea, 0 = river mouth, 255 = inland sink; mouths and sinks
are basin outlets), or `nodata=np.nan` for NaN nodata. Direction codes 1, 2, 4, 8, 16, 32, 64, 128 mean E, SE, S, SW, W, NW, N, NE.
Other D8 codes can be passed as `codes={code: (row_step, col_step), ...}` to `number_grid` and `verify`.

`examples/quickstart.py` runs a made-up grid. `examples/number_a_basin.py` numbers GeoTIFFs and writes the signed raster, reach
table and area-weighted upstream means. `area` and `threshold` share units; `upstream_area_km2` retains the input area's units.
`n.reaches(T)` and `n.unit_catchments(T)` require `T >= n.threshold`. Areas and thresholds are compared in float32, as in the
release (MERIT Hydro's upstream area is float32): a pixel is a channel pixel at T when `upa >= float32(T)`, so `11.2` and
`np.float32(11.2)` are the same threshold, and a float64 or float32 area gives the same numbering.

**Reaches and paths.** At a confluence the path continues into the inflowing reach with the largest upstream area; other branches start paths. `path_level` is 1 on the path reaching the basin outlet and one more than the path it joins elsewhere. The confluence pixel belongs to the downstream reach as its upstream-end pixel.


## The released files

FullNest (see [Overview](#overview)) is read with `Release`, which reads only the required row groups of large tables;
row k of the pixel, reach and reach-location tables holds id k + 1. Data file names start with `flownest_merit90m_`:

| File after the prefix | Contents |
|---|---|
| `pixel_seq_id_1km2.tif`, `reach_seq_id_1km2.tif` | signed pixel and reach ids; 0 for land in basins below 0.05 km², −2147483648 for sea/nodata |
| `pixel_1km2.parquet` | 8 pixel columns, including `upstream_channel_pixel_count` |
| `reach_1km2.parquet` | 14 reach columns |
| `reach_location_1km2.parquet` | 11 columns: id, downstream-end coordinates, unit and upstream catchment windows |
| `reach_line_1km2.geo.parquet`, `unit_catchment_1km2.geo.parquet` | `reach_seq_id`, `geometry`, `bbox`; join attributes by id |
| `basin.parquet` | 16 columns, one row for each basin of at least 1 km²; windows are `row_min`, `row_max`, `col_min`, `col_max` |
| `basin.csv` | the same basin table as CSV |
| `basin_id.tif` | `basin_id` of every land pixel of a basin of at least 1 km²; 0 for other land, −2147483648 for sea/nodata |
| `basin.geo.parquet` | basin polygons (`basin_id`, `geometry`, `bbox`) |
| `basin.gpkg` | the same polygons in one layer, `basin` (`fid`, `geometry`, `basin_id`; no `bbox`) |

`basin_id` is FullBasin's global basin ID (releases with `code_version` 10.58 or later in `metadata.json`). The basin table holds `1 .. basin_file_count` (340,991 in the
global release), basin b on row b − 1. A basin of 0.05 to 1 km² has a larger `basin_id` and is in the pixel and reach tables only;
`rel.basin(b)` raises `KeyError` for it, and `rel.basin_of(p)` reads it from the pixel table. The ids do not follow the pixel order.
The FlowDivide level-3 groups are in `metadata.json` (`flowdivide_level3_groups`; `rel.group(g)` returns one, with its basin count,
runs of ids and window); there is no group table.

`metadata.json` must have `schema_version: "4"` and `basin_file_count`; releases with a `code_version` before 10.58, which
number `basin_id` otherwise, are refused. The crosswalk packages (GRDC stations, SWORD reaches and nodes, MERIT-Basins) are
tables to read with pandas or pyarrow, not folders for `Release`; their `schema_version` (4 or 5) does not tell them apart.

```python
import numpy as np
from flownest import Release
rel = Release('/path/to/flownest_merit90m_global_DATE')
b = rel.basin(10670)  # Dungeness
window = rel.window_of(b, prefix='')
first, count = b['outlet_pixel_seq_id'], b['channel_pixel_count']
pixels = rel.pixels(first, count, columns=['upstream_channel_pixel_count'])
first_reach, reach_count = rel.reach_range(b['basin_id'])
reaches = rel.reaches(first_reach, reach_count)
locations = rel.rows('reach_location_1km2.parquet', first_reach, reach_count)
v = rel.values_at([44214, 44216, 44214, 44214], [68244, 68245, 68185, 67941])
ids = np.abs(v[(v != 0) & (v != rel.nodata)].astype(np.int64))
points = rel.rows_at('pixel_1km2.parquet', ids)
```

`rows(name, first, count)`, `reaches(first, count)` and `pixels(first, count)` read `[first, first + count)`.
`reach_range(basin_id, T)` returns `(first, count)`, or `None` for a basin with no reach at T. `rel.upstream(p)` returns a Python
`range`. `rows_at` preserves the order and repetition of scattered ids. `window_of(location)` returns the number raster inside the window of that row, from `upstream_catchment_`;
pass `prefix='unit_catchment_'` for the unit catchment and `prefix=''` for a basin or FlowDivide level-3 group.
All spatial windows use `[row_min, row_max)` and `[col_min, col_max)`. Negative `col_min` wraps across 180 degrees.

**Count and distance conventions.** Every `upstream_*_count` includes the object itself. A reach's own pixels are
`[downstream_end_pixel_seq_id, downstream_end_pixel_seq_id + channel_pixel_count)`; its upstream pixels use
`upstream_channel_pixel_count` from the same first id. Its upstream reaches are `[reach_seq_id, reach_seq_id + upstream_reach_count)`.
`distance_to_basin_outlet_km` is measured from the end of the reach line, the centre of the pixel it flows into. Thus
`length_km + distance_to_basin_outlet_km` is the distance from the upstream end to the basin outlet.
`break_threshold_km2` is the largest threshold at which a reach still begins at the downstream-end pixel of the fine reach: the upstream area at a basin outlet,
or elsewhere the smaller of its own upstream area and the largest other donor's upstream area entering the reach below.

A channel pixel has at least 1 km² upstream, and the outlet pixel of every basin of 0.05 to 1 km² is one too: such a basin is one
pixel, one reach and one path in the 1 km² tables, with `upstream_area_km2` and `break_threshold_km2` below 1, and in no network
above 1 km². `Release` keeps it at the base threshold and drops it above. `number_basin` takes only the basins of the basin
table (`basin_id` 1 to its row count); to number a smaller basin, call `number_grid(..., outlets=[(row, col)], outlets_are_channel_pixels=True)`
on its window, which makes that given outlet a channel pixel as the release does.

The downstream pixel and headwater flag are derived. For reach r, read the reach named by `next_down_reach_seq_id`:
its `downstream_end_pixel_seq_id + channel_pixel_count - 1` is r's downstream pixel. At a basin outlet
(`next_down_reach_seq_id == reach_seq_id`), use r's own `downstream_end_pixel_seq_id`. A headwater has `upstream_reach_count == 1`.


## Any threshold from the base tables

```python
r = rel.reaches_at(11.2, basin_id=10670)
first, crosswalk = rel.reach_at_threshold(11.2, basin_id=10670)
# crosswalk[i] is reach_seq_id_11p2km2 for base reach first + i, or 0 for no network in its basin
w = rel.window_at(11.2, b['row_min'], b['row_max'], b['col_min'], b['col_max'])
```

Use `flowdivide_level3_id=...` to select a whole group, its basins under 1 km² too (`basin_id=...` selects only the basins of
the basin table). Together the groups give every row of a coarser-threshold table. `reaches_at` returns the 15 columns of `reach_<T>` of the coarser-threshold package in order:
the first 13 columns of `reach_1km2` (without `break_threshold_km2`), then `downstream_end_reach_seq_id_1km2` and
`stem_reach_count_1km2`. Reach and path ids are sequential globally at T, as in the coarser-threshold tables. Crosswalk values and the raster
from `window_at` use these threshold ids. Threshold labels use `%.9gkm2` with `.` replaced by `p`, for example `11p2km2`.
`channel_pixel_count` counts this threshold's own channel pixels; `upstream_channel_pixel_count` still counts base channel pixels;
`stem_reach_count_1km2` counts the consecutive base reaches along this reach's stem.
The unit-catchment area sums the base unit catchments that map to this reach. Lengths sum pixel steps using the release earth model.

For a coarser-threshold package, `Release(folder, coarser_thresholds='/path/to/coarser_thresholds')` reads its `reach_<T>` tables
when available. Both folders must have v4 metadata with the same numbering version and the same 64-character numbering fingerprint;
otherwise `Release` raises `ValueError`.
Deriving global threshold ids streams preceding base reach rows to count the retained reaches and paths, and caches those counts.


## Upstream statistics and unit catchments

`n.totals(values)` sums a raster upstream of every channel pixel in one pass over the pixels.
`n.means(values, pixel_area)` returns `(area_weighted_mean, catchment_area)` in two passes. Entry i belongs to pixel `n.first + i`.
For a released raster, use `upstream_totals(raster, upstream_channel_pixel_count, values, first=1)` and
`upstream_means(raster, upstream_channel_pixel_count, values, pixel_area, first=1)`.
Only signed-raster 0 and nodata are skipped; replace the values raster's own nodata before summing.

```sh
python examples/upstream_statistics.py /path/to/release 10670
```

This example compares WGS84 pixel-area sums with MERIT Hydro's upstream area at the reach downstream ends.
`pixel_areas_km2(lat_top, dlat, dlon, nrow)` supplies WGS84 pixel areas, one per row; `upstream_pixels(d8)` supplies pixel-count
upstream area when no area raster is available.

`n.unit_catchments(T)` returns `reach_seq_id`, `first_1`, `count_1`, `first_2`, `count_2`.
Select `first_i <= abs(v) < first_i + count_i` for either range. The first covers the reach's own channel pixels and their hillslopes;
the second covers branches removed by T that follow the final kept branch's upstream subtree. At the base threshold `count_2` is zero.


## Crosswalks: gauges, MERIT-Basins, catchments and coarser views

These modules follow the rules of the release's crosswalk packages; each module's docstring states its rule.

| Module | Functions | Package it reproduces |
|---|---|---|
| `stations` | `read_station_list(path, 'burek2023' or 'plain')`, `place_stations(rel, stations, upa_tif)` | GRDC stations: the 19 columns, with stations upstream of a gauge one run of `station_seq_id` |
| `merit_basins` | `read_merit_basins(zip_or_shp)`, `ReleaseReaches(rel, T)`, `place_reaches(rel, lookup, reaches, upa_tif)` | MERIT-Basins reaches at T, read pixel by pixel with no snapping |
| `catchments` | `nest`, `first_downstream`, `label_window`, `unit_windows`, `upstream_window`, `pair_gauges_with_nodes` | the catchments of gauges, SWORD reaches and SWORD nodes, on any non-empty set of channel pixels or any window holding at least one number |
| `view` | `fold_array(fine, factor, object_count)`, `fold_file(...)` | the coarser view of a number raster (6 × 6 pixels to 1/200°) |

```python
from flownest import Release, stations, merit_basins
rel = Release('/path/to/flownest_merit90m_global_DATE', '/path/to/flownest_merit90m_coarser_thresholds_DATE')
gauges = stations.place_stations(rel, stations.read_station_list('stationquality.csv'), '/path/to/upa.tif')
reaches = merit_basins.read_merit_basins('/path/to/pfaf_7_MERIT_Hydro_v07_Basins_v01.zip')
table = merit_basins.place_reaches(rel, merit_basins.ReleaseReaches(rel, 25), reaches, '/path/to/upa.tif')
```

The SWORD v17c crosswalk is the subpackage `flownest.sword`: each SWORD subnetwork placed in its FlowNest basin, each
reach on a FlowNest river path inside that basin, each node on a channel pixel of that path where a match is found
(objects without one keep the SWORD row with FlowNest ids 0), with `match_confidence`, the issue
columns and the three tables (subnetwork, reach, node) of the release. SWORD and MERIT are not edited. It needs the extra
`crosswalks` (pyogrio, shapely), the SWORD v17c GeoPackages and the MERIT Hydro upa and wth rasters:

```python
from flownest.sword import pipeline

if __name__ == '__main__':      # needed on macOS and Windows, where worker processes start the script anew
    pipeline.run('/path/to/flownest_merit90m_global_DATE', '/path/to/sword_v17c_gpkg', '/path/to/upa.tif', '/path/to/wth.tif',
                 '/path/to/work')                           # all subnetworks, one after another
    pipeline.run(..., systems=[3000026, 7000001])           # a few subnetworks
    pipeline.run(..., processes=1)                          # no worker processes
```

`catchments` then gives the catchments of its reaches and nodes.


## Comparing numberings

`n.fingerprint()` hashes the grid, pixel positions, downstream links, upstream area and signed raster.
`offset_table(old, new)` returns `(first, count, offset)` runs; `offset=None` means the pixels left the network.
`translate(table, p)` and `read_through(old, new, p)` translate an id or read the new signed-raster value at its original location.
A corrected grid is a new version with its own fingerprint.


## Tests

```sh
pip install -e ".[test]"
pytest
```

Tests use made-up grids to check code behavior. The release tests require `FLOWNEST_TEST_RELEASE`, `FLOWNEST_TEST_D8` and
`FLOWNEST_TEST_UPA` (`FLOWNEST_TEST_COARSER_THRESHOLDS` for the coarser-threshold package) and go through every basin, so they suit the small
synthetic release. `Release.number_basin` rebuilds a basin's numbering from its original D8 and upstream-area rasters;
`reach_windows` rebuilds the unit and upstream catchment windows and unit-catchment areas.


## Checks against the release

The release `20261113_195421` was produced by the authors' separate implementation in C, which is not distributed; comments
that mention "C" refer to it. The scripts of `release_checks/` compare this package with that release (each script's docstring
gives its arguments):

| Script | What is compared | Result |
|---|---|---|
| `check_numbering_against_release.py` | 98 basins numbered again from D8 and upstream area (Dungeness, Nisqually, Siletz, 30 each of 1–100, 100–1,000 and 1,000–10,000 km², 5 of 10,000–100,000 km²): pixel rasters, reaches at 1, 5 and 25 km² and their windows and areas, reaches derived from the base tables (2,316,149 channel pixels, 146,563 reaches, 30,184 derived reaches) | identical |
| `check_grdc_stations_against_release.py` | all 19 columns of the GRDC crosswalk, 10,241 stations | identical |
| `check_merit_basins_against_release.py` | all 18 columns of the MERIT-Basins crosswalk at 25 km², 2,938,143 reaches | identical |
| `check_catchments_of_gauges_and_sword_against_release.py` | nesting of 10,241 gauges, 240,111 SWORD reaches and 10,729,230 nodes; the 10,241 gauge–node pairs | identical |
| `check_catchment_rasters_on_demo_window.py` | label rasters of the catchments package, unit and upstream windows and node unit pixel counts in the demo window | identical |
| `check_sword_crosswalk_against_release_tables.py` | each placed SWORD node and reach against the pixel and basin tables (row, col, reach, path, area, basin) | 0 differences |
| `check_merit_basins_river_systems_on_fullbasin.py` | MERIT-Basins river systems against FullBasin basins | see below |
| `check_sword_python_against_release.py` | all columns of the three SWORD tables written by `flownest.sword` for some subnetworks; global seq ids by order, upstream counts not above the release's | identical on 34 subnetworks (14,834 reaches, 695,618 nodes) and on the 5 subnetworks of the Amazon run (24,456 reaches, 1,315,846 nodes) |

MERIT-Basins against FullBasin, basin first: all 62,216 MERIT-Basins river systems lie each in one FullBasin basin, and each
outlet reach begins on a pixel next to that basin's outlet pixel. Summed over the river systems of a basin, MERIT-Basins' upstream
area differs from the basin's area by a median of 0.13% in the 48,037 basins draining to the sea (99.0% within 1%) and of 0.16%
in the 8,359 endorheic ones (67.2% within 1%); in the 30 largest basins by at most 0.26%.


## Limits

- Signed int32 raster ids support at most 2,147,483,647 channel pixels per numbering; number continents apart or raise the threshold.
- `number_grid` holds the grid in memory; `number_basin` holds a basin's window, suitable up to about 10⁵ km².
- Closed flow-direction loops and NaN upstream area on a valid pixel are refused. Endorheic sinks are basin outlets.


## Citing

Cite the FlowNest paper and this repository with the release used, and FullNest by version:

Jiang, L., Wu, H., Chen, W., Yamazaki, D., & Huang, Z. (submitted). FlowNest: A unified coding system for position on global
river networks. Water Resources Research.

Jiang, L., Wu, H., Chen, W., Yamazaki, D., & Huang, Z. (2026). FullNest (MERIT Hydro, 90 m) (Version 20261113_195421)
[Dataset]. Zenodo. https://doi.org/10.5281/zenodo.24393217

Jiang, L., Wu, H., Chen, W., Yamazaki, D., & Huang, Z. (2026). FullNest coarser thresholds and crosswalks (MERIT Hydro, 90 m)
(Version 20261113_195421) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.24393244


## Acknowledgements

**MERIT Hydro** (Yamazaki et al., 2019; [10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873)) is the grid
FullNest numbers, and gives the 4 × 4 example. Its basin identifiers and group codes are those of **FullBasin** v1.1
([10.5281/zenodo.22895352](https://doi.org/10.5281/zenodo.22895352)). The crosswalks place the **GRDC** gauges as located on
MERIT Hydro by Burek and Smilovic ([10.5281/zenodo.6906577](https://doi.org/10.5281/zenodo.6906577)), **SWORD** v17c
([10.5281/zenodo.23870877](https://doi.org/10.5281/zenodo.23870877)) and **MERIT-Basins** (Lin et al., 2019;
[10.1029/2019WR025287](https://doi.org/10.1029/2019WR025287)). The released files keep the terms given in the README of each package.
