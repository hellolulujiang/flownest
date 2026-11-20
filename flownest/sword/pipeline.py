"""Run the SWORD v17c crosswalk end to end.

    from flownest.sword import pipeline
    pipeline.run(release_dir, sword_gpkg_dir, upa_tif, wth_tif, work_dir)              # all systems
    pipeline.run(..., systems=[3000026, 7000001])                                      # a few systems (trial)

Steps, each writing into work_dir:
1. basins:    every SWORD reach sampled on the 90 m basin grid; each system (subnetwork) given its FlowNest basin(s).
2. reaches:   each reach matched to a FlowNest path inside its own basin; flow direction checked (one system at a time).
3. nodes:     each SWORD node placed on a channel pixel of the matched path (one system at a time).
4. merge:     the per-system tables joined into one reach table and one node table; seq ids numbered.
5. finish:    reaches rescued by their placed nodes; issue columns added.
6. export:    the three crosswalk tables (subnetwork, reach, node) written with their release column names.
Systems are run continent by continent (each continent's SWORD file read once), largest system first."""
import os
import traceback

import pandas as pd

from . import inputs as IN
from . import basins, direction, export, finish, merge, nodes, reaches


def systems_to_run(only):
    """The systems with a FlowNest basin, ordered by continent and then by number of reaches (most first)."""
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=[''])
    S = S[S.basin_id.notna()]
    if only is not None:
        missing = sorted(set(only) - set(S.subnetwork_id))
        if missing:
            raise ValueError(f'systems without a FlowNest basin or not in SWORD: {missing}')
        S = S[S.subnetwork_id.isin(only)]
    return S


def run_systems(S, line_dir, point_dir):
    """Match reaches, check direction and place nodes, system by system, into <dir>/systems/; record each system in
    <dir>/progress.csv (status ok or fail). Returns the systems that failed with their tracebacks."""
    lsys, psys = line_dir + 'systems/', point_dir + 'systems/'
    os.makedirs(lsys, exist_ok=True)
    os.makedirs(psys, exist_ok=True)
    prog, failed = [], []
    for cont in sorted(S.continent.unique()):
        reaches.use_continent_cache(cont)
        nodes.use_continent_nodes(cont)
        n = reaches.CACHE['R'].subnetwork_id.value_counts()
        sub = S[S.continent == cont].copy()
        sub['n'] = sub.subnetwork_id.map(n).fillna(0).astype(int)
        for sid in sub.sort_values(['n', 'subnetwork_id'], ascending=[False, True]).subnetwork_id:
            sid, status, k = int(sid), 'ok', 0
            try:
                reaches.main(sid, lsys)
                direction.main(sid, lsys)
                nodes.main(sid, lsys, psys)
                k = len(pd.read_parquet(psys + f'sword_node_to_flownest_pixel_{sid}.parquet', columns=['node_id']))
            except Exception:
                status = 'fail'
                failed.append((sid, traceback.format_exc()))
            prog.append((sid, cont, status, k))
    P = pd.DataFrame(prog, columns=['subnetwork_id', 'continent', 'status', 'nodes'])
    P.to_csv(line_dir + 'progress.csv', index=False)
    P.to_csv(point_dir + 'progress.csv', index=False)
    return failed


def run(release_dir, sword_gpkg_dir, upa_tif, wth_tif, work_dir, systems=None, processes=6):
    """All six steps. With `systems`, only those systems are matched and the global checks of step 4 are skipped."""
    IN.configure(sword_gpkg_dir, release_dir, upa_tif, wth_tif, work_dir)
    basins.run(processes)
    S = systems_to_run(systems)
    line6, point6 = IN.WORK + 'line6/', IN.WORK + 'point6/'
    failed = run_systems(S, line6, point6)
    if failed:
        raise RuntimeError(f'{len(failed)} systems failed; first:\n{failed[0][1]}')
    line_out, point_out = IN.WORK + 'reaches/', IN.WORK + 'nodes/'
    merge.merge_reaches(line6, line_out)
    merge.merge_nodes(point6, point_out, all_systems=systems is None)
    finish.main(line_out, point_out)
    return export.main(IN.FACE, line_out, point_out, IN.FN, IN.WORK + 'release/')
