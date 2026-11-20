"""Step 6: the three SWORD crosswalk tables (subnetwork, reach, node) with their release column names and
match_confidence, written with README.md, metadata.json and SHA256SUMS.txt into a new time-stamped folder."""
import hashlib
import json
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re
import time

from . import confidence as CF
from . import inputs as IN
from .reaches import stored


PREFIX = 'flownest_merit90m_crosswalk_sword_v17c_'
CODE_FIX = [('along_flownest_flow', 'along_flow'), ('against_flownest_flow', 'against_flow'),
            ('reversed_vs_flownest_flow', 'reversed_vs_flow'), ('another_flownest_channel_closer', 'another_channel_closer'),
            ('no_flownest_land', 'no_land')]
SENTENCE_COLS = ('cause', 'action', 'why_not_fixed')
ISSUE_COLS = ('issue',) + SENTENCE_COLS
SENT_FIX = [(' (see the neighbour pair table)', ''), (', see the seam table', ''),
            ("lie on a path of another basin (the line step searches only in the reach's own basin)",
             "lie on a path outside its own basin (own_basin_id), where reach matching does not search"),
            ('there is no basin to search in, so any path would be a guess',
             'it was searched in the subnetwork basin (subnetwork_basin_id) without a match, and most of its line lies outside '
             'every FlowNest basin, so it is not matched from its continued nodes either'),
            ('each reach keeps its own basin in the reach table', 'each reach keeps its own basin in the reach table (own_basin_id)'),
            ('allowed by rule 2A', 'several nodes may share one pixel: allowed and flagged'),
            ('lakeflag 3', 'sword_lakeflag 3'), ('(lakeflag 1)', '(sword_lakeflag 1)'), ('(type 6)', '(sword_type 6)'),
            ('face step:', 'subnetwork matching:'),
            ('line stages 1 to 3', 'reach matching stages 1 to 3'), ('in the point step', 'in node placement'),
            ('as in the line step', 'as in reach matching'), ('matched in the line step', 'matched in reach matching'),
            ('would contradict the line step', 'would contradict reach matching'),
            ('the line step searches', 'reach matching searches'), ('the line step matches', 'reach matching matches'),
            ('in the line step', 'in reach matching'), ('line step:', 'reach matching:')]
OLD_WORDS = r'face step|line step|point step|line stages|(?<!sword_)lakeflag|(?<!sword_)type 6'
UPA_DEC = ('upa_decreases_downstream',
           'upstream area is smaller than at the previous node upstream along SWORD (the node lies against the FlowNest flow, '
           'or SWORD and MERIT channels differ here)', 'kept', 'SWORD and MERIT are not edited')
SHARE_X = ('shares_pixel', 'a node of another SWORD system lies on this pixel too (the systems meet or overlap here)', 'kept',
           'several nodes may share one pixel: allowed and flagged')
OFF_PATH = ('not_on_matched_path',
            'continued from the neighbour reach onto a path other than the path(s) matched for this reach (stage 4 needs only '
            '80% of the nodes of the reach on the matched path(s))', 'kept', 'the node follows the FlowNest flow from the neighbour reach')
FAR_REACH = ('most_nodes_far_from_pixel',
             'more than half of the placed nodes are farther than half the width plus 180 m from their pixel: the MERIT '
             'channel runs away from the SWORD line here (a shifted MERIT channel, a SWORD side channel, or a wrong match)',
             'kept and flagged',
             'SWORD and MERIT are not edited (for a strict subset, leave out reaches with this flag)')
STATUS = ('no_land', 'one_basin', 'main_basin_plus_small_bits', 'also_in_other_large_basin')
LINK_KIND = ('cut_inside_reach', 'estuary_delta', 'inland_canal', 'inland_lake', 'inland_bifurcation', 'inland_divide')
RELEASE_FILES_READ = ['flownest_merit90m_pixel_seq_id_1km2.tif', 'flownest_merit90m_basin.parquet', 'flownest_merit90m_reach_1km2.parquet',
                      'flownest_merit90m_pixel_1km2.parquet', 'flownest_merit90m_reach_seq_id_1km2.tif']
# released subnetwork columns that every row fills (checked before writing)
SUBNETWORK_NOT_NULL = ('sword_subnetwork_id', 'sword_continent', 'sword_reach_count', 'sword_river_length_km', 'basin_id',
                       'second_basin_id', 'other_large_basin_count', 'other_large_basin_id', 'status', 'link_kind')


def write_parquet(df, path, not_null=()):
    """df to parquet, every column nullable (as C writes it); the columns in not_null must have no missing value."""
    assert not df[list(not_null)].isna().any().any(), path
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


STEP_ORDER = ['3x3', '5x5', '7x7', 'beyond_7x7', 'between_neighbors', 'against_flow',
              'land_part_cover', 'partly_on_neighbor_path', 'within_water_surface', 'short_path_into_neighbor',
              'area_guided_switch', 'area_guided_rescue', 'continued_from_neighbor']


def add_issue(df, mask, entry):
    """Append one entry (issue, cause, action, why_not_fixed) to the four issue columns of the rows in mask ('; ' between entries)."""
    for c, v in zip(ISSUE_COLS, entry):
        s = df.loc[mask, c]
        df.loc[mask, c] = np.where(s.eq(''), v, s + '; ' + v)


def drop_issue(df, mask, code):
    """Remove the entry whose issue is code from the rows in mask (at the same position in all four issue columns)."""
    idx = np.flatnonzero(np.asarray(mask))
    vals = {c: df[c].values[idx] for c in ISSUE_COLS}
    out = {c: [] for c in ISSUE_COLS}
    for k in range(len(idx)):
        parts = {c: (vals[c][k].split('; ') if vals[c][k] else []) for c in ISSUE_COLS}
        keep = [j for j, x in enumerate(parts['issue']) if x != code]
        assert len(keep) < len(parts['issue'])
        for c in ISSUE_COLS:
            out[c].append('; '.join(parts[c][j] for j in keep))
    for c in ISSUE_COLS:
        col = df[c].values.copy()
        col[idx] = out[c]
        df[c] = col


def check_issue_counts(df, t):
    """Each row has as many entries in cause, action and why_not_fixed as in issue."""
    k = [(df[c].str.count('; ') + 1).where(df[c].ne(''), 0) for c in ISSUE_COLS]
    for c, x in zip(ISSUE_COLS[1:], k[1:]):
        assert (x.values == k[0].values).all(), (t, c, 'entry count differs from issue')


def reliability(R):
    """By matching step: reaches, share with most nodes far from their pixel, and share (of reaches with facc) whose SWORD
    facc_total is within 0.5-2 times the MERIT upstream area."""
    rows = []
    for s in STEP_ORDER:
        g = R[R.matched_at_step == s]
        if not len(g):
            continue
        r = np.round(stored(g.area_ratio.dropna().values), 3)   # the written ratio, checked on 3 decimals as area_check is
        rows.append({'matched_at_step': s, 'reaches': int(len(g)),
                     'share_most_nodes_far': round(float(g.issue.str.contains(FAR_REACH[0], regex=False).mean()), 4),
                     'reaches_with_facc_ratio': int(len(r)),
                     'share_facc_ratio_0p5_to_2': round(float(((r >= 0.5) & (r <= 2)).mean()), 4) if len(r) else None,
                     'area_check': {k: int(v) for k, v in g.area_check.value_counts().items()}})
    return rows
# Ratios, shares and covers are kept as computed through every step and every rule; they are written with 6 significant
# digits, and match_confidence reads the cover as written (as C does).
STORED_6 = {'subnetwork': ['sea_fraction', 'river_length_share', 'second_river_length_share', 'other_large_river_length_share'],
            'reach': ['sea_fraction', 'cover', 'runner_up_cover', 'pair_unordered_cover', 'area_ratio_before_switch', 'land_cover',
                      'merit_water_fraction', 'upstream_area_to_sword_facc_ratio', 'best_cover_3x3', 'best_cover_5x5', 'best_cover_7x7',
                      'best_cover_stage2_window'],
            'node': ['upstream_area_to_sword_facc_ratio']}
REACH_SAME = ['sword_reach_seq_id', 'upstream_sword_reach_count', 'flownest_downstream_pixel_seq_id', 'subnetwork_id',
              'nodes', 'nodes_placed', 'upstream_end_pixel_seq_id', 'downstream_end_pixel_seq_id']


def flownest_pixel_area(pix_file, seq):
    """upstream_area_km2 (float32, not rounded) of these pixels in the FlowNest pixel table, where row i (from 0) is
    pixel_seq_id i + 1."""
    need = np.unique(seq)
    f = pq.ParquetFile(pix_file)
    val = np.zeros(len(need), np.float32)
    start = lo = 0
    for k in range(f.metadata.num_row_groups):
        n = f.metadata.row_group(k).num_rows
        hi = int(np.searchsorted(need, start + n, side='right'))
        if hi > lo:
            t = f.read_row_group(k, columns=['pixel_seq_id', 'upstream_area_km2'])
            idx = need[lo:hi] - start - 1
            assert (t.column(0).to_numpy()[idx] == need[lo:hi]).all()
            val[lo:hi] = t.column(1).to_numpy()[idx]
        lo, start = hi, start + n
    assert lo == len(need)
    return val[np.searchsorted(need, seq)]


def basin_of_pixels(B, seq):
    """The FlowNest basin of each pixel: the channel pixels of a basin are the range [outlet_pixel_seq_id,
    outlet_pixel_seq_id + channel_pixel_count); seq 0, or a seq in no range, gives 0."""
    B = B.sort_values('outlet_pixel_seq_id')
    st = B.outlet_pixel_seq_id.values.astype(np.int64)
    en = st + B.channel_pixel_count.values.astype(np.int64)
    assert (st[1:] >= en[:-1]).all(), 'FlowNest basin ranges overlap'
    seq = np.asarray(seq, np.int64)
    i = np.clip(np.searchsorted(st, seq, side='right') - 1, 0, None)
    ok = (seq > 0) & (seq >= st[i]) & (seq < en[i])
    return np.where(ok, B.basin_id.values[i], 0).astype(np.int64)


def sha256(p):
    """SHA-256 of a file, read in 4 MiB blocks."""
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def on_unique(s, f):
    """Apply f to the distinct values only and map back (11 million node rows, a few hundred distinct sentences)."""
    u = pd.Series(s.dropna().unique(), dtype=object)
    m = dict(zip(u, f(u)))
    return s.map(m).where(s.notna(), s)


def fix_codes(df, skip=('sword_river_name',)):
    """Shorten the internal codes in all text columns (CODE_FIX), except river names."""
    def f(u):
        for a, b in CODE_FIX:
            u = u.str.replace(a, b, regex=False)
        return u
    for c in df.columns:
        if df[c].dtype == object and c not in skip:
            df[c] = on_unique(df[c], f)
    return df


def name_pat(a):
    """Regular expression for column name a as a whole word."""
    return r'(?<![A-Za-z0-9_])' + re.escape(a) + r'(?![A-Za-z0-9_])'


def fix_sentences(df, names, used):
    """Replace the old column names that sentences mention by the release names; names: old -> new (only names with an
    underscore, so plain words are left alone). used counts the rows changed."""
    for c in SENTENCE_COLS:
        vc = df[c].value_counts()
        u = pd.Series(vc.index, dtype=object)
        new = u.copy()
        for a, b in names.items():
            hit = new.str.contains(name_pat(a), regex=True, na=False)
            if hit.any():
                used[a] = used.get(a, 0) + int(vc.values[hit.values].sum())
                new = new.str.replace(name_pat(a), b, regex=True)
        df[c] = df[c].map(dict(zip(u, new))).where(df[c].notna(), df[c])
    return df


def apply_map(df, M, table):
    """Keep and rename the columns of one table as the rename map says (every column must be in the map)."""
    m = M[M.table == table]
    want = list(m.internal_name)
    have = list(df.columns)
    assert set(have) == set(want), (table, 'missing in map', sorted(set(have) - set(want)), 'missing in data', sorted(set(want) - set(have)))
    keep = m[m.release_name != '']
    out = df[list(keep.internal_name)].rename(columns=dict(zip(keep.internal_name, keep.release_name)))
    assert not any('flownest' in c.lower() for c in out.columns), table
    return out


def order_by_seq(df, seq, ident):
    """Rows ordered by seq (1..n, no gaps), the rows with seq 0 last in ident order. Returns (table, n)."""
    z = (df[seq].values == 0)
    o = np.lexsort((df[ident].values, df[seq].values, z))
    df = df.iloc[o].reset_index(drop=True)
    n = int((df[seq] > 0).sum())
    assert (df[seq].values[:n] == np.arange(1, n + 1)).all() and (df[seq].values[n:] == 0).all(), seq
    return df, n


WIDTH_CLASSES = (('lt30m', '< 30 m', 30.0), ('30to100m', '30-100 m', 100.0), ('100to500m', '100-500 m', 500.0),
                 ('500to1000m', '500-1000 m', 1000.0), ('ge1000m', '>= 1000 m', np.inf))


def confidence_counts(R, N):
    """Rows of each match_confidence tier: reaches, nodes, river reaches (sword_type 1, sword_lakeflag 0) by SWORD width
    class (widths of 0 or -1 fall in < 30 m) and reaches by SWORD region (the first digit of sword_reach_id)."""
    def tiers(s):
        n = s.value_counts()
        return {k: int(n.get(k, 0)) for k in CF.TIERS}
    river = R[(R.sword_type == 1) & (R.sword_lakeflag == 0)]
    w = river.sword_width_m.values
    cls = np.select([w < x[2] for x in WIDTH_CLASSES[:-1]], range(len(WIDTH_CLASSES) - 1), len(WIDTH_CLASSES) - 1)
    region = R.sword_reach_id.astype(str).str[0].astype(int)
    return {'reach': tiers(R.match_confidence), 'node': tiers(N.match_confidence),
            'river_reach_by_sword_width': {k: tiers(river.match_confidence[cls == i]) for i, (k, _, _) in enumerate(WIDTH_CLASSES)},
            'reach_by_sword_region': {str(r): tiers(R.match_confidence[region == r]) for r in sorted(region.unique())},
            'river_reach': 'sword_type 1 and sword_lakeflag 0', 'region': 'the first digit of sword_reach_id'}


def codes_of(table, tier):
    """The codes of one tier, in table order; `upstream_area_disagrees` (tier set by `area_note`) is listed as medium."""
    return ', '.join(f'`{c}`' for c, t in table.items() if t == tier or (t is None and tier == CF.MEDIUM))


def confidence_section(mc):
    """The README section on match_confidence, ending with the table of every code of confidence_reason."""
    pct = lambda a, b: f'{round(100 * a / b, 1) if b else 0.0:.1f}%'
    t = f"""## match_confidence: can a row be used, and why

Every reach row and every node row has `match_confidence` (`high`, `medium`, `low` or `not_found`) and `confidence_reason`,
the codes that give the tier, joined by `; `. Every tier has a reason, `high` included. The reach table has exactly one row
per SWORD reach and the node table one row per SWORD node, so `sword_reach_id` and `sword_node_id` are unique join keys.
The codes form a closed list (the table at the end of this section); the program stops on a code that is not in it, so
no row is `high` with a code nobody classified.

Reaches:

- `high`: all four checks pass, and `confidence_reason` names them: `cover_at_least_80pct` (at least 80% of the reach on its
  path), `upstream_area_agrees` (the MERIT upstream area within 2 times of the SWORD facc_total), `most_nodes_near_pixel` (at
  least half of the placed nodes within half the width plus 180 m of their pixel) and `along_flow` (the placed nodes run along
  the FlowNest flow).
- `medium`: matched, no low code, and one of {codes_of(CF.REACH_CODES, CF.MEDIUM)}.
  `upstream_area_disagrees` counts as medium only with an `area_note` (a likely reason; reason `upstream_area_disagrees_explained`).
  A matched reach without such a code that misses one of the four checks is medium too, the missing check named:
  `cover_below_80pct`, `upstream_area_not_checked` (no SWORD facc_total, or the area chose the path), `nodes_not_checked`,
  `reversed_vs_flow`.
- `low`: matched, with one of {codes_of(CF.REACH_CODES, CF.LOW)},
  or `upstream_area_disagrees` without an `area_note` (reason `upstream_area_disagrees_unexplained`). Check these before use.
- `not_found`: no FlowNest path, and one reason: `mostly_sea`, `not_in_any_basin`, `no_matched_neighbor`, `candidate_not_connected`,
  `only_short_path_nearby`, `partly_on_path`, `far_from_path` or `no_land` (meanings in the table below).

Low codes are named before medium ones. A node takes the worse of its own tier and its reach's tier (a placed node of a
reach not found is at most `low`). Its own codes: `not_found` {codes_of(CF.NODE_CODES, CF.NOT_FOUND)}; `low` {codes_of(CF.NODE_CODES, CF.LOW)}; `medium` {codes_of(CF.NODE_CODES, CF.MEDIUM)}. Its reason lists its own codes, `node_near_pixel` when it is
`high`, and then `reach_<tier>` (the reach's own reason is in the reach table).

SWORD derives its facc from MERIT Hydro, so `upstream_area_agrees` shows that the SWORD reach and the FlowNest path follow the
same MERIT channel; it is not an independent measure of the area.

Which rows to use:

| use | rows |
|---|---|
| discharge and anything that needs the upstream area (SWOT discharge, gauges, assimilation) | `high`; `medium` after a look at the reaches with `upstream_area_disagrees_explained` (the area note says why the two areas differ) |
| routing along the network (passing a value downstream from reach to reach) | `high`; `medium` without `reversed_vs_flow`, `neighbor_pair_against_flow` or `two_paths_against_flow`; nodes without a `seam_` code |
| water surface elevation, slope, width along the river | `high` and `medium` |
| maps and counts | `high`, `medium` and `low` |
| `not_found` | a reach with no FlowNest path matched (some of its nodes may still be placed), or a node not placed; `confidence_reason` says why |

| `match_confidence` | reaches | share | nodes | share |
|---|---:|---:|---:|---:|
"""
    nr, nn = sum(mc['reach'].values()), sum(mc['node'].values())
    t += ''.join(f"| `{k}` | {mc['reach'][k]:,} | {pct(mc['reach'][k], nr)} | {mc['node'][k]:,} | {pct(mc['node'][k], nn)} |\n"
                 for k in CF.TIERS)

    def rows(d, label):
        out = ''
        for key, row in d.items():
            n = sum(row.values())
            if label is None and n == 0:
                continue
            out += (f"| {label[key] if label else key} | {n:,} | " + ' | '.join(f'{row[k]:,}' for k in CF.TIERS)
                    + f" | {pct(n - row['not_found'], n)} |\n")
        return out
    head = '| {} | reaches | high | medium | low | not_found | matched |\n|---|---:|---:|---:|---:|---:|---:|\n'
    t += ("\nRiver reaches (`sword_type` 1, `sword_lakeflag` 0) by SWORD width (`sword_width_m`; widths of 0 or -1 count as < 30 m):\n\n"
          + head.format('width') + rows(mc['river_reach_by_sword_width'], {k: s for k, s, _ in WIDTH_CLASSES})
          + "\nAll reaches by SWORD region (the first digit of `sword_reach_id`):\n\n"
          + head.format('region') + rows(mc['reach_by_sword_region'], None)
          + "\n### Codes of confidence_reason\n\nEvery code that can appear in `confidence_reason`, the table it appears in and the "
            "tier it gives (`-`: `reach_<tier>` repeats\non a node the tier of its reach).\n\n" + CF.meanings_table() + '\n')
    return t


def write_readme(O, meta, cols, release_name):
    """README.md of the release folder O, for the FlowNest release release_name (the same text as the C program writes)."""
    c = meta['counts']
    rbm = meta['reach_basin_id']
    t = f"""# FlowNest on MERIT Hydro 90 m: SWORD v17c crosswalk

This crosswalk links every SWORD v17c river system (subnetwork), reach and node to the FlowNest release
`{release_name}` at its base threshold (1 km2), in three steps:

- subnetwork matching, to a FlowNest basin (`basin_id`): {c['sword_subnetworks_with_basin']:,} of {c['sword_subnetworks']:,} systems;
- reach matching, to a FlowNest river path (`path_seq_id_1km2`, plus `path2_seq_id_1km2` when two paths are needed): {c['sword_reaches_matched_to_path']:,} of {c['sword_reaches']:,} reaches;
- node placement, on a FlowNest channel pixel (`pixel_seq_id`): {c['sword_nodes_placed']:,} of {c['sword_nodes']:,} nodes.

SWORD and MERIT are not edited. A match should be the same water: the same river corridor, the same order along the
flow, and the same upstream area. SWORD v17c takes its drainage areas from MERIT-Basins (derived from MERIT Hydro), so the MERIT
upstream area on the matched path should agree with them. The area used is the SWORD node facc_total (`sword_facc_total_km2`) of the same SWORD v17c build
(0.0.15) as every other SWORD field: the area of the whole river, all channels at a split counted together. Reaches
are matched by river lines and topology first; the upstream area then checks every match (`area_check`) and, on
main-channel reaches, switches or adds a match where exactly one nearby long path carries the same water
(`matched_at_step` `area_guided_switch` or `area_guided_rescue`). Upstream areas are MERIT upa.

`matched_at_step` names the step that matched each reach, in the words of the FlowNest paper (Appendix B).
`match_confidence` and `confidence_reason` sum up, per reach and per node, whether the row can be used and why.

## Files

- `{PREFIX}subnetwork`, `{PREFIX}reach`: one row per SWORD subnetwork or reach, each as Parquet and CSV.
- `{PREFIX}node`: one row per SWORD node, as Parquet only.
- `metadata.json`: counts, file sizes and SHA-256, the FlowNest numbering these numbers belong to, and the meaning of every column.
- `SHA256SUMS.txt`: the SHA-256 of every other file (`shasum -a 256 -c SHA256SUMS.txt`).

## Names

- Columns starting with `sword_` hold SWORD values or SWORD numbers (`sword_reach_id`, `sword_node_id`, and the new global
  `sword_reach_seq_id`, `sword_node_seq_id`). FlowNest values keep their FlowNest names (`pixel_seq_id`, `reach_seq_id_1km2`,
  `path_seq_id_1km2`, `basin_id`, `upstream_area_km2`, `row`, `col`). The other columns describe the match (step, cover,
  upstream area check, distances, counts, confidence).
- `_seq_id` numbers belong to this FlowNest numbering. `basin_id` is the FullBasin global basin ID (MERIT Hydro basins numbered
  by outlet upstream area, largest first, 1 the Amazon), the same number as in FullBasin and in the FlowNest tables; it is not
  part of the FlowNest numbering.
- In id and number columns (`..._id`, `..._seq_id`), `0` means none (not matched or not placed); real ids start at 1.
  `row` and `col` are MERIT 90 m pixel indices counted from 0 (`col` 0 is a real column), empty for a node not placed.
  Value columns (areas, ratios, covers, distances) are empty when they do not apply. Ratios, shares,
  covers and fractions are stored with 6 significant digits; the checks judge an area ratio
  on its value rounded to 3 decimals, a subnetwork share or sea fraction on 4 decimals, a reach's cover and
  sea fraction on the value as computed, so a value stored right at a limit (0.5, 0.8, 2) can fall on either
  side of it. SWORD values (widths, facc_total) are SWORD's own, unchanged. Some SWORD
  codes use 0 as a value: `sword_main_side` 0 is the main channel, `sword_lakeflag` 0 is a river.
- Empty values: in Parquet an empty number is null and an empty text is an empty string (in the subnetwork table
  `sword_river_name` is null); in the CSV both are empty fields. The reach table keeps SWORD's own `NODATA` in
  `sword_river_name`; the subnetwork name is null when no reach of the system has a name.
- In the CSV, `sword_continent` `NA` means North America; read it with `keep_default_na=False`.

## Global numbers

- `sword_node_seq_id` numbers the placed nodes in FlowNest pixel order (by `pixel_seq_id`; nodes on one pixel by smaller
  SWORD `dist_out`, then `sword_node_id`). `sword_reach_seq_id` numbers the reaches with a placed node in the same way, by
  `most_downstream_pixel_seq_id`.
- `upstream_sword_node_count` counts the placed nodes upstream along FlowNest flow, the node itself included. Those nodes are
  `[sword_node_seq_id, sword_node_seq_id + upstream_sword_node_count)`. The node table is stored in this order, so they are
  consecutive rows. The reach table works the same way with `sword_reach_seq_id` and `upstream_sword_reach_count`.
- Upstream means along FlowNest (MERIT) flow, not along SWORD topology. The numbers hold only for `numbering_sha256`
  `{meta['numbering_sha256']}`.
- The reach `basin_id` (0 when no node is placed) is the FlowNest basin holding `most_downstream_pixel_seq_id`, the smallest
  `pixel_seq_id` among the placed nodes of the reach. A FlowNest basin is one range of pixel numbers, so the reaches of one
  basin are consecutive rows, and each range `[sword_reach_seq_id, sword_reach_seq_id + upstream_sword_reach_count)` lies in
  one basin.
  It need not be the basin the reach was searched in (`own_basin_id`; the `basin_id` of its subnetwork when `own_basin_id` is 0):
  {rbm['numbered_reaches_basin_not_search_basin']:,} numbered reaches without a path, most of them tidal (`sword_lakeflag` 3),
  have the placed node with their smallest `pixel_seq_id`, continued from a neighbour reach, in another basin.
  {rbm['numbered_reaches_with_nodes_in_several_basins']:,} numbered reaches, all without a path, have placed nodes in more than
  one basin, so not on one flow line. Their `basin_id` follows the same rule: the basin holding the smallest `pixel_seq_id`
  among their placed nodes, the pixel whose order gives `sword_reach_seq_id`. For
  {rbm['of_those_with_nodes_in_several_basins_not_basin_of_smallest_sword_dist_out_node']:,} of them this is not the basin of
  the placed node SWORD puts furthest downstream (smallest SWORD `dist_out`).

## Rules in short

- Subnetwork matching: a system goes to the FlowNest basin holding most of its land river length (`status`; `link_kind`
  says how a system that also reaches another large basin crosses into it).
- Reach matching: a reach is searched only in its own basin (`own_basin_id`; the `basin_id` of its subnetwork when it is 0),
  only among paths longer than 8 km, in windows of 3x3, then 5x5, then 7x7 pixels (`matched_at_step` `3x3`, `5x5`, `7x7`).
  It needs at least 80% of its length covered (`cover`). Two paths are combined only when no single path covers 80%.
  Beyond 7x7 (`beyond_7x7`), only paths connected to matched neighbour reaches are accepted; a path switch made to connect
  with the neighbours (`fix_note`) counts as `beyond_7x7` when the new path needs a window beyond 7x7. Then a reach is
  placed between matched neighbours without the 80% threshold (`between_neighbors`), or keeps two paths whose order is
  against the flow (`against_flow`). After that, four rules with stricter checks take some of the reaches still not
  matched, using only neighbour reaches matched before them (no chaining) and never lake reaches (`sword_lakeflag` 1):
  `land_part_cover` (the
  downstream end lies in the sea; the cover counts the land part only), `partly_on_neighbor_path` (one long path covers 50% to
  80% and every matched neighbour reach is on it or joins it), `within_water_surface` (no neighbour reach is matched; the only
  long path within half the SWORD width plus 3 pixels covers 80%) and `short_path_into_neighbor` (a path of 8 km or less that
  flows into the path of the matched downstream neighbour). Where the SWORD width is at least 100 m, they also need MERIT
  water (`wth`) on at least half of the nearest path pixels.
  Then the same-water rule: on main-channel reaches (`sword_main_side` 0) with their own basin, less than half in the sea and a
  SWORD facc_total in order with the neighbour reaches, a reach not matched (lake reaches included) or matched to a path whose area
  disagrees takes the path that covers at least 50% of it in the `beyond_7x7` window when exactly one such path has an agreeing
  area, and that path is longer than 8 km, runs along both ends of the reach, does not run against the flow, and is the
  path of every trusted matched neighbour reach (area agrees, already switched or matched by this rule, or no facc) or joins
  it. Below 80% cover, or off the river corridor (median distance to the path beyond half the SWORD width plus 180 m and
  beyond 3 pixels; lake reaches excepted), it also needs a witness: a matched neighbour reach whose area agrees or which was
  itself switched or matched by this rule. The rule runs in rounds: each round judges all reaches on the state at the start of
  the round and changes them together; two neighbour reaches that would both change to paths that do not join are both left.
  Repeated until no reach changes (each reach changes at most once); such reaches are `area_guided_switch` or
  `area_guided_rescue`. A switched reach is often flagged `{FAR_REACH[0]}`: there the MERIT channel that
  carries the SWORD facc_total runs up to about 1 km from the SWORD centreline, and the MERIT channel close to the line drains far
  less; the flag tells the user that the two lines are apart, not that the path is wrong.
  Reaches still not matched take the path(s) of their nodes continued from a matched neighbour reach (`continued_from_sword_reach_id`)
  when at least 80% of all its nodes lie on them (`continued_from_neighbor`; `cover` is then the fraction of all nodes of the
  reach).
- Node placement: nodes of a matched reach are placed in order along its path(s) (`placed_by` matched_path).
  Nodes of other reaches may be continued from a placed neighbour reach along the FlowNest flow (`placed_by`
  continued_from_neighbor); such a node can lie on a path other than the one matched for its reach (`confidence_reason`
  `not_on_matched_path`). A node farther than half the width plus 180 m from its pixel is flagged (`far_from_pixel`); the
  width is the node width, else the reach width when the node width is not positive, else 0.
- Confluences: SWORD and MERIT often put the same confluence a few pixels apart. When the nodes of a reach matched to one or
  two paths are placed along the FlowNest flow (not reaches `against_flow`), they are kept above the
  MERIT confluence where another SWORD reach flowing into the same downstream reach joins (so their upstream area does not
  include that reach), and below the MERIT confluence where an upstream SWORD reach joins (so it does include that reach); the
  water of a neighbour reach is traced by the first pixel of its matched path. The rule acts only when the nodes of the reach
  straddle such a confluence. The nodes are moved only when no moved node ends up farther than half the width plus 180 m from
  its pixel (or farther than it was), and not along reaches with `area_check` disagrees, whose path may be a side channel (such
  reaches are not used as neighbours either). When all nodes of a reach lie on one side of such a confluence, the two
  networks differ in topology there, not by a few pixels, and nothing is moved.
- SWORD lake and reservoir reaches (`sword_lakeflag` 1) are matched like the others up to `against_flow`, but not by the four
  extra rules: inside a lake the MERIT flow lines run anywhere towards the outlet and do not follow the SWORD lake centreline.
  The same-water rule may match them, since the upstream area tells which MERIT channel carries the water through the lake.

## How sure the reach matches are

The farther a reach had to search, the less sure its match. `matched_at_step` tells where it was found. A reach is flagged
`{FAR_REACH[0]}` when more than half of its placed nodes are farther than half the width plus 180 m from their pixel.
`area_check` says whether the MERIT upstream area on the path agrees with the SWORD facc_total within 2 times (`agrees`), does not
(`disagrees`: review), cannot be compared (`no_facc`), or chose the path (`used_in_matching`, the area-guided reaches, whose
ratio is therefore not an independent check). For `disagrees`, `area_note` lists likely reasons and
`same_area_pixel_distance_km` gives the median, over the nodes that have one within 11 km, of the distance to the nearest MERIT
pixel whose upstream area agrees (nodes without one are left out, so it does not say how many nodes have one): a short distance means
another MERIT channel nearby carries the SWORD facc_total, so check the match; a long distance, or none within 11 km
(`no_same_area_pixel_within_11km`), means no MERIT channel nearby carries the SWORD facc_total, so no better path was found and
MERIT and SWORD disagree on the area there.

| `matched_at_step` | reaches | flagged `{FAR_REACH[0]}` | `area_check` agrees | disagrees | no_facc | used_in_matching |
|---|---:|---:|---:|---:|---:|---:|
""" + ''.join(f"| `{r['matched_at_step']}` | {r['reaches']:,} | {100 * r['share_most_nodes_far']:.1f}% | "
              + ' | '.join(f"{r['area_check'].get(k, 0):,}" for k in ('agrees', 'disagrees', 'no_facc', 'used_in_matching')) + " |\n"
              for r in meta['reach_reliability_by_step']) + f"""
For a strict subset, keep `matched_at_step` `3x3`, `5x5` or `7x7` with `area_check` `agrees` and leave out reaches flagged
`{FAR_REACH[0]}`; for nodes, also leave out nodes flagged `far_from_pixel`. `match_confidence` (next section) sums these checks up per row.

""" + confidence_section(meta['match_confidence']) + """## Sources

- SWORD v17c: Gearon, J. H., Altenau, E. H., Pavelsky, T. M., Durand, M. T., Collot d'Escury, N., Yang, X.,
  Malaterre, P.-O., Frasson, R. P. d. M., & Bendezu, L. (2026). SWOT River Database (SWORD) (Version v17c)
  [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23870877
- SWORD: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., & Bendezu, L. (2021).
  The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for
  satellite data products. Water Resources Research, 57, e2021WR030054. https://doi.org/10.1029/2021WR030054

Columns copied from SWORD (`sword_` columns other than the new `sword_reach_seq_id` and `sword_node_seq_id`) keep the
terms of use of SWORD.

## Columns
"""
    for tb in ('subnetwork', 'reach', 'node'):
        t += f"\n### {tb}\n\n| column | meaning |\n|---|---|\n"
        t += ''.join(f"| `{n}` | {m.replace('|', '/')} |\n" for n, m in cols[tb].items())
    open(O + 'README.md', 'w').write(t)


def main(face_dir, line_dir, point_dir, fn_dir, out_parent, sword_gpkg_dir=None):
    """Write the three release tables, README.md, metadata.json and SHA256SUMS.txt into a new folder under out_parent;
    returns the folder."""
    M = pd.read_csv(os.path.join(os.path.dirname(__file__), 'rename_map_sword_v17c_crosswalk.csv'), keep_default_na=False)
    names = {}
    for a, b in M[(M.release_name != '') & (M.internal_name != M.release_name)][['internal_name', 'release_name']].values:
        if '_' in a:
            assert names.get(a, b) == b, (a, names[a], b)
            names[a] = b
    f_face = face_dir + 'sword_system_to_flownest_basin_with_issues.csv'
    f_line = line_dir + 'sword_reach_to_flownest_path_global_with_issues.parquet'
    f_prch = point_dir + 'sword_reach_node_placement_global.parquet'
    f_node = point_dir + 'sword_node_to_flownest_pixel_global.parquet'
    f_num = point_dir + 'sword_seq_id_numbering.json'
    num = json.load(open(f_num))

    S = pd.read_csv(f_face, float_precision='round_trip', keep_default_na=False, na_values=[''])
    for c in ['basin_id', 'flowdivide_level3_id', 'is_endorheic', 'second_basin_id', 'other_large_basin_id', 'best_outlet_reach',
              'outlet_basin_id', 'outlet_steps_up', 'systems_in_basin', 'link_count']:
        S[c] = S[c].astype('Int64')
    for c in ['basin_id', 'flowdivide_level3_id', 'second_basin_id', 'other_large_basin_id', 'best_outlet_reach', 'outlet_basin_id']:
        S[c] = S[c].fillna(0)
    for c in ['issue', 'cause', 'action', 'why_not_fixed', 'status', 'link_kind', 'link_direct', 'link_lakeflags']:
        S[c] = S[c].fillna('').astype(str)

    L = pd.read_parquet(f_line).drop(columns=['sword_facc_flag'])
    P = pd.read_parquet(f_prch)
    assert L.reach_id.is_unique and P.reach_id.is_unique and len(L) == len(P) and set(L.reach_id) == set(P.reach_id)
    P = P.set_index('reach_id').reindex(L.reach_id).reset_index()
    for c in REACH_SAME:
        assert (L[c].values == P[c].values).all(), ('line and point reach tables differ', c)
    TIER_COLS = ('match_confidence', 'confidence_reason')
    extra = [c for c in M[M.table == 'reach'].internal_name if c not in L.columns and c not in TIER_COLS]
    assert set(extra) <= set(P.columns), extra
    R = pd.concat([L, P[extra]], axis=1)
    for c in ISSUE_COLS:
        assert R[c].notna().all(), c
    R['own_basin_id'] = R.own_basin_id.fillna(0).astype(np.int64)
    R['path_level'] = R.path_level.astype('Int64')
    st4 = R.match_stage.values == 4

    N = pd.read_parquet(f_node)
    assert N.node_id.is_unique
    nl = R.sword_facc_total_km2.isna() & R.sea_fraction.isna()
    fac = N.groupby('reach_id').sword_facc_total_km2.median().reindex(R.reach_id).values   # the median of the node facc, not rounded
    sys0 = set(S.subnetwork_id[S.basin_id == 0])
    assert set(R.subnetwork_id[nl]) <= sys0, set(R.subnetwork_id[nl])
    R.loc[nl, 'sword_facc_total_km2'] = fac[nl.values]
    R.loc[nl, 'sea_fraction'] = 1.0
    far = (R.nodes_placed > 0) & (2 * R.nodes_far_from_pixel > R.nodes_placed)
    add_issue(R, far.values, FAR_REACH)
    check_issue_counts(R, 'reach')
    rel = reliability(R)

    assert os.path.basename(fn_dir.rstrip('/')) == num['flownest_release'], (fn_dir, num['flownest_release'])
    f_pix = fn_dir + 'flownest_merit90m_pixel_1km2.parquet'
    pl = N.pixel_seq_id.values > 0
    a = flownest_pixel_area(f_pix, N.pixel_seq_id.values[pl])
    assert np.allclose(np.round(a.astype(np.float64), 2), N.merit_upa_km2.values[pl], rtol=0, atol=1e-6)
    assert np.array_equal(a.astype(np.float32), N.merit_upa_float32_km2.values[pl].astype(np.float32))
    assert N.merit_upa_km2[~pl].isna().all()
    u = np.full(len(N), np.nan, np.float32)
    u[pl] = a
    N['merit_upa_km2'] = u
    N = N.drop(columns=['merit_upa_float32_km2'])
    N['upa_to_facc_ratio'] = N.merit_upa_km2.astype(np.float64) / N.sword_facc_total_km2
    for c in ISSUE_COLS:
        N[c] = N[c].fillna('')

    o = np.lexsort((N.node_id.values, -N.dist_out_m.values, N.reach_id.values))
    u = N.merit_upa_km2.values.astype(np.float64)[o]
    rid = N.reach_id.values[o]
    prev = np.r_[np.nan, u[:-1]]
    prev[np.r_[True, rid[1:] != rid[:-1]]] = np.nan
    dec = np.zeros(len(N), bool)
    dec[o] = (u < prev - 1e-6)
    old = N.issue.str.contains('upa_decreases_downstream', regex=False).values
    assert not (old & ~dec).any() and (dec <= pl).all(), int((old & ~dec).sum())
    add_issue(N, dec & ~old, UPA_DEC)
    N['cause'] = on_unique(N.cause, lambda x: x.str.replace('nodes placed against the FlowNest flow direction', UPA_DEC[1], regex=False))
    N['why_not_fixed'] = on_unique(N.why_not_fixed, lambda x: x.str.replace('see the reversed or against-flow flag', UPA_DEC[3], regex=False))
    assert int(N.cause.str.count(re.escape(UPA_DEC[1])).sum()) == int(dec.sum())
    pc = pd.Series(N.pixel_seq_id.values[pl]).value_counts()
    cnt_pix = np.where(pl, pd.Series(N.pixel_seq_id.values).map(pc).fillna(0).values, 0).astype(np.int64)
    shr = pl & (cnt_pix > 1)
    olds = N.issue.str.contains('shares_pixel', regex=False).values
    assert not (olds & ~shr).any() and (cnt_pix >= N.nodes_on_this_pixel.values).all()
    N['nodes_on_this_pixel'] = cnt_pix
    add_issue(N, shr & ~olds, SHARE_X)
    r4 = R[st4].set_index('reach_id')
    in4 = N.reach_id.isin(r4.index).values
    drop_issue(N, in4 & pl & N.issue.str.contains('reach_not_matched_to_path', regex=False).values, 'reach_not_matched_to_path')
    assert not (in4 & N.issue.str.contains('reach_not_matched_to_path', regex=False).values).any()
    pat = r'line step: reach not matched to a path \(([a-z_]+)\), continuing from neighbour reaches failed \(([a-z_0-9]+)\)'
    m71 = in4 & ~pl
    assert (N.cause[m71].str.count(pat) == 1).all()
    N.loc[m71, 'cause'] = N.cause[m71].str.replace(pat, r'reach matching stages 1 to 3 did not match the reach (\1), stage 4 matched it '
                                                   r'from its continued nodes, but continuing from neighbour reaches did not reach this node (\2)', regex=True)
    p1 = r4.path_seq_id.reindex(N.reach_id[in4]).values
    p2 = r4.path2_seq_id.reindex(N.reach_id[in4]).values
    ps = N.path_seq_id.values[in4]
    prt = np.where(ps == 0, '', np.where(ps == p1, 'a', np.where((p2 > 0) & (ps == p2), 'b', '')))
    N.loc[in4, 'part'] = prt
    off = np.zeros(len(N), bool)
    off[np.flatnonzero(in4)[(ps > 0) & (prt == '')]] = True
    add_issue(N, off, OFF_PATH)
    on = np.flatnonzero(in4)[prt != '']
    med = pd.Series(N.merit_upa_km2.values[on].astype(np.float64)).groupby(N.reach_id.values[on]).median().reindex(R.reach_id[st4])
    assert med.notna().all() and np.allclose(med.values, R.merit_upa_on_path_km2.values[st4], rtol=0, atol=0.1 + 1e-6)
    R['merit_upa_on_path_km2'] = R.merit_upa_on_path_km2.astype(np.float64)
    R.loc[st4, 'merit_upa_on_path_km2'] = med.round(1).values
    qn = pd.Series(N.merit_upa_km2.values[on].astype(np.float64) / N.sword_facc_total_km2.values[on]).groupby(N.reach_id.values[on]).median()
    qn = qn.reindex(R.reach_id[st4]).values
    assert np.allclose(qn, R.area_ratio.values[st4], rtol=3e-3, atol=1e-3, equal_nan=True), 'stage 4 area ratio differs from line_10'
    q3 = np.round(qn, 3)
    chk = np.where(np.isnan(qn), 'no_facc', np.where((q3 >= 0.5) & (q3 <= 2), 'agrees', 'disagrees'))
    assert (chk == R.area_check.values[st4]).all(), 'stage 4 area_check changes with the float32 upstream area'
    R.loc[st4, 'area_ratio'] = qn
    rel = reliability(R)
    assert (N.chain_pos.values[~pl] == 0).all() and (N.chain_pos.values[pl] >= 0).all()
    N['chain_pos'] = np.where(pl, N.chain_pos.values + 1, 0).astype(np.int64)
    assert N.along_flownest_flow.dtype == bool
    N['along_flownest_flow'] = N.along_flownest_flow.astype('boolean').where(pl, pd.NA)
    check_issue_counts(N, 'node')
    gN = pd.DataFrame({'reach_id': N.reach_id.values, 'd': dec, 's': shr}).groupby('reach_id').sum()
    R['nodes_upa_decreases'] = gN.d.reindex(R.reach_id).values.astype(np.int64)
    R['nodes_sharing_pixel'] = gN.s.reindex(R.reach_id).values.astype(np.int64)

    f_bas = fn_dir + 'flownest_merit90m_basin.parquet'
    B = pd.read_parquet(f_bas, columns=['basin_id', 'outlet_pixel_seq_id', 'channel_pixel_count'])
    search = R.basin_id.values.astype(np.int64)
    assert (search == np.where(R.own_basin_id.values > 0, R.own_basin_id.values, R.system_basin_id.values)).all()
    dpx = R.flownest_downstream_pixel_seq_id.values.astype(np.int64)
    nb = basin_of_pixels(B, dpx)
    numbered = dpx > 0
    assert (nb[numbered] > 0).all() and (nb[~numbered] == 0).all(), 'a placed node lies outside every basin'
    matched = R.match_stage.values > 0
    assert numbered[matched].all() and (nb[matched] == search[matched]).all(), 'a reach with a path changed basin'
    nbn = basin_of_pixels(B, N.pixel_seq_id.values)
    assert (nbn[pl] > 0).all() and (nbn[~pl] == 0).all()
    kb = pd.DataFrame({'r': N.reach_id.values[pl], 'b': nbn[pl]}).groupby('r').b.nunique()
    multi = kb.index[kb.values > 1]
    chg = numbered & (nb != search)
    rb = {'numbered_reaches_basin_not_search_basin': int(chg.sum()), 'of_those_without_path': int((chg & ~matched).sum()),
          'unnumbered_reaches_set_to_0': int((~numbered & (search > 0)).sum()),
          'numbered_reaches_with_nodes_in_several_basins': int(len(multi)),
          'of_those_with_nodes_in_several_basins_without_path': int((~matched[R.reach_id.isin(multi).values]).sum())}
    Np = pd.DataFrame({'r': N.reach_id.values[pl], 'd': N.dist_out_m.values[pl], 'i': N.node_id.values[pl], 'b': nbn[pl]})
    swb = Np.sort_values(['r', 'd', 'i']).drop_duplicates('r').set_index('r').b
    rbi = pd.Series(nb, index=R.reach_id.values)
    rb['of_those_with_nodes_in_several_basins_not_basin_of_smallest_sword_dist_out_node'] = \
        int((swb.reindex(multi).values != rbi.reindex(multi).values).sum())
    assert rb['numbered_reaches_basin_not_search_basin'] == rb['of_those_without_path'], rb
    assert rb['numbered_reaches_with_nodes_in_several_basins'] == rb['of_those_with_nodes_in_several_basins_without_path'], rb
    R['basin_id'] = nb

    for df in (R, N):
        for c in TIER_COLS:
            df[c] = ''
    used = {}
    tabs = {}
    for t, df in (('subnetwork', S), ('reach', R), ('node', N)):
        out = apply_map(df, M, t)
        out = fix_sentences(fix_codes(out), names, used)
        if 'rescue_not_done' in out:
            for a, b in zip('defg', ('land_part_cover', 'partly_on_neighbor_path', 'within_water_surface', 'short_path_into_neighbor')):
                out['rescue_not_done'] = out['rescue_not_done'].str.replace(rf'(^|;){a}:', rf'\g<1>{b}:', regex=True)
            assert not out['rescue_not_done'].str.contains(r'(?:^|;)[defg]:', regex=True, na=False).any()
        if 'switch_not_done' in out:
            out['switch_not_done'] = out['switch_not_done'].str.replace(':upstrea:', ':upstream_continuation_switch:', regex=False)
            assert out['switch_not_done'].fillna('').str.fullmatch(r'|\d+:upstream_continuation_switch:needs_\d+x\d+').all()
        for c in SENTENCE_COLS:
            for a, b in SENT_FIX:
                out[c] = out[c].str.replace(a, b, regex=False)
            assert not out[c].str.contains('seam table|pair table', regex=True, na=False).any(), (t, c)
            bad = out[c][out[c].str.contains(OLD_WORDS, regex=True, na=False)]
            assert bad.empty, (t, c, bad.unique()[:3])
        check_issue_counts(out, t)
        tabs[t] = out
    for t, cs in STORED_6.items():
        for c in cs:
            tabs[t][c] = stored(tabs[t][c].values.astype(np.float64))
    S, R, N = tabs['subnetwork'], tabs['reach'], tabs['node']
    R['match_confidence'], R['confidence_reason'] = CF.reach_confidence(R)
    N['match_confidence'], N['confidence_reason'] = CF.node_confidence(N, R.set_index('sword_reach_id').match_confidence)
    S = S.sort_values('sword_subnetwork_id').reset_index(drop=True)
    R, nr = order_by_seq(R, 'sword_reach_seq_id', 'sword_reach_id')
    N, nn = order_by_seq(N, 'sword_node_seq_id', 'sword_node_id')

    assert S.sword_subnetwork_id.is_unique and set(R.sword_subnetwork_id) <= set(S.sword_subnetwork_id)
    assert set(N.sword_reach_id) == set(R.sword_reach_id)
    g = N.groupby('sword_reach_id')
    Ri = R.set_index('sword_reach_id')
    assert (g.size().reindex(Ri.index).values == Ri.nodes.values).all()
    pl = N[N.pixel_seq_id > 0]
    gp = pl.groupby('sword_reach_id')
    assert (gp.size().reindex(Ri.index).fillna(0).values == Ri.nodes_placed.values).all()
    mn = gp.pixel_seq_id.min().reindex(Ri.index).fillna(0).astype(np.int64)
    assert (mn.values == Ri.most_downstream_pixel_seq_id.values).all()
    assert ((Ri.sword_reach_seq_id > 0) == (Ri.nodes_placed > 0)).all()
    assert ((N.sword_node_seq_id > 0) == (N.pixel_seq_id > 0)).all()
    assert (N.sword_subnetwork_id.values == R.set_index('sword_reach_id').sword_subnetwork_id.reindex(N.sword_reach_id).values).all()
    assert nn == num['sword_nodes_numbered'] and nr == num['sword_reaches_numbered'], (nn, nr, num)
    rbas = R.basin_id.values[:nr]
    assert (R.basin_id.values[nr:] == 0).all() and (rbas > 0).all()
    starts = np.diff(rbas, prepend=-1) != 0           # also right when no reach is numbered
    assert pd.Series(rbas[starts]).is_unique, 'the reaches of a basin are not consecutive rows'
    last = np.arange(nr) + R.upstream_sword_reach_count.values[:nr] - 1
    assert (rbas[last] == rbas).all(), 'an upstream reach range crosses a basin'
    placed = N.pixel_seq_id.values > 0
    assert not (N.row.values[~placed].any() or N.col.values[~placed].any()), 'an unplaced node has a pixel'
    for c in ('row', 'col'):                          # empty, not 0, for a node not placed (col 0 is a real column)
        N[c] = N[c].astype('Int64').where(placed, pd.NA)
    for t, df in (('subnetwork', S), ('reach', R), ('node', N)):
        for c in df.columns:
            if c.endswith('_id') or c.endswith('_id_1km2') or c in ('chain_pos', 'found_window', 'switch_window'):
                assert df[c].notna().all() and (df[c] >= 0).all(), (t, c)
            if df[c].dtype == object:
                assert not df[c].str.contains('flownest', regex=False, na=False).any(), (t, c)
    assert (N.row.isna().values == ~placed).all() and (N.col.isna().values == ~placed).all() and (N.row.dropna() >= 0).all()
    uq = {(i, c): pd.Series(x[c].dropna().unique(), dtype=object) for i, x in enumerate((S, R, N)) for c in SENTENCE_COLS}
    left = {a for a in names for u in uq.values() if a != names[a] and u.str.contains(name_pat(a), regex=True, na=False).any()}
    assert not left, left

    # only the released columns are written (the others were needed for the checks and match_confidence above)
    released = {t: list(M[(M.table == t) & (M.released == 'yes')].release_name) for t in ('subnetwork', 'reach', 'node')}
    stamp = time.strftime('%Y%m%d_%H%M', time.gmtime(time.time() + 8 * 3600))
    O = f'{out_parent}{PREFIX[:-1]}_{stamp}/'
    os.makedirs(O)
    files = {}
    for t, df in (('subnetwork', S), ('reach', R), ('node', N)):
        out = df[released[t]]
        for ext in ('parquet',) if t == 'node' else ('parquet', 'csv'):    # the node table only as Parquet
            p = O + PREFIX + t + '.' + ext
            write_parquet(out, p, SUBNETWORK_NOT_NULL if t == 'subnetwork' else ()) if ext == 'parquet' else out.to_csv(p, index=False)
            files[os.path.basename(p)] = {'rows': len(out), 'bytes': os.path.getsize(p), 'sha256': sha256(p)}
    cnt = {
        'sword_subnetworks': len(S), 'sword_subnetworks_with_basin': int((S.basin_id > 0).sum()),
        'sword_reaches': len(R), 'sword_reaches_matched_to_path': int((R.match_stage > 0).sum()),
        'sword_reaches_numbered': nr, 'sword_nodes': len(N), 'sword_nodes_placed': nn,
        'status': {k: int((S.status == k).sum()) for k in STATUS},
        'link_kind': {k: int((S.link_kind == k).sum()) for k in LINK_KIND},
    }
    assert sum(cnt['status'].values()) == len(S) and sum(cnt['link_kind'].values()) == int((S.link_kind != '').sum())
    cols = {t: {r.release_name: r.meaning for r in M[(M.table == t) & (M.released == 'yes')].itertuples()}
            for t in ('subnetwork', 'reach', 'node')}
    fmeta = json.load(open(fn_dir + 'metadata.json'))
    gpkg = sword_gpkg_dir or IN.SW
    sword_files = [IN.gpkg_path(c, gpkg) for c in ('as', 'sa', 'na', 'eu', 'af', 'oc')]
    f_map = os.path.join(os.path.dirname(__file__), 'rename_map_sword_v17c_crosswalk.csv')
    from .. import __version__
    meta = {
        'schema_version': '5', 'product': 'FlowNest',
        'what': 'a crosswalk from SWORD v17c (subnetwork, reach, node) to a FlowNest release on MERIT Hydro 90 m, at the base '
                'threshold (1 km2) (flownest.sword); subnetwork to basin, reach to river path, node to channel pixel',
        'external_dataset': 'SWORD v17c', 'numbering_version': num['numbering_version'],
        'code_version': f'flownest {__version__} (Python)',
        'numbering_sha256': num['numbering_sha256'], 'seq_id_note': num['note'], 'ranges': num['ranges'],
        'code_commit': os.environ.get('FLOWNEST_CODE_COMMIT') or 'none',
        'release': {'generation': fmeta.get('generation'), 'code_commit': fmeta.get('code_commit'),
                    'metadata_sha256': sha256(fn_dir + 'metadata.json'), 'files_read': RELEASE_FILES_READ},
        'inputs': {os.path.basename(p): {'bytes': os.path.getsize(p), 'sha256': sha256(p)} for p in sword_files + [f_map]},
        'counts': cnt,
        'reach_basin_id': dict(rule='basin of the placed node with the smallest pixel_seq_id (most_downstream_pixel_seq_id); 0 if no node is placed; '
                                    'the search basin is own_basin_id, or the basin_id of its subnetwork when own_basin_id is 0', **rb),
        'columns': cols,
        'reach_reliability_by_step': rel,
        'match_confidence': confidence_counts(R, N),
        'files': files,
    }
    write_readme(O, meta, cols, num['flownest_release'])
    json.dump(meta, open(O + 'metadata.json', 'w'), indent=1, ensure_ascii=False)
    with open(O + 'SHA256SUMS.txt', 'w') as f:
        for name in sorted(os.listdir(O)):
            if name != 'SHA256SUMS.txt':
                f.write(f'{sha256(O + name)}  {name}\n')
    return O
