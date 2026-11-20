"""match_confidence and confidence_reason of the SWORD crosswalk rows, from the released issue codes.

Tiers, best first: high, medium, low, not_found. Every issue code has one tier (a code not listed stops the program, so
a new code never passes as high unclassified).

Reach: the tier is the worst tier of its codes. The reason lists its low and medium codes (not_found gives only the
`not_found_reason`). A matched reach with no medium or low code is high only with all four pieces of evidence: cover
at least 0.8, upstream area within 2x of SWORD facc (`area_check` agrees), at most half of its placed nodes far from
their pixel, and its nodes along the flow; otherwise it is medium and the reason names what is missing. An area
disagreement is medium when `area_note` explains it, low when not.

Node: the worse of its own tier and its reach's tier (a placed node of an unmatched reach is at most low); the
reason lists its own not_found, low and medium codes and ends with `reach_<tier>`.
"""
import numpy as np

TIERS = ('high', 'medium', 'low', 'not_found')
HIGH, MEDIUM, LOW, NOT_FOUND = range(4)

REACH_CODES = {
    'not_matched': NOT_FOUND,
    'most_nodes_far_from_pixel': LOW, 'neighbor_not_connected': LOW, 'no_node_placed': LOW, 'node_area_disagrees': LOW,
    'upstream_area_disagrees': None,
    'reversed_vs_flow': MEDIUM, 'neighbor_pair_against_flow': MEDIUM, 'two_paths_against_flow': MEDIUM,
    'matched_between_neighbors': MEDIUM, 'matched_by_continuing_from_neighbor': MEDIUM, 'matched_by_upstream_area': MEDIUM,
    'path_switched_by_upstream_area': MEDIUM, 'matched_partly_on_path': MEDIUM, 'matched_within_water_surface': MEDIUM,
    'matched_short_path': MEDIUM, 'matched_on_land_part': MEDIUM, 'switch_not_done': MEDIUM,
    'pair_rejected_order': HIGH, 'path_switched': HIGH,
}
NODE_CODES = {
    'not_placed': NOT_FOUND,
    'far_from_pixel': LOW, 'not_on_matched_path': LOW, 'reach_not_matched_to_path': LOW, 'seam_not_connected': LOW,
    'reversed_vs_flow': MEDIUM, 'against_flow_part': MEDIUM, 'upa_decreases_downstream': MEDIUM,
    'seam_backward_on_same_path': MEDIUM, 'seam_junction_reversed': MEDIUM,
    'shares_pixel': HIGH, 'moved_below_upstream_reach_end': HIGH, 'moved_above_merit_confluence': HIGH,
    'moved_below_merit_confluence': HIGH, 'above_merit_confluence': HIGH, 'below_merit_confluence': HIGH,
}
HIGH_REASON = 'cover_at_least_80pct; upstream_area_agrees; most_nodes_near_pixel; along_flow'

# every code confidence_reason can hold: (code, table, tier, meaning), for the code table of the README; the same list
# as in the README of the SWORD crosswalk package. The tier is empty for reach_<tier> on a node.
REASON_MEANINGS = (
    ('cover_at_least_80pct', 'reach', 'high', 'at least 80% of the reach lies on its matched path(s)'),
    ('upstream_area_agrees', 'reach', 'high',
     'the MERIT upstream area on the path is within 2 times of the SWORD facc_total'),
    ('most_nodes_near_pixel', 'reach', 'high',
     'at least half of the placed nodes lie within half the width plus 180 m of their pixel'),
    ('along_flow', 'reach', 'high', 'the placed nodes run along the FlowNest flow'),
    ('cover_below_80pct', 'reach', 'medium',
     'less than 80% of the reach lies on its path(s) (a check that is missing, not a fault)'),
    ('upstream_area_not_checked', 'reach', 'medium',
     'no SWORD facc_total, or the upstream area chose the path, so the area gives no independent check'),
    ('nodes_not_checked', 'reach', 'medium', 'the distance of the nodes to their pixels could not be checked'),
    ('reversed_vs_flow', 'reach and node', 'medium',
     'the reach or node runs against the FlowNest flow (distributary, tidal reach or SWORD direction)'),
    ('upstream_area_disagrees_explained', 'reach', 'medium',
     'the MERIT upstream area on the path is more than 2 times off the SWORD facc_total; `area_note` gives a likely '
     'reason'),
    ('neighbor_pair_against_flow', 'reach', 'medium',
     'with a SWORD neighbour reach, the MERIT flow on the shared path or at the confluence runs the other way'),
    ('two_paths_against_flow', 'reach', 'medium', 'the reach needs two paths, joined in an order against the flow'),
    ('matched_between_neighbors', 'reach', 'medium',
     'placed between its matched SWORD neighbour reaches, without the 80% cover'),
    ('matched_by_continuing_from_neighbor', 'reach', 'medium',
     'matched through its nodes continued from a matched neighbour reach along the FlowNest flow'),
    ('matched_by_upstream_area', 'reach', 'medium',
     'matched by the same-water rule: the only nearby long path whose upstream area agrees'),
    ('path_switched_by_upstream_area', 'reach', 'medium',
     'switched by the same-water rule to the only nearby long path whose upstream area agrees'),
    ('matched_partly_on_path', 'reach', 'medium',
     'one long path covers 50% to 80% of the reach and the matched neighbour reaches are on it or join it'),
    ('matched_within_water_surface', 'reach', 'medium',
     'no neighbour reach matched; the only long path within half the SWORD width plus 3 pixels covers 80%'),
    ('matched_short_path', 'reach', 'medium',
     'a path of 8 km or less that flows into the path of the matched downstream neighbour reach'),
    ('matched_on_land_part', 'reach', 'medium',
     'the downstream end of the reach lies in the sea; the cover counts the land part only'),
    ('switch_not_done', 'reach', 'medium',
     'the downstream neighbour reach continues on another path, which covers this reach only beyond 7x7; not '
     'switched'),
    ('most_nodes_far_from_pixel', 'reach', 'low',
     'more than half of the placed nodes lie farther than half the width plus 180 m from their pixel'),
    ('neighbor_not_connected', 'reach', 'low',
     'the path is neither the path of a matched SWORD neighbour reach nor joined to it at a confluence'),
    ('no_node_placed', 'reach', 'low', 'matched to a path, but none of its nodes could be placed on it'),
    ('node_area_disagrees', 'reach', 'low',
     'the path was chosen by its upstream area, but at its placed nodes the area is more than 2 times off'),
    ('upstream_area_disagrees_unexplained', 'reach', 'low',
     'the MERIT upstream area on the path is more than 2 times off the SWORD facc_total, with no likely reason'),
    ('mostly_sea', 'reach', 'not_found', 'more than half of the reach lies in the sea beyond the MERIT coastline'),
    ('not_in_any_basin', 'reach', 'not_found',
     'most of the reach lies outside every FlowNest basin, though less than half in the sea'),
    ('no_matched_neighbor', 'reach', 'not_found',
     'a long path covers 80% only beyond 7x7, and no SWORD neighbour reach is matched to confirm it'),
    ('candidate_not_connected', 'reach', 'not_found',
     'a long path covers 80% only beyond 7x7, but it is not the path of a matched neighbour reach nor joined to it'),
    ('only_short_path_nearby', 'reach', 'not_found', 'within 7x7 only paths of 8 km or shorter cover the reach'),
    ('partly_on_path', 'reach', 'not_found', 'only 50% to 80% of the reach lies on one long path'),
    ('far_from_path', 'reach', 'not_found',
     'the reach is far from every long path (mostly a lake centreline, or a channel MERIT does not draw)'),
    ('no_land', 'reach', 'not_found', 'the SWORD system lies on no MERIT land'),
    ('node_near_pixel', 'node', 'high', 'the node lies within half the width plus 180 m of its pixel'),
    ('against_flow_part', 'node', 'medium', 'a SWORD distributary that MERIT draws as a tributary'),
    ('upa_decreases_downstream', 'node', 'medium',
     'the upstream area is smaller than at the previous node upstream along SWORD'),
    ('seam_backward_on_same_path', 'node', 'medium',
     'this node and the node of the neighbour reach are on one path but in the wrong order'),
    ('seam_junction_reversed', 'node', 'medium',
     'SWORD says the upstream reach flows into the downstream reach, but in MERIT the flow runs the other way'),
    ('far_from_pixel', 'node', 'low', 'the node lies farther than half the width plus 180 m from its pixel'),
    ('not_on_matched_path', 'node', 'low',
     'continued from the neighbour reach onto a path other than the path(s) matched for its reach'),
    ('reach_not_matched_to_path', 'node', 'low',
     'its reach has no path; the node was continued from a placed neighbour reach'),
    ('seam_not_connected', 'node', 'low',
     'this node and the node of the neighbour reach are on FlowNest paths not connected along the flow'),
    ('not_placed', 'node', 'not_found', 'the node could not be placed on a FlowNest pixel'),
    ('reach_high', 'node', '', 'the tier of the reach of the node (its reason is in the reach table)'),
    ('reach_medium', 'node', '', 'as reach_high'),
    ('reach_low', 'node', '', 'as reach_high'),
    ('reach_not_found', 'node', '', 'as reach_high'),
)


def meanings_table():
    """The README code table (markdown), one row per code of REASON_MEANINGS."""
    return '| code | table | tier | meaning |\n|---|---|---|---|\n' + ''.join(
        f'| `{c}` | {tb} | {tier or "-"} | {m} |\n' for c, tb, tier, m in REASON_MEANINGS)


def check_known(reasons):
    """Stop on a code of confidence_reason that the README code table (REASON_MEANINGS) does not explain."""
    known = {c for c, _, _, _ in REASON_MEANINGS}
    bad = {c for r in reasons for c in r.split('; ')} - known
    if bad:
        raise ValueError(f'confidence_reason codes without a meaning: {sorted(bad)}')


def named_codes(issue, table, explained=False):
    """(worst tier, codes named in the order not_found, low, medium; each once) of one issue cell. An upstream area
    disagreement is medium and named `upstream_area_disagrees_explained` when `area_note` gives a reason, else low and
    `_unexplained`."""
    tc = []
    for c in (issue.split('; ') if issue else []):
        if c not in table:
            raise ValueError(f'issue code {c!r} has no confidence tier')
        t = table[c]
        if t is None:
            t, c = (MEDIUM, c + '_explained') if explained else (LOW, c + '_unexplained')
        tc.append((t, c))
    named = []
    for tier in (NOT_FOUND, LOW, MEDIUM):
        named += [c for t, c in tc if t == tier and c not in named]
    return max([t for t, _ in tc], default=HIGH), named


def reach_confidence(R):
    """R: the released reach table. Returns (tier names, reasons)."""
    tiers, reasons = [], []
    cols = zip(R.issue.values, R.area_note.fillna('').values, R.not_found_reason.fillna('').values, R.cover.values,
               R.area_check.fillna('').values, R.nodes_placed.values, R.nodes_far_from_pixel.values, R.orientation.fillna('').values)
    for issue, note, nf, cover, check, placed, far, orient in cols:
        tier, reason = named_codes(issue, REACH_CODES, note != '')
        if tier == NOT_FOUND:
            reason = [nf or 'not_matched']
        elif tier == HIGH:
            miss = [name for name, ok in (('cover_below_80pct', np.isfinite(cover) and cover >= 0.8),
                                          ('upstream_area_not_checked', check == 'agrees'),
                                          ('nodes_not_checked', placed > 0 and 2 * far <= placed),
                                          ('reversed_vs_flow', orient != 'reversed')) if not ok]
            tier, reason = (MEDIUM, miss) if miss else (HIGH, [HIGH_REASON])
        assert reason, issue
        tiers.append(TIERS[tier])
        reasons.append('; '.join(reason))
    check_known(set(reasons))
    return np.array(tiers, object), np.array(reasons, object)


def node_confidence(N, reach_tier):
    """N: the released node table; reach_tier: Series of reach tier names by sword_reach_id."""
    rt = N.sword_reach_id.map(reach_tier)
    assert rt.notna().all(), 'a node of a reach with no tier'
    rank = {n: i for i, n in enumerate(TIERS)}
    out = {}
    for key in set(zip(N.issue.values, rt.values)):
        issue, r = key
        own, reason = named_codes(issue, NODE_CODES)
        found = own if own == NOT_FOUND else max(own, min(rank[r], LOW))
        if found == HIGH:
            reason = reason + ['node_near_pixel']
        out[key] = (TIERS[found], '; '.join(reason + ['reach_' + r]))
    check_known({b for _, b in out.values()})
    v = [out[k] for k in zip(N.issue.values, rt.values)]
    return np.array([a for a, _ in v], object), np.array([b for _, b in v], object)
