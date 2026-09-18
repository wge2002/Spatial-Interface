"""Fast geometry: region-constrained fits, relational targets, bounded stage plans.

Model-free, browser-free, simulator-free. Every function here takes plain numbers
so the whole file is unit-testable, exactly like ``spatial_interface/geometry_ref.py`` (whose
frame conversions, pairing checks and small linear algebra this module reuses
rather than re-deriving). The JS constants are evaluated by the MCP server.

The contract this implements is ``docs/FAST_GEOMETRY_PROTOCOL.md``. What it
deliberately does not do:

* it never reads MuJoCo object state, segmentation ids, the name->ground-truth
  map or the success flag — inputs are the RGB-D cloud already in the browser,
  the calibration, the robot's own telemetry, and the model's own picks;
* it never treats a model-supplied name as a verified identity;
* it never reports a 6-DoF pose the fit does not support, and never upgrades
  proximity into attachment.
"""

from __future__ import annotations

import math

try:
    from . import geometry_ref as geom
except ImportError:  # direct script/test import
    import geometry_ref as geom

# ── gates ────────────────────────────────────────────────────────────────────

# A cluster below this is reported unknown rather than as a small object: the
# cloud is downsampled, so a thin return is evidence of sampling or occlusion.
MIN_OBJECT_POINTS = 60
MIN_BAND_POINTS = 18

# Single-link connectivity radius. Two returns further apart than this are not
# joined, so a bowl and the plate beside it stay separate clusters.
CLUSTER_RADIUS_M = 0.02

# Support surface: points within this of the lowest z mode are called table.
SUPPORT_BAND_M = 0.012
SUPPORT_MIN_FRACTION = 0.15  # below this the region is treated as object-only

# Ring/disc fits use the upper band, where a bowl's rim actually is.
RING_TOP_FRACTION = 0.25
# A band closes over returns this close to its own edge, so a flat top is one band
# rather than an arbitrary arc of one. Smaller than SUPPORT_BAND_M on purpose: this
# resolves ties, it does not merge a rim with the wall below it.
BAND_TIE_M = 0.004
RING_MIN_BEARING_DEG = 200.0
# A horizontal circle fit only describes an upright object. Beyond this tilt of
# the band's own fitted plane, the XY fit is a projection of a tilted circle and
# is refused instead of being reported as a measurement.
RING_MAX_TILT_DEG = 20.0

# RMS fit residual above which the card is invalid, per shape. blob has none: it
# claims nothing beyond a centroid, an AABB and an axis.
RESIDUAL_GATE_M = {"ring": 0.008, "disc": 0.010, "plane_patch": 0.006}

SHAPES = ("ring", "disc", "plane_patch", "blob")

# How far one *virtual target edit* may move the target in a single browser call
# (target_edit.MAX_POSITION_DELTA_M is 0.1 m; stay under it so a rounded solve is
# not rejected at the boundary). This is a UI editing limit and says NOTHING about
# how far the robot can travel in one physical waypoint: the controller
# interpolates to whatever target it is given, and sim_env.move_to's own
# waypoint_max_step governs the motion. Several edits may be chained into ONE
# physical waypoint — see ACTION_SEGMENT_M.
MAX_EDIT_STEP_M = 0.09
# Same for orientation: target_edit.MAX_ROTATION_DEG is 90, so a larger turn is
# walked in several edits before one execute rather than refused.
MAX_EDIT_ROTATION_DEG = 80.0

# How far one *physical* segment may travel before the executor stops to take a
# new paired observation. This is a perception/control choice, not a robot limit:
# the cloud is only re-measured at a segment boundary (record_sim streams
# skip_pcl frames during motion), so a longer segment means a longer stretch
# executed open-loop against geometry measured before it started. 0.25 m covers
# the whole LG8 tabletop reach in one motion; a relation whose solved target is
# further than this is split, and every split is counted and reported as a real
# physical cost rather than presented as a saving.
ACTION_SEGMENT_M = 0.25

# Re-association searches only this far from the last known centre.
CHECK_WINDOW_EXTRA_M = 0.03
CHECK_MAX_RESIDUAL_RATIO = 2.5
CHECK_AMBIGUOUS_RATIO = 0.6  # runner-up this close in size is ambiguous
# How many window clusters are judged against the stored card before one is
# accepted. Taking the largest cluster instead was a real defect: once a held bowl
# hangs over a plate, the plate's own search window contains both, the bowl is the
# bigger cluster, and the plate lost its id to a size_mismatch against an object it
# is not. Correspondence decides which cluster is the stored one; size and
# appearance still refuse, they just refuse the right candidate.
MAX_MATCH_CANDIDATES = 4

# Size correspondence. A re-fit whose radius or footprint disagrees with the
# stored one by more than this is a *different* thing that happens to be round;
# reporting the change without refusing was how "another circle nearby" kept an
# id alive. Absolute floor plus a relative term, because a 2 mm change matters on
# a 3 cm rim and does not on a 15 cm plate.
CHECK_MAX_SIZE_CHANGE_M = 0.012
CHECK_MAX_SIZE_CHANGE_FRACTION = 0.25

# Appearance correspondence, from the cloud's own per-point colour (RGB-D, an
# input the eval already allows). Mean-colour distance in unit RGB, and the
# fraction of the coarse colour histogram the two views share.
CHECK_MAX_COLOR_DISTANCE = 0.22
CHECK_MIN_COLOR_OVERLAP = 0.55
COLOR_BINS = 3  # 3x3x3 = 27 bins; coarse on purpose, shading must not matter
MIN_COLOR_POINTS = 20

# Attachment: a constant object-to-fingertip offset, accepted only with real
# motion between the observations, real *object* motion tracking it, and a small
# residual across them.
ATTACH_MIN_EXCITATION_M = 0.02
ATTACH_MAX_RESIDUAL_M = 0.015
ATTACH_MIN_OBSERVATIONS = 2
# The object's own measured displacement must be a real displacement, not the
# noise of re-fitting a stationary object. Below this the observations carry no
# evidence that this object moved at all.
ATTACH_MIN_OBJECT_DISPLACEMENT_M = 0.010
# ...and it must track the fingertip: |Δobject − Δtip| over each observation pair.
ATTACH_MAX_CODISPLACEMENT_M = 0.010
# A held object must also track the fingertip *proportionally*: an object that
# moved 3 mm while the tip moved 40 mm is not being carried.
ATTACH_MIN_DISPLACEMENT_RATIO = 0.5
# Beyond this rotation the stored world-frame offset no longer describes the same
# grasp, so it is either mapped through the measured gripper axes or refused.
ATTACH_MAX_ORIENTATION_CHANGE_DEG = 10.0

# Physical travel is re-planned from measured telemetry at every segment boundary,
# so "arrived" needs its own threshold and the loop needs its own cap: the
# controller's own residual would otherwise keep producing a short next segment
# forever. Below SEGMENT_ARRIVED_M the remaining distance is inside what one
# waypoint of controller error can close, and further segments would be noise.
SEGMENT_ARRIVED_M = 0.005
MAX_SEGMENTS_PER_MOVE = 6

# Refinement inside one fg_run stage.
MAX_CORRECTIONS = 2
PROGRESS_FRACTION = 0.20
DEFAULT_TOLERANCE_M = 0.01
MAX_STAGES = 8
MAX_WAYPOINT_BUDGET = 24
DEFAULT_WAYPOINT_BUDGET = 12
# Whole-request wall clock. Held well under the harness's 180 s MCP tool timeout
# (codex_harness.codex_mcp_overrides) so the executor stops itself and reports
# what it did, instead of the transport cancelling a call whose later stages are
# still driving the robot.
DEFAULT_DEADLINE_S = 110.0
MAX_DEADLINE_S = 150.0

LIMITS = (
    "Fits use the currently visible, downsampled RGB-D cloud only. Residuals "
    "describe how well the declared shape explains the returns that were seen, "
    "not how close the fit is to the real object: an occluded or clipped surface "
    "can fit well and still be wrong. No collision proof, no contact dynamics, "
    "no verified object identity."
)


# ── browser-side extraction (evaluated by the MCP server) ────────────────────

# Project every cloud point to viewport pixels and keep the ones inside the
# model's region. The region test happens in the browser so only the kept points
# cross the boundary; a whole 100k cloud per object would not.
#
# This is the change that makes a card an object rather than a neighbourhood: the
# old path ray-cast one pixel and averaged a sphere around the hit, so anything
# within the radius — table, neighbouring object — entered the mean. Here a point
# outside the drawn region cannot contribute however near it is in 3-D.
REGION_POINTS_JS = """args => {
    if (typeof pcGeo === 'undefined' || !pcGeo || !pcGeo.attributes.position)
        return null;
    if (typeof camera === 'undefined' || !camera || typeof renderer === 'undefined')
        return null;
    const pos = pcGeo.attributes.position.array;
    const col = pcGeo.attributes.color ? pcGeo.attributes.color.array : null;
    const n = pos.length / 3;
    // The canvas contract: pixels are CLIENT coordinates (getBoundingClientRect
    // + project), the same space page.mouse and the crosshair overlay use, so a
    // region drawn from a screenshot lands where the model saw the object. The
    // camera-feed path below is a DIFFERENT space (that feed's own K/E, in feed
    // pixels); the two never mix, and each returns the viewport it resolved
    // against so the caller converts fractions with the right rect.
    const rect = renderer.domElement.getBoundingClientRect();
    const reg = args.region;
    // Even-odd ray crossing; a polygon is given in viewport pixels.
    const inPoly = (x, y, pts) => {
        let inside = false;
        for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
            const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
            if (((yi > y) !== (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
        }
        return inside;
    };
    const v = new THREE.Vector3();
    const kept = [];
    let projected = 0, behind = 0;
    for (let i = 0; i < n; i++) {
        v.set(pos[i*3], pos[i*3+1], pos[i*3+2]).project(camera);
        if (!isFinite(v.x) || !isFinite(v.y)) continue;
        // Outside the depth range means behind the camera or past the far plane;
        // such a point has no meaningful pixel and must not be region-tested.
        if (v.z < -1 || v.z > 1) { behind++; continue; }
        const px = rect.left + (v.x * 0.5 + 0.5) * rect.width;
        const py = rect.top + (-v.y * 0.5 + 0.5) * rect.height;
        projected++;
        let hit = false;
        if (reg.kind === 'box')
            hit = px >= reg.x0 && px <= reg.x1 && py >= reg.y0 && py <= reg.y1;
        else if (reg.kind === 'point') {
            const dx = px - reg.x, dy = py - reg.y;
            hit = dx*dx + dy*dy <= reg.radius_px * reg.radius_px;
        } else hit = inPoly(px, py, reg.points);
        if (!hit) continue;
        // 8 values per point: xyz, pixel uv, rgb. Colour comes from the cloud's
        // own per-point attribute (the RGB half of RGB-D), so appearance
        // correspondence across frames uses a measured input, not a label.
        kept.push(pos[i*3], pos[i*3+1], pos[i*3+2], px, py,
                  col ? col[i*3] : -1, col ? col[i*3+1] : -1,
                  col ? col[i*3+2] : -1);
    }
    return {kept: kept, stride: 8, cloud_points: n, projected: projected,
            outside_depth_range: behind, has_color: !!col,
            viewport: {left: rect.left, top: rect.top,
                       width: rect.width, height: rect.height}};
}"""

# Points within a sphere of a robot-frame centre, for re-association. Kept
# separate from REGION_POINTS_JS because a check has no region: the model is not
# looking, and the window is the stored geometry's own scale.
CAM_REGION_POINTS_JS = """args => {
    if (typeof pcGeo === 'undefined' || !pcGeo || !pcGeo.attributes.position)
        return null;
    if (typeof camInfo === 'undefined' || !camInfo || !camInfo[args.cam])
        return null;
    const info = camInfo[args.cam];
    const K = info.K, E = info.E, size = info.img_size;
    const f = K[0], cx = K[2], cy = K[5];
    // World (robot frame) -> camera frame needs E^-1. E is a rigid transform, so
    // the inverse is R^T and -R^T t; no general matrix inverse is needed.
    const R = [[E[0], E[1], E[2]], [E[4], E[5], E[6]], [E[8], E[9], E[10]]];
    const t = [E[3], E[7], E[11]];
    const pos = pcGeo.attributes.position.array;
    const col = pcGeo.attributes.color ? pcGeo.attributes.color.array : null;
    const n = pos.length / 3;
    const zOff = args.ui_z_offset;
    const reg = args.region;
    const inPoly = (x, y, pts) => {
        let inside = false;
        for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
            const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
            if (((yi > y) !== (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
        }
        return inside;
    };
    const kept = [];
    let projected = 0, behind = 0;
    for (let i = 0; i < n; i++) {
        // UI -> robot: the inverse of the UI transform used everywhere else.
        const ux = pos[i*3], uy = pos[i*3+1], uz = pos[i*3+2];
        const wx = ux / 10, wy = -uz / 10, wz = uy / 10 + zOff;
        const dx = wx - t[0], dy = wy - t[1], dz = wz - t[2];
        const cxx = R[0][0]*dx + R[1][0]*dy + R[2][0]*dz;
        const cyy = R[0][1]*dx + R[1][1]*dy + R[2][1]*dz;
        const czz = R[0][2]*dx + R[1][2]*dy + R[2][2]*dz;
        if (!(czz > 1e-6)) { behind++; continue; }
        const px = (f * cxx / czz + cx) / size;
        const py = (f * cyy / czz + cy) / size;
        if (!isFinite(px) || !isFinite(py)) continue;
        if (px < 0 || px > 1 || py < 0 || py > 1) continue;
        projected++;
        let hit = false;
        if (reg.kind === 'box')
            hit = px >= reg.u0 && px <= reg.u1 && py >= reg.v0 && py <= reg.v1;
        else if (reg.kind === 'point') {
            const ddx = (px - reg.u) * size, ddy = (py - reg.v) * size;
            hit = ddx*ddx + ddy*ddy <= reg.radius_px * reg.radius_px;
        } else hit = inPoly(px, py, reg.points);
        if (!hit) continue;
        kept.push(ux, uy, uz, px * size, py * size,
                  col ? col[i*3] : -1, col ? col[i*3+1] : -1,
                  col ? col[i*3+2] : -1);
    }
    // Feed pixels, resolved against this camera's own K/E and img_size — NOT the
    // canvas rect. Fractions mean the same thing on both surfaces; pixels do not.
    return {kept: kept, stride: 8, cloud_points: n, projected: projected,
            outside_depth_range: behind, has_color: !!col,
            viewport: {left: 0, top: 0, width: size, height: size}};
}"""

WINDOW_POINTS_JS = """args => {
    if (typeof pcGeo === 'undefined' || !pcGeo || !pcGeo.attributes.position)
        return null;
    const pos = pcGeo.attributes.position.array;
    const col = pcGeo.attributes.color ? pcGeo.attributes.color.array : null;
    const n = pos.length / 3;
    const [cx, cy, cz] = args.center;
    const r2 = args.radius * args.radius;
    const kept = [];
    for (let i = 0; i < n; i++) {
        const dx = pos[i*3] - cx, dy = pos[i*3+1] - cy, dz = pos[i*3+2] - cz;
        if (dx*dx + dy*dy + dz*dz <= r2)
            kept.push(pos[i*3], pos[i*3+1], pos[i*3+2],
                      col ? col[i*3] : -1, col ? col[i*3+1] : -1,
                      col ? col[i*3+2] : -1);
    }
    return {kept: kept, stride: 6, cloud_points: n, has_color: !!col};
}"""

# Draw the fit over the canvas: the kept cluster's projected extent, the fitted
# ring/disc circle, and the plane normal. Overlay only — it mutates no scene
# object, sends nothing on the websocket and touches no button.
OVERLAY_JS = """args => {
    const id = '__fgOverlay';
    let svg = document.getElementById(id);
    if (svg) svg.remove();
    if (!args || !args.shapes || !args.shapes.length) return {drawn: 0};
    const NS = 'http://www.w3.org/2000/svg';
    svg = document.createElementNS(NS, 'svg');
    svg.id = id;
    Object.assign(svg.style, {position: 'fixed', left: '0', top: '0',
        width: '100vw', height: '100vh', pointerEvents: 'none', zIndex: 99999});
    let drawn = 0;
    for (const s of args.shapes) {
        if (s.kind === 'polyline' && s.points.length > 1) {
            const el = document.createElementNS(NS, 'polyline');
            el.setAttribute('points', s.points.map(p => p.join(',')).join(' '));
            el.setAttribute('fill', 'none');
            el.setAttribute('stroke', s.color || '#00ff88');
            el.setAttribute('stroke-width', s.width || 2);
            if (s.dash) el.setAttribute('stroke-dasharray', s.dash);
            svg.appendChild(el); drawn++;
        } else if (s.kind === 'circle') {
            const el = document.createElementNS(NS, 'circle');
            el.setAttribute('cx', s.center[0]); el.setAttribute('cy', s.center[1]);
            el.setAttribute('r', s.radius_px);
            el.setAttribute('fill', 'none');
            el.setAttribute('stroke', s.color || '#ffcc00');
            el.setAttribute('stroke-width', s.width || 2);
            svg.appendChild(el); drawn++;
        } else if (s.kind === 'text') {
            const el = document.createElementNS(NS, 'text');
            el.setAttribute('x', s.at[0]); el.setAttribute('y', s.at[1]);
            el.setAttribute('fill', s.color || '#ffffff');
            el.setAttribute('font-size', s.size || 13);
            el.setAttribute('font-family', 'monospace');
            el.textContent = s.text;
            svg.appendChild(el); drawn++;
        }
    }
    document.body.appendChild(svg);
    return {drawn: drawn};
}"""

CLEAR_OVERLAY_JS = """() => {
    const el = document.getElementById('__fgOverlay');
    if (el) el.remove();
    return true;
}"""


# ── request validation ───────────────────────────────────────────────────────

class Rejected(ValueError):
    """A request that must not touch the browser or the robot at all."""


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise Rejected(f"{name} must be a finite number")
    return float(value)


def _fraction(value, name):
    x = _finite(value, name)
    if not 0.0 <= x <= 1.0:
        raise Rejected(f"{name} must be within [0, 1]")
    return x


def validate_region(region):
    """Normalize one model-declared region, in (u, v) fractions.

    Pixels are resolved later against the live viewport: a region validated in
    fractions stays meaningful if the window is a different size than the model
    assumed, and cannot silently address pixels outside the canvas.
    """
    if not isinstance(region, dict) or "kind" not in region:
        raise Rejected("region must be an object with a kind")
    kind = region["kind"]
    if kind == "box":
        extra = set(region) - {"kind", "u0", "v0", "u1", "v1"}
        if extra:
            raise Rejected(f"box region has unexpected fields: {sorted(extra)}")
        u0, v0 = _fraction(region.get("u0"), "u0"), _fraction(region.get("v0"), "v0")
        u1, v1 = _fraction(region.get("u1"), "u1"), _fraction(region.get("v1"), "v1")
        if u1 - u0 < 0.01 or v1 - v0 < 0.01:
            raise Rejected("box region must have u1>u0 and v1>v0 by at least 0.01")
        return {"kind": "box", "u0": u0, "v0": v0, "u1": u1, "v1": v1}
    if kind == "point":
        extra = set(region) - {"kind", "u", "v", "radius_px"}
        if extra:
            raise Rejected(f"point region has unexpected fields: {sorted(extra)}")
        radius = _finite(region.get("radius_px", 40), "radius_px")
        if not 8.0 <= radius <= 200.0:
            raise Rejected("radius_px must be within [8, 200]")
        return {"kind": "point", "u": _fraction(region.get("u"), "u"),
                "v": _fraction(region.get("v"), "v"), "radius_px": radius}
    if kind == "polygon":
        extra = set(region) - {"kind", "points"}
        if extra:
            raise Rejected(f"polygon region has unexpected fields: {sorted(extra)}")
        pts = region.get("points")
        if not isinstance(pts, list) or not 3 <= len(pts) <= 12:
            raise Rejected("polygon region needs 3 to 12 [u, v] points")
        out = []
        for i, p in enumerate(pts):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise Rejected(f"polygon point {i} must be [u, v]")
            out.append([_fraction(p[0], f"polygon[{i}].u"),
                        _fraction(p[1], f"polygon[{i}].v")])
        return {"kind": "polygon", "points": out}
    raise Rejected("region kind must be 'box', 'point' or 'polygon'")


def region_to_pixels(region, viewport):
    """Resolve a fraction region against the live viewport rect."""
    left, top = viewport["left"], viewport["top"]
    w, h = viewport["width"], viewport["height"]
    if region["kind"] == "box":
        return {"kind": "box", "x0": left + region["u0"] * w, "y0": top + region["v0"] * h,
                "x1": left + region["u1"] * w, "y1": top + region["v1"] * h}
    if region["kind"] == "point":
        return {"kind": "point", "x": left + region["u"] * w,
                "y": top + region["v"] * h, "radius_px": region["radius_px"]}
    return {"kind": "polygon",
            "points": [[left + u * w, top + v * h] for u, v in region["points"]]}


def validate_bind(arguments):
    """Validate a whole fg_bind request before any browser interaction."""
    if not isinstance(arguments, dict):
        raise Rejected("arguments must be an object")
    extra = set(arguments) - {"surface", "objects"}
    if extra:
        raise Rejected(f"unexpected fields: {sorted(extra)}")
    surface = arguments.get("surface", "canvas")
    if surface not in ("canvas", "agentview", "wrist"):
        raise Rejected("surface must be 'canvas', 'agentview' or 'wrist'")
    objects = arguments.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= 4:
        raise Rejected("objects must be a list of 1 to 4 entries")
    out = []
    names = set()
    for i, spec in enumerate(objects):
        if not isinstance(spec, dict):
            raise Rejected(f"objects[{i}] must be an object")
        extra = set(spec) - {"name", "region", "shape"}
        if extra:
            raise Rejected(f"objects[{i}] has unexpected fields: {sorted(extra)}")
        name = spec.get("name")
        if not isinstance(name, str) or not 1 <= len(name) <= 40:
            raise Rejected(f"objects[{i}].name must be a string of 1..40 characters")
        if name in names:
            raise Rejected(f"duplicate object name {name!r}")
        names.add(name)
        shape = spec.get("shape", "blob")
        if shape not in SHAPES:
            raise Rejected(f"objects[{i}].shape must be one of {list(SHAPES)}")
        out.append({"name": name, "shape": shape,
                    "region": validate_region(spec.get("region"))})
    return {"surface": surface, "objects": out}


# ── support plane, clustering ────────────────────────────────────────────────

def split_support(points, *, band_m=SUPPORT_BAND_M):
    """Separate the support surface (table) from what stands on it.

    The table is the lowest *populated* z band, not simply the minimum z: a single
    low outlier would otherwise define the surface and lift the whole split.  The
    band is found by histogramming z at band resolution and taking the lowest bin
    whose population is at least a tenth of the fullest bin, which ignores stray
    returns without needing an outlier model.
    """
    if not points:
        return {"support": [], "object": [], "support_z": None,
                "support_fraction": 0.0}
    zs = [p[2] for p in points]
    lo, hi = min(zs), max(zs)
    if hi - lo <= band_m:
        # Everything is within one band: this region is a flat surface, and
        # calling part of it "object" would invent structure that is not there.
        return {"support": list(points), "object": [], "support_z": round(lo, 4),
                "support_fraction": 1.0}
    nbins = max(1, int(math.ceil((hi - lo) / band_m)))
    counts = [0] * nbins
    for z in zs:
        counts[min(nbins - 1, int((z - lo) / band_m))] += 1
    threshold = max(counts) * 0.1
    base = next((i for i, c in enumerate(counts) if c >= threshold), 0)
    support_z = lo + base * band_m
    cut = support_z + band_m
    support = [p for p in points if p[2] <= cut]
    obj = [p for p in points if p[2] > cut]
    return {"support": support, "object": obj, "support_z": round(support_z, 4),
            "support_fraction": round(len(support) / len(points), 3)}


def cluster_points(points, *, radius_m=CLUSTER_RADIUS_M):
    """Single-link clusters at ``radius_m``, largest first.

    A uniform grid of cell size ``radius_m`` bounds the neighbour search to the 27
    adjacent cells, so this stays linear in the point count for the region-sized
    inputs it sees. Two returns further apart than the radius are never joined,
    which is what keeps a bowl and an adjacent plate separate.
    """
    if not points:
        return []
    cells: dict[tuple, list[int]] = {}
    for i, p in enumerate(points):
        key = (int(math.floor(p[0] / radius_m)), int(math.floor(p[1] / radius_m)),
               int(math.floor(p[2] / radius_m)))
        cells.setdefault(key, []).append(i)
    r2 = radius_m * radius_m
    seen = [False] * len(points)
    clusters = []
    for start in range(len(points)):
        if seen[start]:
            continue
        seen[start] = True
        stack, members = [start], [start]
        while stack:
            i = stack.pop()
            pi = points[i]
            ci = (int(math.floor(pi[0] / radius_m)), int(math.floor(pi[1] / radius_m)),
                  int(math.floor(pi[2] / radius_m)))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((ci[0] + dx, ci[1] + dy, ci[2] + dz), ()):
                            if seen[j]:
                                continue
                            pj = points[j]
                            if ((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2
                                    + (pi[2] - pj[2]) ** 2) <= r2:
                                seen[j] = True
                                stack.append(j)
                                members.append(j)
        clusters.append([points[i] for i in members])
    clusters.sort(key=len, reverse=True)
    return clusters


def _centroid(points):
    n = len(points)
    return [sum(p[i] for p in points) / n for i in range(3)]


def _aabb(points):
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return lo, hi


# ── appearance signature (the RGB half of RGB-D) ─────────────────────────────

def color_signature(colors):
    """A compact, comparable description of what a cluster looks like.

    Mean RGB plus a coarse 3x3x3 histogram. Deliberately coarse: the point is to
    tell a white plate from a red bowl across two views of the same scene, not to
    match shading, so a fine histogram would report a difference for every change
    in viewing angle. ``None`` when the cloud carries no usable colour, and the
    caller must then say so rather than treat "no evidence" as "matched".
    """
    usable = [c for c in colors
              if c is not None and len(c) == 3 and all(v >= 0.0 for v in c)]
    if len(usable) < MIN_COLOR_POINTS:
        return None
    n = len(usable)
    mean = [sum(c[i] for c in usable) / n for i in range(3)]
    hist = [0] * (COLOR_BINS ** 3)
    for c in usable:
        idx = 0
        for v in c:
            b = min(COLOR_BINS - 1, max(0, int(v * COLOR_BINS)))
            idx = idx * COLOR_BINS + b
        hist[idx] += 1
    return {"points": n, "mean_rgb": [round(v, 4) for v in mean],
            "histogram": [round(h / n, 4) for h in hist]}


def color_match(stored_sig, new_sig):
    """Compare two appearance signatures. Never invents evidence it does not have.

    Returns ``status`` in ``matched`` / ``mismatch`` / ``unavailable``. The last is
    not a pass: a caller that requires appearance correspondence must treat it as a
    missing check and say which evidence it did have instead.
    """
    if not stored_sig or not new_sig:
        return {"status": "unavailable",
                "reason": "the cloud carries no per-point colour for one of the views"}
    distance = math.sqrt(sum((a - b) ** 2 for a, b
                             in zip(stored_sig["mean_rgb"], new_sig["mean_rgb"])))
    overlap = sum(min(a, b) for a, b in zip(stored_sig["histogram"],
                                            new_sig["histogram"]))
    out = {"mean_rgb_distance": round(distance, 4),
           "histogram_overlap": round(overlap, 4),
           "gates": {"max_mean_rgb_distance": CHECK_MAX_COLOR_DISTANCE,
                     "min_histogram_overlap": CHECK_MIN_COLOR_OVERLAP}}
    if distance > CHECK_MAX_COLOR_DISTANCE or overlap < CHECK_MIN_COLOR_OVERLAP:
        out["status"] = "mismatch"
        out["reason"] = (f"appearance differs: mean colour distance "
                         f"{distance:.3f} (gate {CHECK_MAX_COLOR_DISTANCE:g}), "
                         f"histogram overlap {overlap:.3f} (gate "
                         f"{CHECK_MIN_COLOR_OVERLAP:g})")
        return out
    out["status"] = "matched"
    return out


def size_match(stored, candidate):
    """Whether a re-fit is the same *size* as the stored reference.

    Compares the fitted radius when both have one, and the horizontal footprint
    otherwise. A tolerance that is absolute-plus-relative keeps this meaningful
    across a 3 cm rim and a 15 cm plate. Returns ``status`` matched/mismatch, or
    ``unavailable`` when neither view supports a size at all.
    """
    pairs = []
    if stored.get("radius_m") is not None and candidate.get("radius_m") is not None:
        pairs.append(("radius_m", float(stored["radius_m"]),
                      float(candidate["radius_m"])))
    se, ce = stored.get("extent_m") or {}, candidate.get("extent_m") or {}
    for axis in ("x", "y"):
        if se.get(axis) is not None and ce.get(axis) is not None:
            pairs.append((f"extent_{axis}", float(se[axis]), float(ce[axis])))
    if not pairs:
        return {"status": "unavailable",
                "reason": "neither the stored reference nor the re-fit has a size"}
    worst = None
    for key, old, new in pairs:
        tolerance = max(CHECK_MAX_SIZE_CHANGE_M,
                        abs(old) * CHECK_MAX_SIZE_CHANGE_FRACTION)
        change = abs(new - old)
        entry = {"field": key, "stored": round(old, 4), "measured": round(new, 4),
                 "change_m": round(new - old, 4), "tolerance_m": round(tolerance, 4)}
        if change > tolerance and (worst is None or change > worst["change"]):
            worst = {"change": change, "entry": entry}
    out = {"compared": [{"field": k, "stored": round(o, 4), "measured": round(n, 4),
                         "change_m": round(n - o, 4)} for k, o, n in pairs]}
    if worst:
        out["status"] = "mismatch"
        out["worst"] = worst["entry"]
        out["reason"] = (f"{worst['entry']['field']} changed by "
                         f"{worst['entry']['change_m']:+.4f} m, beyond the "
                         f"{worst['entry']['tolerance_m']:.4f} m tolerance: this is a "
                         f"differently sized object, not the stored one")
        return out
    out["status"] = "matched"
    return out


def _solve3(a, b):
    """Gaussian elimination with partial pivoting; None when near-singular."""
    m = [list(row) + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


# ── shape fits ───────────────────────────────────────────────────────────────

def bearing_coverage_deg(points, center):
    """How much of a full turn the returns actually span around ``center``.

    A circle fitted to a 90-degree arc puts its centre far outside the observed
    data — an extrapolation that looks like a measurement. Coverage is the number
    the caller gates on to avoid that. Computed as 360 minus the largest angular
    gap, so a nearly-closed ring reads high and an arc reads low, regardless of how
    the samples are distributed within the covered part.
    """
    if len(points) < 3:
        return 0.0
    angles = sorted(math.atan2(p[1] - center[1], p[0] - center[0]) for p in points)
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    gaps.append(angles[0] + 2 * math.pi - angles[-1])
    return round(360.0 - math.degrees(max(gaps)), 1)


def fit_circle_xy(points):
    """Algebraic (Kasa) circle fit in the horizontal plane.

    Linear least squares on x^2+y^2 = 2ax + 2by + c, so there is no iteration to
    diverge and no initial guess to bias the result. The residual returned is the
    RMS *radial* deviation, which is the quantity that matters for a rim: it says
    how far the returns sit from the fitted circle, in metres.
    """
    n = len(points)
    if n < 3:
        return None
    sx = sy = sxx = syy = sxy = sz = szx = szy = 0.0
    for p in points:
        x, y = p[0], p[1]
        z = x * x + y * y
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y
        sz += z; szx += z * x; szy += z * y
    sol = _solve3([[2 * sxx, 2 * sxy, sx], [2 * sxy, 2 * syy, sy], [2 * sx, 2 * sy, n]],
                  [szx, szy, sz])
    if sol is None:
        return None
    a, b, c = sol
    inner = a * a + b * b + c
    if inner <= 0:
        return None
    radius = math.sqrt(inner)
    residuals = [math.hypot(p[0] - a, p[1] - b) - radius for p in points]
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    return {"center_xy": [a, b], "radius_m": radius, "residual_rms_m": rms,
            "coverage_deg": bearing_coverage_deg(points, [a, b])}


def fit_plane(points):
    """Least-squares plane through the points, normal signed upward.

    The normal is the smallest-eigenvalue direction of the covariance, which is the
    total-least-squares plane; fitting z as a function of (x, y) instead would fail
    on a vertical patch. Residual is RMS orthogonal distance.
    """
    n = len(points)
    if n < 3:
        return None
    c = _centroid(points)
    cov = [[sum((p[i] - c[i]) * (p[j] - c[j]) for p in points) / n
            for j in range(3)] for i in range(3)]
    values, vectors = geom.jacobi_eigh(cov)
    normal = vectors[2]
    length = math.sqrt(sum(x * x for x in normal))
    if length < 1e-12:
        return None
    normal = [x / length for x in normal]
    if normal[2] < 0:  # a plane has no intrinsic side; report the upward one
        normal = [-x for x in normal]
    residuals = [sum((p[i] - c[i]) * normal[i] for i in range(3)) for p in points]
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    return {"center": c, "normal": normal, "residual_rms_m": rms,
            "flatness_eigenvalue": values[2]}


def _band(points, *, fraction, sign):
    """The extreme ``fraction`` of the cluster by height, closed over ties.

    A count-only cut is wrong for a flat top: every return sits at the same
    height, so which quarter of them the band keeps is decided by sort order, and
    what comes back is a contiguous arc of a plate that then fails bearing
    coverage as if the plate had been half occluded. So the count is a floor, and
    the band then extends through everything within BAND_TIE_M of the last return
    kept — a flat top comes back whole, while a tall object gains at most a few
    millimetres of depth.
    """
    if not points:
        return []
    ordered = sorted(points, key=lambda p: sign * p[2], reverse=True)
    count = min(len(ordered),
                max(MIN_BAND_POINTS, int(round(len(ordered) * fraction))))
    cutoff = sign * ordered[count - 1][2] - BAND_TIE_M
    while count < len(ordered) and sign * ordered[count][2] >= cutoff:
        count += 1
    return ordered[:count]


def top_band(points, *, fraction=RING_TOP_FRACTION):
    """The upper ``fraction`` of the cluster by height, where a rim is.

    Averaging a whole bowl gives a point inside the bowl's volume, which is not
    the rim and not the base; the two have to be separated to be usable. The band
    is always at least MIN_BAND_POINTS deep when the cluster allows it, so a
    sparse cluster does not fit a circle to three returns.
    """
    return _band(points, fraction=fraction, sign=1.0)


def bottom_band(points, *, fraction=RING_TOP_FRACTION):
    return _band(points, fraction=fraction, sign=-1.0)


# ── object cards ─────────────────────────────────────────────────────────────

def _r(p, digits=4):
    return {k: round(float(x), digits) for k, x in zip("xyz", p)}


def _key3(p):
    """A hashable key for one cloud point, for pairing points back to colours."""
    return (round(p[0], 6), round(p[1], 6), round(p[2], 6))


def _touches_border(pixels, region_px, *, margin=3.0):
    """Whether the kept cluster reaches the region boundary.

    A clipped cluster's extent is a lower bound: the object continues outside the
    drawn region, so its radius and AABB understate it. Reported rather than
    corrected, because nothing local can recover what was not selected.
    """
    if not pixels:
        return False
    if region_px["kind"] == "box":
        return any(x <= region_px["x0"] + margin or x >= region_px["x1"] - margin
                   or y <= region_px["y0"] + margin or y >= region_px["y1"] - margin
                   for x, y in pixels)
    if region_px["kind"] == "point":
        r = region_px["radius_px"] - margin
        return any(math.hypot(x - region_px["x"], y - region_px["y"]) >= r
                   for x, y in pixels)
    xs = [p[0] for p in region_px["points"]]
    ys = [p[1] for p in region_px["points"]]
    return any(x <= min(xs) + margin or x >= max(xs) - margin
               or y <= min(ys) + margin or y >= max(ys) - margin
               for x, y in pixels)


def upright_applicable(band):
    """Whether a horizontal (XY) circle fit is applicable to these returns.

    A Kasa fit in XY assumes the rim lies in a horizontal plane. When it does not —
    a tilted bowl, a wall, a fit that has caught a sloping surface — the fitted
    centre and radius are the projection of a tilted circle and understate its
    size in one direction. The assumption is therefore CHECKED, not just declared:
    a plane through the band must have a near-vertical normal. Returns
    ``(ok, detail)``; ``ok`` is None when there are too few returns to tell.
    """
    plane = fit_plane(band) if len(band) >= MIN_BAND_POINTS else None
    if plane is None:
        return None, {"reason": "too few returns in the band to test the assumption"}
    # Signed upward already, so the z component is the cosine to world up.
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, plane["normal"][2]))))
    detail = {"band_plane_tilt_deg": round(tilt, 1),
              "gate_deg": RING_MAX_TILT_DEG,
              "band_plane_residual_rms_m": round(plane["residual_rms_m"], 5)}
    return tilt <= RING_MAX_TILT_DEG, detail


def build_card(*, name, shape, points, pixels, region_px, support, clusters,
               from_frame, surface, colors=None, support_colors=None):
    """One object card from a region's kept points. Never raises on bad geometry.

    ``valid`` is the field the executor gates on. Everything the fit could not
    establish is null with a reason; a card whose residual, support or coverage
    fails its shape's gate is returned invalid *with* its numbers, so the model can
    see why and re-draw the region rather than guess.
    """
    card = {
        "name": name, "name_note": "your label; not a verified object identity",
        "shape": shape, "frame": "robot", "from_frame": from_frame,
        "surface": surface,
        "fit": {"region_points": len(points) + len(support),
                "object_points": len(points),
                "support_points": len(support),
                "support_fraction": None, "residual_rms_m": None,
                "bearing_coverage_deg": None,
                "clipped_by_region": _touches_border(pixels, region_px),
                "cluster_count": len(clusters),
                "second_cluster_points": len(clusters[1]) if len(clusters) > 1 else 0},
        "center": None, "base_center": None, "top_z": None,
        "radius_m": None, "height_m": None, "extent_m": None,
        "up": None, "up_source": None, "principal_axis": None,
        "keypoints": {}, "grasp_candidates": [],
        "appearance": None,
        "valid": False, "reasons": [],
    }
    reasons = card["reasons"]
    # Appearance is stored on the card so a later re-association has something to
    # correspond against. Computed from the returns actually kept for the fit.
    card["appearance"] = color_signature(colors or [])
    if card["appearance"] is None:
        card["fit"]["appearance_note"] = (
            "no per-point colour available for these returns, so a later re-check "
            "cannot use appearance correspondence")
    if len(points) < MIN_OBJECT_POINTS:
        reasons.append(f"only {len(points)} object returns in the region, below the "
                       f"{MIN_OBJECT_POINTS}-point gate")
        if shape != "plane_patch":
            return card
    if len(clusters) > 1:
        ratio = len(clusters[1]) / max(1, len(clusters[0]))
        if ratio >= CHECK_AMBIGUOUS_RATIO:
            reasons.append(f"a second cluster of comparable size ({len(clusters[1])} "
                           f"vs {len(clusters[0])} points) is inside the region, so "
                           f"which one you meant is ambiguous; draw a tighter region")
    lo, hi = _aabb(points) if points else ([None] * 3, [None] * 3)
    if points:
        card["extent_m"] = {k: round(hi[i] - lo[i], 4) for i, k in enumerate("xyz")}
        card["top_z"] = round(hi[2], 4)

    if shape == "plane_patch":
        # A plane_patch is the one shape that may legitimately BE the support
        # surface: the table, or a flat top that split_support put in the support
        # band. So when the object band is too thin, fall back to the support
        # returns and *drop the object-count reason*, which described a different
        # shape's precondition. Leaving it in was how a plane_patch fitted from a
        # perfectly good tabletop came back permanently invalid because
        # ``object`` happened to be empty.
        target = points if len(points) >= MIN_BAND_POINTS else support
        used_support = target is support
        if used_support:
            reasons[:] = [r for r in reasons if "object returns in the region" not in r]
            card["fit"]["fitted_from"] = "support_band"
            card["fit"]["support_band_note"] = (
                "fitted from the support returns: this patch is the surface "
                "split_support identified, not something standing on it")
        else:
            card["fit"]["fitted_from"] = "object_band"
        plane = fit_plane(target) if len(target) >= MIN_BAND_POINTS else None
        if plane is None:
            reasons.append(f"too few returns for a plane fit ({len(target)} available, "
                           f"{MIN_BAND_POINTS} needed)")
            return card
        card["fit"]["plane_points"] = len(target)
        card["fit"]["residual_rms_m"] = round(plane["residual_rms_m"], 5)
        card["center"] = _r(plane["center"])
        card["base_center"] = card["center"]
        card["up"] = _r(plane["normal"], 9)
        card["up_source"] = "fitted_plane_normal"
        card["keypoints"] = {"patch_center": card["center"]}
        if used_support and support_colors:
            # The appearance must describe the returns that were actually fitted.
            card["appearance"] = color_signature(support_colors) or card["appearance"]
        gate = RESIDUAL_GATE_M["plane_patch"]
        if plane["residual_rms_m"] > gate:
            reasons.append(f"plane residual {plane['residual_rms_m']:.4f} m exceeds "
                           f"the {gate:g} m gate: these returns are not flat")
        card["valid"] = not reasons
        return card

    if shape == "blob":
        if not points:
            reasons.append("no object returns above the support surface")
            return card
        c = _centroid(points)
        card["center"] = _r(c)
        card["base_center"] = _r([c[0], c[1], lo[2]])
        card["height_m"] = round(hi[2] - lo[2], 4)
        card["up"] = _r([0, 0, 1], 9)
        card["up_source"] = "assumed_world_up_blob_has_no_fitted_normal"
        card["keypoints"] = {"centroid": card["center"],
                            "highest_point": _r(max(points, key=lambda p: p[2])),
                            "lowest_point": _r(min(points, key=lambda p: p[2]))}
        summary = geom.summarize_points(points, radius_m=0.0, support=len(points),
                                        stride=1, source="fast_geometry_blob")
        card["principal_axis"] = summary.get("principal_axis")
        # A blob claims a centroid, an AABB and possibly an axis. There is no
        # residual to gate, so validity rests on support alone — deliberately, so
        # the honest fallback stays usable when a shape fit is not warranted.
        card["valid"] = len(points) >= MIN_OBJECT_POINTS and not reasons
        return card

    # ring / disc
    band = top_band(points)
    circle = fit_circle_xy(band) if len(band) >= 3 else None
    if circle is None:
        reasons.append("could not fit a circle to the upper band of returns")
        return card
    card["fit"]["residual_rms_m"] = round(circle["residual_rms_m"], 5)
    card["fit"]["bearing_coverage_deg"] = circle["coverage_deg"]
    band_z = sum(p[2] for p in band) / len(band)
    card["center"] = _r([circle["center_xy"][0], circle["center_xy"][1], band_z])
    card["radius_m"] = round(circle["radius_m"], 4)
    card["height_m"] = round(hi[2] - lo[2], 4)
    base = bottom_band(points)
    card["base_center"] = _r([sum(p[0] for p in base) / len(base),
                             sum(p[1] for p in base) / len(base), lo[2]])
    # The XY circle fit presumes an upright object. Say so, and check it: `up` here
    # is an ASSUMPTION that was tested, never a normal this fit measured.
    card["up"] = _r([0, 0, 1], 9)
    card["up_source"] = "assumed_world_up_upright_prior_checked_against_band_plane"
    upright_ok, upright_detail = upright_applicable(band)
    card["fit"]["upright_assumption"] = dict(
        upright_detail, applicable=upright_ok,
        note=("a horizontal circle fit only describes an upright object; this is "
              "the check that the returns are consistent with that prior, not a "
              "fitted 3-D orientation"))
    if upright_ok is False:
        reasons.append(
            f"the upper band's own plane is tilted "
            f"{upright_detail['band_plane_tilt_deg']:.0f} deg from horizontal "
            f"(gate {RING_MAX_TILT_DEG:g}), so a horizontal circle fit is not "
            f"applicable: its centre and radius are the projection of a tilted "
            f"circle, not measurements of it")
    elif upright_ok is None:
        reasons.append("too few returns to test whether the upright assumption "
                       "behind the horizontal circle fit applies")
    cx, cy = circle["center_xy"]
    r = circle["radius_m"]
    card["keypoints"] = {
        "rim_center": card["center"],
        "rim_x_plus": _r([cx + r, cy, band_z]),
        "rim_x_minus": _r([cx - r, cy, band_z]),
        "rim_y_plus": _r([cx, cy + r, band_z]),
        "rim_y_minus": _r([cx, cy - r, band_z]),
        "base_center": card["base_center"],
    }
    # Grasp candidates are rim points with the jaw axis across the rim (radial),
    # so closing brings the fingers onto the wall rather than along it. They are
    # geometric offers, not a graspability claim: nothing here models contact.
    for key, axis in (("rim_x+", [1.0, 0.0, 0.0]), ("rim_x-", [-1.0, 0.0, 0.0]),
                      ("rim_y+", [0.0, 1.0, 0.0]), ("rim_y-", [0.0, -1.0, 0.0])):
        point = [cx + axis[0] * r, cy + axis[1] * r, band_z]
        card["grasp_candidates"].append({
            "id": key, "point": _r(point),
            "opening_axis": _r(axis, 9),
            "approach": _r([0.0, 0.0, -1.0], 9),
            "note": "from the fitted rim; no contact or force model"})
    gate = RESIDUAL_GATE_M[shape]
    if circle["residual_rms_m"] > gate:
        reasons.append(f"circle residual {circle['residual_rms_m']:.4f} m exceeds the "
                       f"{gate:g} m gate: the upper band is not a {shape}")
    if circle["coverage_deg"] < RING_MIN_BEARING_DEG:
        reasons.append(f"the returns span only {circle['coverage_deg']:.0f} deg of "
                       f"bearing (gate {RING_MIN_BEARING_DEG:g}), so the fitted "
                       f"centre is an extrapolation from an arc, not a measurement")
    if card["fit"]["clipped_by_region"]:
        reasons.append("the cluster reaches the region border, so its extent is a "
                       "lower bound; radius and centre may be understated")
    if len(points) < MIN_OBJECT_POINTS:
        pass  # already recorded above
    card["valid"] = not reasons
    return card


# ── re-association across frames ─────────────────────────────────────────────

UNKNOWN_REASONS = ("no_paired_frame", "same_frame_no_new_evidence",
                   "insufficient_support", "fit_degraded", "shift_exceeds_window",
                   "ambiguous_two_candidates", "occluded", "no_stored_geometry",
                   "size_mismatch", "appearance_mismatch",
                   "no_correspondence_evidence")


def check_window_radius(card):
    """How far from the last known centre a re-fit may look."""
    scale = card.get("radius_m")
    if scale is None:
        extent = card.get("extent_m") or {}
        spans = [v for v in (extent.get("x"), extent.get("y")) if v is not None]
        scale = (max(spans) / 2.0) if spans else 0.05
    return round(2.0 * float(scale) + CHECK_WINDOW_EXTRA_M, 4)


def _evaluate_candidate(stored, cluster, *, split, by_point, from_frame, prior,
                        anchor, window, expected_shift, rival_points):
    """Judge ONE window cluster against the stored card. Returns a full outcome.

    Split out of ``reassociate`` because the window may hold more than one thing:
    once a held bowl hangs over the plate it is being placed on, the plate's own
    search window contains both objects, so *which* cluster to test cannot be
    decided by size — it has to be decided by correspondence, which means testing
    each candidate the same way and letting the gates pick.
    """
    out = {"status": "unknown", "reason": None, "object_points": len(cluster)}
    card = build_card(name=stored.get("name"), shape=stored.get("shape"),
                      points=cluster, pixels=[], region_px={"kind": "box",
                      "x0": -1e9, "y0": -1e9, "x1": 1e9, "y1": 1e9},
                      support=split["support"], clusters=[cluster],
                      from_frame=from_frame, surface="window",
                      colors=[by_point.get(_key3(p)) for p in cluster],
                      support_colors=[by_point.get(_key3(p))
                                      for p in split["support"]])
    # The rival count describes the WINDOW, not this cluster, so it is restored
    # after build_card (which was handed this cluster alone so a second candidate
    # could not invalidate the fit before correspondence had chosen between them).
    card["fit"]["second_cluster_points"] = rival_points
    if card["center"] is None:
        out.update(reason="fit_degraded", fit=card["fit"],
                   fit_reasons=card["reasons"])
        return out, card
    new_center = [card["center"][k] for k in "xyz"]
    shift = math.sqrt(sum((a - b) ** 2 for a, b in zip(new_center, anchor)))
    out.update(center=card["center"], base_center=card["base_center"],
               top_z=card["top_z"], radius_m=card["radius_m"],
               keypoints=card["keypoints"], grasp_candidates=card["grasp_candidates"],
               extent_m=card["extent_m"], height_m=card["height_m"],
               fit=card["fit"], residual_rms_m=card["fit"]["residual_rms_m"],
               center_shift_m=round(math.sqrt(sum(
                   (a - prior[k]) ** 2 for a, k in zip(new_center, "xyz"))), 4),
               search_window_m=window)
    if expected_shift:
        out["prediction_error_m"] = round(shift, 4)
    if shift > window:
        out["reason"] = "shift_exceeds_window"
        return out, card
    stored_residual = (stored.get("fit") or {}).get("residual_rms_m")
    new_residual = card["fit"]["residual_rms_m"]
    if (stored_residual and new_residual
            and new_residual > max(stored_residual * CHECK_MAX_RESIDUAL_RATIO,
                                   RESIDUAL_GATE_M.get(stored.get("shape"), 1.0))):
        out["reason"] = "fit_degraded"
        return out, card
    if stored.get("radius_m") is not None and card["radius_m"] is not None:
        out["radius_change_m"] = round(card["radius_m"] - stored["radius_m"], 4)
    if not card["valid"]:
        out.update(reason="fit_degraded", fit_reasons=card["reasons"])
        return out, card

    # Size correspondence. Previously this was *reported* as radius_change_m and
    # nothing refused on it, so swapping in another circle of a different size kept
    # the stored id. It is now a gate.
    sizes = size_match(stored, card)
    out["size_match"] = sizes
    if sizes["status"] == "mismatch":
        out.update(reason="size_mismatch", message=sizes["reason"])
        return out, card

    # Appearance correspondence, from measured colour.
    appearance = color_match(stored.get("appearance"), card.get("appearance"))
    out["appearance_match"] = appearance
    if appearance["status"] == "mismatch":
        out.update(reason="appearance_mismatch", message=appearance["reason"])
        return out, card
    if appearance["status"] == "unavailable":
        # No colour to correspond on. Geometry alone may still be enough, but only
        # when it is unambiguous: no rival cluster, sizes that do correspond, and a
        # shift well inside the window. Otherwise refuse rather than let "no
        # evidence" read as "matched".
        unambiguous = (rival_points == 0 and sizes["status"] == "matched"
                       and shift <= window * 0.5)
        if not unambiguous:
            out.update(reason="no_correspondence_evidence", message=(
                "no per-point colour is available for appearance correspondence, and "
                "the geometry alone is not decisive here (rival cluster "
                f"{rival_points} points, size check {sizes['status']}, shift "
                f"{shift:.4f} m of a {window:.4f} m window). Re-bind from a view "
                "where this object is clearly separated."))
            return out, card

    out["status"] = "matched"
    out["match_evidence"] = {
        "same_shape_refitted": stored.get("shape"),
        "center_shift_m": out["center_shift_m"],
        "residual_rms_m": new_residual,
        "size_match": sizes,
        "appearance_match": appearance,
        "rival_cluster_points": rival_points,
        "checks_not_available": (["appearance"]
                                 if appearance["status"] == "unavailable" else []),
        "note": ("re-fit near the last known position with size and appearance "
                 "correspondence; evidence that this is the same local object, not "
                 "proof of identity"),
    }
    return out, card


def reassociate(stored, window_points, *, from_frame, expected_shift=None,
                window_colors=None):
    """Re-fit a stored card inside its search window on a NEW paired frame.

    Association is geometric AND appearance-based, and all of it is checked before
    the id survives:

    * the same shape re-fits, with a residual that has not degraded;
    * the size corresponds — a differently sized round thing is refused, not
      reported as a radius change (``size_mismatch``);
    * the appearance corresponds, from the cloud's own per-point colour
      (``appearance_mismatch``); when neither view carries colour, the outcome is
      ``no_correspondence_evidence`` unless the geometry alone is unambiguous, and
      the missing check is named in the evidence either way;
    * the shift is inside the window.

    Every sufficiently populated cluster in the window is judged that way, nearest
    to the anchor first, and the id survives only if *exactly one* corresponds. Two
    corresponding candidates are ``ambiguous_two_candidates``; none is the nearest
    candidate's own refusal. Picking the largest cluster and testing only that one
    was a defect, not a simplification: a placement puts the held object inside the
    target's window, where it is often the bigger cluster, and the target then lost
    its id to a ``size_mismatch`` measured against an object it is not.

    That is still weaker than tracking a verified identity, and it is reported as
    what it is. ``expected_shift`` (supplied when the object is believed attached)
    only *centres the search*: the prediction is reported next to the measurement
    and never substituted for it.
    """
    out = {"name": stored.get("name"), "shape": stored.get("shape"),
           "from_frame": from_frame, "status": "unknown", "reason": None,
           "predicted_center": _r(expected_shift) if expected_shift else None,
           "predicted_center_note": ("a search hint from the stored attachment "
                                     "offset; not an observation")
           if expected_shift else None,
           "center": None, "center_shift_m": None, "radius_change_m": None,
           "residual_rms_m": None, "prediction_error_m": None}
    prior = stored.get("center")
    if not prior:
        out["reason"] = "no_stored_geometry"
        return out
    if window_colors is not None and len(window_colors) != len(window_points):
        raise Rejected("window_colors must be parallel to window_points")
    colors = list(window_colors) if window_colors else [None] * len(window_points)
    by_point = {}
    for p, c in zip(window_points, colors):
        by_point.setdefault(_key3(p), c)
    split = split_support(window_points)
    obj = split["object"]
    clusters = cluster_points(obj)
    if not clusters or len(clusters[0]) < MIN_OBJECT_POINTS:
        out["reason"] = "insufficient_support"
        out["object_points"] = len(obj)
        return out
    anchor = expected_shift if expected_shift else [prior[k] for k in "xyz"]
    window = check_window_radius(stored)
    # Nearest-first, so "the candidate that came closest" is a defined thing to
    # report when none of them corresponds.
    populated = [c for c in clusters if len(c) >= MIN_OBJECT_POINTS]
    ranked = sorted(populated,
                    key=lambda c: sum((a - b) ** 2
                                      for a, b in zip(_centroid(c), anchor))
                    )[:MAX_MATCH_CANDIDATES]
    considered, accepted = [], []
    for cluster in ranked:
        rival = max((len(c) for c in clusters if c is not cluster), default=0)
        verdict, _card = _evaluate_candidate(
            stored, cluster, split=split, by_point=by_point, from_frame=from_frame,
            prior=prior, anchor=anchor, window=window, expected_shift=expected_shift,
            rival_points=rival)
        considered.append(verdict)
        if verdict["status"] == "matched":
            accepted.append(verdict)
    summary = [{"object_points": v.get("object_points"), "center": v.get("center"),
                "status": v["status"], "reason": v.get("reason")}
               for v in considered]
    if len(accepted) > 1:
        out.update(reason="ambiguous_two_candidates",
                   cluster_sizes=[len(c) for c in clusters[:3]],
                   candidates_considered=summary,
                   message=(f"{len(accepted)} clusters in the search window all "
                            f"correspond to this reference on shape, size and "
                            f"appearance, so which one it is cannot be decided from "
                            f"this frame"))
        return out
    chosen = accepted[0] if accepted else (considered[0] if considered else None)
    if chosen is None:
        out["reason"] = "insufficient_support"
        out["object_points"] = len(obj)
        return out
    out.update({k: v for k, v in chosen.items() if v is not None})
    out["status"] = chosen["status"]
    out["reason"] = chosen.get("reason")
    if len(considered) > 1:
        out["candidates_considered"] = summary
    return out


# ── attachment ───────────────────────────────────────────────────────────────

def fit_attachment(observations):
    """Whether the object is being carried, from measured co-displacement.

    ``observations`` are ``{"frame", "object_center", "fingertip"}`` from *distinct
    paired frames*, each ``object_center`` a real re-measurement (a predicted
    position must never be passed in here — that would make the conclusion
    circular). ``gripper_state_class`` and ``approach``/``opening`` are used when
    present.

    A constant-offset RMS alone does NOT establish attachment, and the
    counterexample is concrete: object at (0, 0) in both frames, fingertip at
    (0, 0) then (0, 0.02). The two offsets are (0, 0) and (0, -0.02), their mean is
    (0, -0.01) and the RMS spread about it is exactly 0.01 m — inside a 0.015 m
    gate. An empty close beside a stationary object would have passed. So the gates
    are, in order:

    1. the *object* must have measurably moved (``ATTACH_MIN_OBJECT_DISPLACEMENT_M``);
    2. its displacement must track the fingertip's, both as a vector difference
       (``ATTACH_MAX_CODISPLACEMENT_M``) and in magnitude
       (``ATTACH_MIN_DISPLACEMENT_RATIO``);
    3. the fingertip must have moved enough to excite the test at all;
    4. the offsets must be mutually consistent (the original RMS gate, now one
       condition among several rather than the whole test);
    5. the gripper must be *known* closed in every observation — a missing or
       "unknown" class is refused, not skipped;
    6. the gripper orientation must be measured in at least two observations and
       must not have changed materially, since a world-frame offset does not
       survive a rotated grasp and cannot be rotated without its axes.

    The counterexample fails (1) and (2) and is refused with
    ``reason_code="object_did_not_move"``.
    """
    out = {"status": "unknown", "observations": len(observations),
           "offset_m": None, "residual_m": None, "motion_excitation_m": None,
           "object_displacement_m": None, "co_displacement_error_m": None,
           "displacement_ratio": None, "reason": None, "reason_code": None,
           "gates": {"min_object_displacement_m": ATTACH_MIN_OBJECT_DISPLACEMENT_M,
                     "max_co_displacement_error_m": ATTACH_MAX_CODISPLACEMENT_M,
                     "min_displacement_ratio": ATTACH_MIN_DISPLACEMENT_RATIO,
                     "min_tip_excitation_m": ATTACH_MIN_EXCITATION_M,
                     "max_offset_residual_m": ATTACH_MAX_RESIDUAL_M},
           "note": ("attachment inferred from the object's own measured motion "
                    "tracking the fingertip's; not a contact or force model, and "
                    "not evidence about grasp quality")}

    def _fail(code, message):
        out["reason_code"] = code
        out["reason"] = message
        return out

    frames = {o.get("frame") for o in observations}
    if (len(observations) < ATTACH_MIN_OBSERVATIONS
            or len(frames) < ATTACH_MIN_OBSERVATIONS or None in frames):
        return _fail("insufficient_observations",
                     f"needs {ATTACH_MIN_OBSERVATIONS} observations from distinct "
                     f"paired frames; got {len(frames)} distinct")
    if any(o.get("object_center_is_predicted") for o in observations):
        return _fail("predicted_position_supplied",
                     "one observation carries a predicted object position rather "
                     "than a measurement; attachment cannot be established from "
                     "the prediction it would justify")

    tips = [list(o["fingertip"]) for o in observations]
    objects = [list(o["object_center"]) for o in observations]

    # The gripper must be *known* closed in EVERY observation. Earlier this
    # skipped observations that carried no class, so telemetry that failed to
    # report went in as consent: two frames of co-displacement with no gripper
    # state at all were accepted as "attached", which is the same shape of error
    # as accepting the constant-offset RMS on its own. A missing precondition is
    # not a satisfied one, so an unreported or "unknown" class refuses here.
    classes = [o.get("gripper_state_class") for o in observations]
    out["gripper_state_classes"] = classes
    missing = [i for i, c in enumerate(classes) if c is None or c == "unknown"]
    if missing:
        return _fail("gripper_state_unknown",
                     f"observations {missing} carry no usable gripper state class, so "
                     f"nothing here establishes the gripper was closed while the "
                     f"object moved with it; supply the measured class from each "
                     f"paired frame's own telemetry")
    if any(c != "closed" for c in classes):
        return _fail("gripper_not_closed",
                     f"the gripper is reported {sorted(set(classes))} across these "
                     f"observations; an object cannot be carried by an open gripper")

    # Orientation. A world-frame offset only describes the same grasp while the
    # gripper holds its orientation, so an *unmeasurable* orientation is refused
    # too: accepting it would store an offset with no axes, and every later
    # relation would reuse that world vector across whatever rotation happened.
    turn = _max_orientation_change_deg(observations)
    out["orientation_change_deg"] = turn
    if turn is None:
        return _fail("gripper_orientation_unknown",
                     "fewer than two observations carry a measured approach/opening "
                     "pair, so neither the grasp orientation nor its change is known; "
                     "the offset could not be re-expressed after any rotation")
    if turn > ATTACH_MAX_ORIENTATION_CHANGE_DEG:
        return _fail("orientation_changed",
                     f"the gripper turned {turn:.1f} deg between these observations "
                     f"(gate {ATTACH_MAX_ORIENTATION_CHANGE_DEG:g}), so a single "
                     f"world-frame object-to-tip offset does not describe both; "
                     f"re-calibrate without rotating, or hold orientation")

    # Per-pair displacements. The strongest available pair decides excitation; the
    # WORST pair decides co-displacement, so one good pair cannot cover a bad one.
    tip_disp = 0.0
    obj_disp_at_best = 0.0
    worst_codisp = 0.0
    worst_ratio = None
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            dt = [tips[j][k] - tips[i][k] for k in range(3)]
            do = [objects[j][k] - objects[i][k] for k in range(3)]
            dt_n = math.sqrt(sum(x * x for x in dt))
            do_n = math.sqrt(sum(x * x for x in do))
            if dt_n > tip_disp:
                tip_disp = dt_n
                obj_disp_at_best = do_n
            codisp = math.sqrt(sum((a - b) ** 2 for a, b in zip(do, dt)))
            worst_codisp = max(worst_codisp, codisp)
            if dt_n >= ATTACH_MIN_EXCITATION_M:
                ratio = do_n / dt_n
                worst_ratio = ratio if worst_ratio is None else min(worst_ratio, ratio)
    out["motion_excitation_m"] = round(tip_disp, 4)
    out["object_displacement_m"] = round(obj_disp_at_best, 4)
    out["co_displacement_error_m"] = round(worst_codisp, 4)
    out["displacement_ratio"] = (None if worst_ratio is None
                                 else round(worst_ratio, 3))

    offsets = [[objects[i][k] - tips[i][k] for k in range(3)]
               for i in range(len(observations))]
    mean = [sum(o[k] for o in offsets) / len(offsets) for k in range(3)]
    residual = math.sqrt(sum(sum((o[k] - mean[k]) ** 2 for k in range(3))
                             for o in offsets) / len(offsets))
    out["offset_m"] = _r(mean)
    out["residual_m"] = round(residual, 4)

    if tip_disp < ATTACH_MIN_EXCITATION_M:
        return _fail("insufficient_excitation",
                     f"the fingertip moved only {tip_disp:.4f} m between these "
                     f"observations (gate {ATTACH_MIN_EXCITATION_M:g} m), so a "
                     f"constant offset would also fit an object that never moved")
    if obj_disp_at_best < ATTACH_MIN_OBJECT_DISPLACEMENT_M:
        return _fail("object_did_not_move",
                     f"the object moved {obj_disp_at_best:.4f} m while the fingertip "
                     f"moved {tip_disp:.4f} m (object gate "
                     f"{ATTACH_MIN_OBJECT_DISPLACEMENT_M:g} m): it stayed where it "
                     f"was, so nothing is being carried — an empty close, or the "
                     f"object slipped out immediately")
    if worst_codisp > ATTACH_MAX_CODISPLACEMENT_M:
        return _fail("not_co_moving",
                     f"the object's displacement differs from the fingertip's by "
                     f"{worst_codisp:.4f} m (gate {ATTACH_MAX_CODISPLACEMENT_M:g} m): "
                     f"it is not moving rigidly with the gripper")
    if worst_ratio is not None and worst_ratio < ATTACH_MIN_DISPLACEMENT_RATIO:
        return _fail("displacement_too_small",
                     f"the object moved only {worst_ratio:.2f} of the fingertip's "
                     f"displacement (gate {ATTACH_MIN_DISPLACEMENT_RATIO:g}): it is "
                     f"being nudged or is slipping, not carried")
    if residual > ATTACH_MAX_RESIDUAL_M:
        return _fail("offset_inconsistent",
                     f"offset varies by {residual:.4f} m across the observations "
                     f"(gate {ATTACH_MAX_RESIDUAL_M:g} m): the object is not "
                     f"moving rigidly with the gripper")

    out["status"] = "attached"
    # Store the grasp orientation the offset was measured under, so a later
    # relation can either verify it still holds or rotate the offset properly.
    basis = _grasp_basis(observations[-1])
    if basis is not None:
        out["measured_under"] = {
            "approach": _r(basis["approach"], 6),
            "opening": _r(basis["opening"], 6),
            "offset_local_m": _r(_to_local(mean, basis), 4),
            "note": ("the offset expressed in the gripper's own axes; this is what "
                     "survives a rotation, the world-frame offset does not")}
    return out


def _grasp_basis(observation):
    """Right-handed gripper basis from a measured approach/opening pair, or None."""
    a, o = observation.get("approach"), observation.get("opening")
    if not a or not o:
        return None
    a = _unit3(a)
    o = _unit3(o)
    if a is None or o is None:
        return None
    # Remove any non-orthogonality rather than assuming it away.
    dot = sum(x * y for x, y in zip(a, o))
    o = _unit3([x - dot * y for x, y in zip(o, a)])
    if o is None:
        return None
    third = [a[1] * o[2] - a[2] * o[1], a[2] * o[0] - a[0] * o[2],
             a[0] * o[1] - a[1] * o[0]]
    return {"approach": a, "opening": o, "third": third}


def _unit3(v):
    try:
        vals = [float(x) for x in (v.values() if isinstance(v, dict) else v)]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or not all(math.isfinite(x) for x in vals):
        return None
    length = math.sqrt(sum(x * x for x in vals))
    if length < 1e-9:
        return None
    return [x / length for x in vals]


def _to_local(world_vector, basis):
    """Express a world-frame vector in the gripper's own axes."""
    return [sum(world_vector[i] * basis[axis][i] for i in range(3))
            for axis in ("approach", "opening", "third")]


def _to_world(local_vector, basis):
    """Inverse of ``_to_local``: back to the world frame under a new basis."""
    return [sum(local_vector[k] * basis[axis][i]
                for k, axis in enumerate(("approach", "opening", "third")))
            for i in range(3)]


def _max_orientation_change_deg(observations):
    """Largest gripper rotation across the observations, or None if unmeasurable.

    ``None`` when *any* observation lacks a measured approach/opening pair, not
    merely when fewer than two have one: dropping the unmeasured ones would report
    a small rotation across the two frames that happened to carry axes while the
    gripper may have turned in the one that did not.
    """
    bases = [_grasp_basis(o) for o in observations]
    if len(bases) < 2 or any(b is None for b in bases):
        return None
    worst = 0.0
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            trace = sum(sum(bases[i][axis][k] * bases[j][axis][k] for k in range(3))
                        for axis in ("approach", "opening", "third"))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
            worst = max(worst, angle)
    return round(worst, 2)


def offset_in_current_orientation(attachment, measured):
    """The stored object-to-tip offset, valid for the gripper's CURRENT orientation.

    Returns ``(offset, detail)``, or ``(None, detail)`` when the offset cannot be
    carried over. A world-frame offset is only reusable while the gripper holds the
    orientation it was measured under, so the rotation must be *measurable*: with
    both bases known the offset is reused inside tolerance and rotated beyond it,
    and with either basis missing it is refused. Reusing the world vector when the
    rotation is unknown is the failure source_review_notes.md names.
    """
    stored = attachment.get("measured_under") or {}
    world = [attachment["offset_m"][k] for k in "xyz"]
    now = _grasp_basis(measured or {})
    then = _grasp_basis({"approach": stored.get("approach"),
                         "opening": stored.get("opening")})
    if now is None or then is None:
        # Refused rather than reused. Reusing the world vector "in case nothing
        # rotated" is exactly the failure source_review_notes.md names: the
        # rotation is unmeasured, so the assumption that there was none is not
        # weaker evidence, it is no evidence, and the offset it licenses would be
        # reported as a measured placement.
        return None, {"mapping": "refused",
                      "reason": ("the gripper's own axes are not measured in "
                                 + ("this frame" if now is None
                                    else "the calibration") +
                                 ", so whether the grasp has rotated since the "
                                 "offset was measured is unknown and the offset "
                                 "cannot be carried over; re-observe on a paired "
                                 "frame carrying end-effector orientation, then "
                                 "re-run calibrate_attachment"),
                      "missing": ("current_gripper_axes" if now is None
                                  else "calibration_gripper_axes")}
    trace = sum(sum(now[a][k] * then[a][k] for k in range(3))
                for a in ("approach", "opening", "third"))
    turn = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
    if turn <= ATTACH_MAX_ORIENTATION_CHANGE_DEG:
        return world, {"mapping": "world_offset_reused",
                       "orientation_change_deg": round(turn, 2),
                       "note": "the gripper is within tolerance of its calibration "
                               "orientation, so the measured offset applies as is"}
    local = stored.get("offset_local_m")
    if not local:
        return None, {"mapping": "refused",
                      "orientation_change_deg": round(turn, 2),
                      "reason": (f"the gripper has turned {turn:.1f} deg since the "
                                 f"offset was measured and no local-frame offset was "
                                 f"stored, so the offset cannot be carried over; "
                                 f"re-run calibrate_attachment")}
    rotated = _to_world([local[k] for k in "xyz"], now)
    return rotated, {"mapping": "rotated_through_measured_gripper_axes",
                     "orientation_change_deg": round(turn, 2),
                     "offset_m": _r(rotated),
                     "note": ("the offset was re-expressed for the gripper's current "
                              "axes; it assumes the object has not moved within the "
                              "grasp, which nothing here measures")}


def attachment_still_holds(attachment, *, object_center, fingertip, measured=None):
    """Check a stored offset against one new MEASURED observation; drop it on failure.

    ``object_center`` must be a re-measurement. ``measured`` is the current
    end-effector reading, used to carry the offset into the gripper's present
    orientation rather than reusing a world vector across a rotation.
    """
    if not attachment or attachment.get("status") != "attached":
        return {"status": "unknown", "reason": "no accepted attachment offset"}
    offset, mapping = offset_in_current_orientation(attachment, measured or {})
    if offset is None:
        return {"status": "unknown", "reason": mapping.get("reason"),
                "offset_mapping": mapping}
    predicted = [fingertip[i] + offset[i] for i in range(3)]
    error = math.sqrt(sum((a - b) ** 2 for a, b in zip(predicted, object_center)))
    out = {"predicted_object_center": _r(predicted), "residual_m": round(error, 4),
           "gate_m": ATTACH_MAX_RESIDUAL_M * 2, "offset_mapping": mapping}
    if error > out["gate_m"]:
        out["status"] = "lost"
        out["reason"] = (f"the object is {error:.4f} m from where the stored offset "
                         f"predicts, beyond {out['gate_m']:g} m: released, slipped or "
                         f"never held")
        return out
    out["status"] = "attached"
    return out


# ── relations -> gripper targets ─────────────────────────────────────────────

RELATIONS = ("approach_grasp", "above", "align_over", "descend_to", "retreat")
NEEDS_ATTACHMENT = ("align_over", "descend_to")
# What each relation's success is actually a statement about, and therefore which
# quantity has to be measured to judge it. One generic "object centre minus goal"
# was wrong for three of the five:
#
# * approach_grasp places the FINGERTIPS at a standoff from a rim point. Comparing
#   the bowl's centre to that rim point reports roughly one radius of error even
#   when the fingertips are exactly where they were asked to be.
# * above places the fingertip over the reference's centre, and the reference is
#   also the subject — so centre-minus-goal is the object compared with itself and
#   reports 0.0000 m however far the gripper actually is.
# * retreat has no object at all.
#
# So a reach is judged at the ENDPOINT (measured fingertip pose against the target
# re-solved from current reference geometry, orientation included, since a grasp
# approach that arrives rotated has not arrived), and only a placement is judged by
# where the held OBJECT ended up.
RELATION_VERIFIED_BY = {"approach_grasp": "endpoint", "above": "endpoint",
                        "retreat": "endpoint", "align_over": "relation",
                        "descend_to": "relation"}
# Orientation error above which an endpoint has not arrived. A grasp approach that
# reaches the right point with the jaws turned 30 deg closes across the wrong axis.
ENDPOINT_MAX_ORIENTATION_DEG = 10.0


def endpoint_residual(target, measured, *, tolerance_m,
                      max_orientation_deg=ENDPOINT_MAX_ORIENTATION_DEG):
    """Measured end-effector pose against a solved target: position AND orientation.

    Position alone is not enough to judge a reach: the fingertip can sit exactly on
    a rim grasp point with the jaw axis along the rim instead of across it, which is
    a different action from the one that was requested. Both errors are reported,
    and ``within_tolerance`` requires both.

    Returns ``status="unknown"`` when the frame carries no measured end effector, or
    when the orientation cannot be compared — never a position-only pass dressed up
    as an arrival.
    """
    if (measured or {}).get("status") != "ok":
        return {"status": "unknown", "basis": "endpoint",
                "reason": "this frame carries no measured end effector, so where the "
                          "gripper actually ended up is unknown"}
    tip = [measured["fingertip_position"][k] for k in "xyz"]
    goal = list(target["position"])
    delta = [tip[i] - goal[i] for i in range(3)]
    error = math.sqrt(sum(d * d for d in delta))
    out = {"status": "measured", "basis": "endpoint", "goal": _r(goal),
           "observed": _r(tip), "error_m": round(error, 4), "per_axis_m": _r(delta),
           "tolerance_m": tolerance_m,
           "orientation_tolerance_deg": max_orientation_deg}
    now = _grasp_basis(measured)
    want = _grasp_basis({"approach": target.get("approach"),
                         "opening": target.get("opening")})
    if now is None or want is None:
        out["status"] = "unknown"
        out["reason"] = ("the gripper's own axes are not measured on this frame, so "
                         "whether it arrived in the requested orientation is unknown; "
                         "a grasp approach cannot be judged from position alone")
        out["missing"] = "current_gripper_axes" if now is None else "target_axes"
        return out
    trace = sum(sum(now[a][k] * want[a][k] for k in range(3))
                for a in ("approach", "opening", "third"))
    turn = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
    out["orientation_error_deg"] = round(turn, 2)
    out["within_tolerance"] = bool(error <= tolerance_m
                                   and turn <= max_orientation_deg)
    if not out["within_tolerance"]:
        out["reason"] = (f"the gripper is {error:.4f} m (gate {tolerance_m:g} m) and "
                         f"{turn:.1f} deg (gate {max_orientation_deg:g} deg) from the "
                         f"solved pose")
    return out


def validate_run(arguments):
    """Validate an entire fg_run request. Nothing executes unless all of it passes.

    The whole point of validating up front is that a schema error in stage 4 must
    not be discovered after stages 1-3 have physically moved the robot.
    """
    if not isinstance(arguments, dict):
        raise Rejected("arguments must be an object")
    extra = set(arguments) - {"stages", "budget"}
    if extra:
        raise Rejected(f"unexpected fields: {sorted(extra)}")
    stages = arguments.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise Rejected(f"stages must be a list of 1 to {MAX_STAGES} entries")
    budget = arguments.get("budget") or {}
    if not isinstance(budget, dict) or set(budget) - {"waypoints", "seconds"}:
        raise Rejected("budget may only contain 'waypoints' and 'seconds'")
    waypoints = budget.get("waypoints", DEFAULT_WAYPOINT_BUDGET)
    if type(waypoints) is not int or not 1 <= waypoints <= MAX_WAYPOINT_BUDGET:
        raise Rejected(f"budget.waypoints must be an integer in [1, {MAX_WAYPOINT_BUDGET}]")
    seconds = _finite(budget.get("seconds", DEFAULT_DEADLINE_S), "budget.seconds")
    if not 5.0 <= seconds <= MAX_DEADLINE_S:
        raise Rejected(f"budget.seconds must be within [5, {MAX_DEADLINE_S:g}]")
    out = []
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict) or len(stage) != 1:
            raise Rejected(f"stages[{i}] must have exactly one key: move, gripper, "
                           f"calibrate_attachment or check")
        kind, body = next(iter(stage.items()))
        if kind == "gripper":
            if body not in ("open", "close"):
                raise Rejected(f"stages[{i}].gripper must be 'open' or 'close'")
            out.append({"kind": "gripper", "action": body})
        elif kind == "check":
            if not isinstance(body, dict) or set(body) - {"refs"}:
                raise Rejected(f"stages[{i}].check may only contain 'refs'")
            refs = body.get("refs")
            if not isinstance(refs, list) or not refs or not all(
                    isinstance(r, str) for r in refs):
                raise Rejected(f"stages[{i}].check.refs must be a non-empty list of ids")
            out.append({"kind": "check", "refs": list(refs)})
        elif kind == "calibrate_attachment":
            if not isinstance(body, dict) or set(body) - {"ref"}:
                raise Rejected(f"stages[{i}].calibrate_attachment may only contain 'ref'")
            if not isinstance(body.get("ref"), str):
                raise Rejected(f"stages[{i}].calibrate_attachment.ref must be an id")
            out.append({"kind": "calibrate_attachment", "ref": body["ref"]})
        elif kind == "move":
            out.append(_validate_move(body, i))
        else:
            raise Rejected(f"stages[{i}] has unknown kind {kind!r}")
    # No "must contain a move" rule. A physical gripper stage on its own is a real
    # and often necessary action — closing on a grasp, releasing a placed object —
    # and requiring a move alongside it would have forced a pointless motion into
    # the same waypoint, which is exactly what the guide tells the model not to do.
    # A run of checks alone is also meaningful: it re-measures without acting, and
    # the reply reports physical=false so that is visible rather than implied.
    return {"stages": out, "budget": {"waypoints": waypoints, "seconds": seconds},
            "physical": any(s["kind"] in ("move", "gripper") for s in out)}


def _validate_move(body, index):
    if not isinstance(body, dict):
        raise Rejected(f"stages[{index}].move must be an object")
    relation = body.get("relation")
    if relation not in RELATIONS:
        raise Rejected(f"stages[{index}].move.relation must be one of {list(RELATIONS)}")
    allowed = {"relation", "ref", "grasp", "standoff_m", "height_m", "distance_m",
               "refine", "tolerance_m"}
    extra = set(body) - allowed
    if extra:
        raise Rejected(f"stages[{index}].move has unexpected fields: {sorted(extra)}")
    move = {"kind": "move", "relation": relation,
            "refine": bool(body.get("refine", False))}
    tolerance = _finite(body.get("tolerance_m", DEFAULT_TOLERANCE_M),
                        f"stages[{index}].move.tolerance_m")
    if not 0.002 <= tolerance <= 0.05:
        raise Rejected(f"stages[{index}].move.tolerance_m must be within [0.002, 0.05]")
    move["tolerance_m"] = tolerance
    if relation == "retreat":
        distance = _finite(body.get("distance_m", 0.08), f"stages[{index}].move.distance_m")
        if not 0.01 <= distance <= 0.20:
            raise Rejected(f"stages[{index}].move.distance_m must be within [0.01, 0.20]")
        move["distance_m"] = distance
        if move["refine"]:
            raise Rejected(f"stages[{index}].move: retreat has no reference to refine "
                           f"against")
        return move
    ref = body.get("ref")
    if not isinstance(ref, str) or not ref:
        raise Rejected(f"stages[{index}].move.relation {relation!r} needs a ref")
    move["ref"] = ref
    if relation == "approach_grasp":
        standoff = _finite(body.get("standoff_m", 0.05), f"stages[{index}].move.standoff_m")
        if not 0.0 <= standoff <= 0.20:
            raise Rejected(f"stages[{index}].move.standoff_m must be within [0, 0.20]")
        move["standoff_m"] = standoff
        if body.get("grasp") is not None and not isinstance(body["grasp"], str):
            raise Rejected(f"stages[{index}].move.grasp must be a candidate id")
        move["grasp"] = body.get("grasp")
    else:
        height = _finite(body.get("height_m", 0.10), f"stages[{index}].move.height_m")
        if not -0.02 <= height <= 0.30:
            raise Rejected(f"stages[{index}].move.height_m must be within [-0.02, 0.30]")
        move["height_m"] = height
    return move


def base_offset_below_center(held_card, offset_mapping):
    """How far the held object's observed base sits below its fitted centre.

    Returns ``(metres, None)`` or ``(None, refusal)``. This is the vertical
    separation between the two points the card actually reports — ``center`` and
    ``base_center`` — so it stays consistent with whatever that shape's centre
    means, which ``height_m / 2`` did not: a ring's centre is its rim, not its
    midpoint.

    Refused rather than approximated in three cases, because each would silently
    turn a place-the-bottom command into a place-something-else command: no card
    for the held object, a card missing either point, and a grasp that has rotated
    since calibration. The separation is a purely vertical quantity taken under the
    upright prior; once the grasp has turned, the object's own base direction is no
    longer world-down and this scalar does not describe it.
    """
    if not held_card:
        return None, ("descend_to positions the held object's own base, and no bound "
                      "reference for the held object is currently measured, so how far "
                      "below the grasp its base sits is unknown")
    center, base = held_card.get("center"), held_card.get("base_center")
    if not center or not base:
        return None, ("the held object's card has no fitted centre and base, so the "
                      "separation between the grasp's reference point and the object's "
                      "base is unknown")
    drop = center["z"] - base["z"]
    if not math.isfinite(drop) or drop < 0:
        return None, (f"the held object's centre-to-base separation came out "
                      f"{drop!r}, which is not a usable vertical extent")
    if offset_mapping.get("mapping") == "rotated_through_measured_gripper_axes":
        return None, ("the grasp has rotated since the offset was measured, so the "
                      "held object's base is no longer directly below its fitted "
                      "centre and this vertical separation does not describe it; "
                      "re-calibrate at the current orientation, or use align_over, "
                      "which positions the centre and needs no base")
    return drop, None


def solve_relation(move, *, card, measured, attachment=None, held_card=None):
    """Resolve one validated move against current geometry. No motion here.

    Returns ``{"status": "ok", "target": ...}`` or ``{"status": "refused", ...}``.
    This is the arithmetic the model used to do by hand, done once, locally, against
    the geometry that was just measured — and refused outright when the geometry or
    the attachment evidence the relation depends on is not there.
    """
    relation = move["relation"]
    if measured.get("status") != "ok":
        return {"status": "refused", "relation": relation,
                "reason": "no measured end effector in this frame"}
    tip = [measured["fingertip_position"][k] for k in "xyz"]
    approach = ([measured["approach"][k] for k in "xyz"]
                if measured.get("approach") else [0.0, 0.0, -1.0])
    opening = ([measured["opening"][k] for k in "xyz"]
               if measured.get("opening") else [1.0, 0.0, 0.0])

    if relation == "retreat":
        distance = move["distance_m"]
        target = [tip[i] - approach[i] * distance for i in range(3)]
        return {"status": "ok", "relation": relation,
                "target": {"position": target, "approach": approach, "opening": opening},
                "explain": (f"back off {distance:g} m along the measured approach "
                            f"axis from the current fingertip")}

    if card is None or not card.get("valid"):
        return {"status": "refused", "relation": relation,
                "reason": "the reference has no valid current geometry",
                "reference_reasons": (card or {}).get("reasons")}

    if relation == "approach_grasp":
        candidates = {c["id"]: c for c in card.get("grasp_candidates") or []}
        chosen = move.get("grasp")
        if chosen is None:
            if not candidates:
                return {"status": "refused", "relation": relation,
                        "reason": "this reference offers no grasp candidates; its "
                                  "shape fit locates no graspable feature"}
            chosen = min(candidates, key=lambda k: sum(
                (candidates[k]["point"][a] - t) ** 2 for a, t in zip("xyz", tip)))
        if chosen not in candidates:
            return {"status": "refused", "relation": relation,
                    "reason": f"unknown grasp candidate {chosen!r}",
                    "available": sorted(candidates)}
        cand = candidates[chosen]
        point = [cand["point"][k] for k in "xyz"]
        axis = [cand["approach"][k] for k in "xyz"]
        jaw = [cand["opening_axis"][k] for k in "xyz"]
        standoff = move["standoff_m"]
        target = [point[i] - axis[i] * standoff for i in range(3)]
        return {"status": "ok", "relation": relation, "grasp": chosen,
                "target": {"position": target, "approach": axis, "opening": jaw},
                "goal_point": point,
                "explain": (f"fingertips {standoff:g} m short of grasp candidate "
                            f"{chosen} along its approach axis")}

    if relation == "above":
        anchor = card.get("center") or card.get("base_center")
        target = [anchor["x"], anchor["y"], anchor["z"] + move["height_m"]]
        return {"status": "ok", "relation": relation,
                "target": {"position": target, "approach": approach, "opening": opening},
                "goal_point": [anchor[k] for k in "xyz"],
                "explain": (f"fingertip {move['height_m']:g} m above the reference "
                            f"centre, orientation unchanged")}

    # align_over / descend_to move the HELD object, so they need the measured
    # offset. Without it the fingertip would be placed over the target and the
    # object would land wherever it happens to hang.
    if attachment is None or attachment.get("status") != "attached":
        return {"status": "refused", "relation": relation,
                "reason": ("this relation moves a held object, and there is no "
                           "measured object-to-gripper offset; run "
                           "calibrate_attachment after closing, with real motion "
                           "between the observations"),
                "attachment": (attachment or {}).get("status", "unknown")}
    offset, offset_mapping = offset_in_current_orientation(attachment, measured)
    if offset is None:
        return {"status": "refused", "relation": relation,
                "reason": offset_mapping.get("reason"),
                "offset_mapping": offset_mapping}
    top = card.get("top_z")
    if top is None:
        return {"status": "refused", "relation": relation,
                "reason": "the target reference has no measured top surface height"}
    anchor = card.get("center") or card.get("base_center")
    note = None
    extra = {}
    if relation == "align_over":
        object_goal = [anchor["x"], anchor["y"], top + move["height_m"]]
    else:
        # descend_to places the held object's own observed BASE at the requested
        # height, and the offset the attachment measured runs to that object's
        # fitted centre — so the goal has to be raised by the separation between
        # those two, measured on the same card.
        #
        # It is emphatically NOT height/2: a ring's centre is the average height of
        # its upper band, i.e. near the rim, and a blob's is the centroid of the
        # returns. Neither is the midpoint of the bounding box, so halving the
        # height dropped a bowl by most of its own depth. Nothing here is a
        # measurement of the true physical bottom either: base_center.z is the
        # lowest height the cloud actually returned, which an occluded or
        # self-shadowed underside makes an upper bound on how low the object goes.
        drop, refusal = base_offset_below_center(held_card, offset_mapping)
        if refusal is not None:
            return {"status": "refused", "relation": relation, "reason": refusal,
                    "held_ref_note": ("descend_to positions the held object's own "
                                      "observed base, so it needs that object bound "
                                      "and currently measured; use align_over to "
                                      "position its centre instead")}
        object_goal = [anchor["x"], anchor["y"], top + move["height_m"] + drop]
        extra = {"held_base_below_center_m": round(drop, 4),
                 "held_base_source": ("centre-to-lowest-return separation measured on "
                                      "the held object's own card, under its upright "
                                      "prior; the lowest return is a bound on the "
                                      "physical bottom, not a measurement of it")}
    out = {"status": "ok", "relation": relation, **extra,
           "target": {"position": [object_goal[i] - offset[i] for i in range(3)],
                      "approach": approach, "opening": opening},
           "goal_point": object_goal, "object_goal": object_goal,
           "attachment_offset_m": _r(offset),
           "attachment_offset_mapping": offset_mapping,
           "explain": (f"held object centre to ({object_goal[0]:.3f}, "
                       f"{object_goal[1]:.3f}, {object_goal[2]:.3f}) using the "
                       f"measured offset; the fingertip target follows from it")}
    if note:
        out["approximation"] = note
    return out


def relation_residual(solved, *, object_center):
    """Observed error of the *relation*, from re-measured geometry.

    Distinct from the end-effector error on purpose: the gripper arriving at the
    solved pose says nothing about the object arriving, and collapsing the two is
    how "the arm got there" gets reported as "the object is placed".
    """
    goal = solved.get("object_goal") or solved.get("goal_point")
    if goal is None or object_center is None:
        return {"status": "unknown",
                "reason": "no re-measured object geometry to compare against the goal"}
    actual = [object_center[k] for k in "xyz"] if isinstance(object_center, dict) \
        else list(object_center)
    delta = [actual[i] - goal[i] for i in range(3)]
    return {"status": "measured", "goal": _r(goal), "observed": _r(actual),
            "error_m": round(math.sqrt(sum(d * d for d in delta)), 4),
            "per_axis_m": _r(delta),
            "horizontal_error_m": round(math.hypot(delta[0], delta[1]), 4)}


def _split(start, end, step_m):
    """Equal-length points from ``start`` to ``end``, last point exactly ``end``."""
    delta = [end[i] - start[i] for i in range(3)]
    distance = math.sqrt(sum(d * d for d in delta))
    if distance <= step_m:
        return [list(end)], distance
    count = int(math.ceil(distance / step_m))
    points = [[start[i] + delta[i] * (k + 1) / count for i in range(3)]
              for k in range(count)]
    points[-1] = list(end)
    return points, distance


def edit_steps(start, end, *, step_m=MAX_EDIT_STEP_M):
    """Virtual target edits needed to move the target from ``start`` to ``end``.

    These are BROWSER edits, not robot motions. ``target_edit`` caps one edit at
    0.1 m, so a longer displacement is walked in several edits — all of them before
    a single execute, so they cost no physical waypoint and no sim steps. This is
    the distinction the earlier version got wrong: it treated the 0.1 m editing cap
    as a limit on how far the robot could travel per waypoint.
    """
    return _split(start, end, step_m)[0]


def segment_path(start, end, *, segment_m=ACTION_SEGMENT_M):
    """Split one relation's motion into PHYSICAL segments, endpoint exact.

    A segment boundary is where the executor stops to take a new paired
    observation, so segmenting is a perception decision, not a robot limit. Each
    segment is one physical waypoint (however many virtual edits it takes to place
    the target — see ``edit_steps``), and each costs real controller time and sim
    steps. The default of ``ACTION_SEGMENT_M`` covers the tabletop in one motion, so
    an ordinary reach is one waypoint; a longer solved displacement is split because
    executing further than that open-loop against pre-move geometry is the risk, and
    every split is counted and reported rather than presented as a saving.
    """
    points, distance = _split(start, end, segment_m)
    return points


def plan_motion(start, end, *, segment_m=ACTION_SEGMENT_M, step_m=MAX_EDIT_STEP_M):
    """The full plan for one relation's motion, with its honest cost.

    Returns the physical segments, how many virtual edits each needs, and the
    totals. ``physical_waypoints`` is what actually costs controller time; ``edits``
    is browser-only work. Reporting both is the point — a reply that showed only the
    edit count would understate physical cost, and one that showed only waypoints
    would hide why a far target takes several browser calls.
    """
    segments = segment_path(start, end, segment_m=segment_m)
    at = list(start)
    plan = []
    for target in segments:
        steps = edit_steps(at, target, step_m=step_m)
        plan.append({"target": [round(v, 5) for v in target], "edits": len(steps),
                     "edit_targets": steps,
                     "distance_m": round(math.sqrt(sum(
                         (target[i] - at[i]) ** 2 for i in range(3))), 4)})
        at = target
    total = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
    return {"segments": plan, "physical_waypoints": len(plan),
            "edits": sum(s["edits"] for s in plan),
            "total_distance_m": round(total, 4),
            "segment_limit_m": segment_m,
            "why_segmented": (None if len(plan) == 1 else
                              (f"the solved displacement is {total:.3f} m, longer than "
                               f"the {segment_m:g} m observation-bounded segment, so "
                               f"the executor stops in between to re-measure; each "
                               f"segment is a real physical waypoint"))}


def orientation_steps(current, target, *, max_deg=MAX_EDIT_ROTATION_DEG):
    """Virtual orientation edits from ``current`` to ``target`` axes.

    ``target_edit`` caps one edit at 90 deg, so a larger turn is walked in several
    edits — browser-only, like ``edit_steps``, and all of them before one execute.
    Each intermediate pair is re-orthogonalized, because interpolating two unit
    vectors independently does not keep them perpendicular and the editor rejects a
    non-orthogonal pair rather than silently correcting a large error.

    Returns ``[]`` when the pair already matches, and ``None`` when either basis is
    unusable — the caller must then refuse rather than execute a guessed pose.
    """
    now, want = _grasp_basis(current), _grasp_basis(target)
    if now is None or want is None:
        return None
    trace = sum(sum(now[a][k] * want[a][k] for k in range(3))
                for a in ("approach", "opening", "third"))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
    if angle <= 1e-6:
        return []
    count = max(1, int(math.ceil(angle / max_deg)))
    out = []
    for k in range(count):
        f = (k + 1) / count
        if k == count - 1:
            out.append({"approach": list(want["approach"]),
                        "opening": list(want["opening"])})
            break
        a = _unit3([now["approach"][i] * (1 - f) + want["approach"][i] * f
                    for i in range(3)])
        o = _unit3([now["opening"][i] * (1 - f) + want["opening"][i] * f
                    for i in range(3)])
        step = _grasp_basis({"approach": a, "opening": o})
        if step is None:
            return None
        out.append({"approach": step["approach"], "opening": step["opening"]})
    return out


def progress_made(previous_error, new_error, *, fraction=PROGRESS_FRACTION):
    """Whether a correction improved the residual enough to justify another.

    Without this a refinement loop can spend its whole budget oscillating around a
    residual it cannot reduce — motion that looks like progress and is not.
    """
    if previous_error is None or new_error is None:
        return None
    return new_error <= previous_error * (1.0 - fraction)
