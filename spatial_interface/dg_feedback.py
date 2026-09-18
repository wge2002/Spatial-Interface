"""Opt-in observation feedback. No object state, action selection or control."""
from __future__ import annotations

import base64
from io import BytesIO
import math
import os

try:
    from . import geometry_ref as geom, fast_geometry_tools as fgt
except ImportError:
    import geometry_ref as geom, fast_geometry_tools as fgt

VARIANTS = ("baseline", "grounded", "paired")
CAMERAS = ("agentview", "wrist")


def variant():
    value = os.environ.get("VIA_DG_FEEDBACK", "baseline")
    if value not in VARIANTS:
        raise ValueError("VIA_DG_FEEDBACK must be baseline, grounded or paired")
    return value


def project(point, calibration):
    """Robot/world metres to normalized RGB coordinates; E is world-from-camera."""
    k, e, size = calibration["K"], calibration["E"], calibration["img_size"]
    delta = [point[i] - e[4*i+3] for i in range(3)]
    xyz = [sum(e[4*j+i]*delta[j] for j in range(3)) for i in range(3)]
    if not all(math.isfinite(x) for x in xyz) or xyz[2] <= 1e-6:
        return None
    u = (k[0]*xyz[0]/xyz[2]+k[2])/size
    v = (k[4]*xyz[1]/xyz[2]+k[5])/size
    return (u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None


def xyz(value):
    return [value[k] for k in "xyz"] if isinstance(value, dict) else list(value)


def temporal_evidence(before, after):
    if not before or not after:
        return {"status": "unknown", "reason": "paired_camera_capture_unavailable"}
    b, a = before["view"], after["view"]
    reason = geom.frame_advance(b["frame"], a["frame"])
    if reason:
        return {"status": "no_new_frame" if b["frame"] == a["frame"] else "unknown",
                "reason": reason, "before_frame": b["frame"], "after_frame": a["frame"]}
    steps = a["sim_steps_used"]-b["sim_steps_used"]
    if steps <= 0:
        return {"status": "unknown", "reason": "nonpositive_sim_step_delta"}
    bm, am = (v["measured_end_effector"] for v in (b, a))
    bp, ap = (xyz(m["fingertip_position"]) for m in (bm, am))
    return {"status": "paired", "before_frame": b["frame"], "after_frame": a["frame"],
            "sim_steps_delta": steps,
            "measured_eef_delta_m": dict(zip("xyz", [round(y-x, 6) for x,y in zip(bp,ap)])),
            "measured_jaw_width_before_m": bm.get("gripper_width_m"),
            "measured_jaw_width_after_m": am.get("gripper_width_m"),
            "commanded_gripper_before": bm.get("gripper_state_class"),
            "commanded_gripper_after": am.get("gripper_state_class"),
            "interpretation": "endpoint sensor evidence only; width/command do not prove grasp; no object tracking"}


CAPTURE_JS = """() => {
    const images = {}, calibration = {};
    for (const label of ['agentview', 'wrist']) {
        const el = document.getElementById(label === 'wrist' ? 'cam-wrist' : 'cam-agentview');
        if (!el || !el.complete || !el.naturalWidth || !camInfo[label])
            throw Error('camera image/calibration unavailable: '+label);
        const canvas = document.createElement('canvas');
        canvas.width = el.naturalWidth; canvas.height = el.naturalHeight;
        canvas.getContext('2d').drawImage(el, 0, 0);
        images[label] = canvas.toDataURL('image/png').split(',')[1];
        calibration[label] = JSON.parse(JSON.stringify(camInfo[label]));
    }
    return {images, calibration};
}"""


async def capture(ctx, view):
    """Copy decoded pixels, checking sensor and displayed-image IDs on both sides."""
    before = await ctx.observe_geometry()
    if (geom.observation_version(before) != view["frame"] or
            geom.pairing_state(before, require_cam_labels=CAMERAS)["status"] != "paired"):
        raise ValueError("camera_capture_not_paired_with_observation")
    result = await ctx.page.evaluate(CAPTURE_JS)
    after = await ctx.observe_geometry()
    if (geom.observation_version(after) != view["frame"] or
            geom.pairing_state(after, require_cam_labels=CAMERAS)["status"] != "paired"):
        raise ValueError("camera_frame_changed_during_capture")
    result["view"] = view
    return result


async def selected_returns(ctx, binding, frame):
    """Snapshot the model-selected cloud; these are ROI returns, not fit inliers."""
    out = {"source_frame": frame, "surface": binding["surface"], "regions": [], "points": []}
    for spec in binding["objects"]:
        extracted, _, error = await fgt.extract_region(ctx, spec, binding["surface"])
        if error or not extracted:
            raise ValueError(error or "selection_cloud_unavailable")
        values, stride = extracted.get("kept") or [], int(extracted.get("stride") or 8)
        if stride != 8:
            raise ValueError("unexpected_region_stride")
        points = [geom.ui_to_robot(values[i:i+3], ctx.ui_z_offset)
                  for i in range(0, len(values)-stride+1, stride)]
        out["points"].extend(points[::max(1, math.ceil(len(points)/160))])
        out["regions"].append({"name": spec["name"], "region": spec["region"], "returns": len(points)})
    if geom.observation_version(await ctx.observe_geometry()) != frame:
        raise ValueError("selection_frame_changed")
    return out


def _canvas(columns, rows):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (448*columns, 490*rows+70), "#14202c")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    return image, ImageDraw.Draw(image), font


def _panel(image, draw, font, snapshot, camera, x, y, title):
    from PIL import Image
    raw = Image.open(BytesIO(base64.b64decode(snapshot["images"][camera]))).convert("RGB")
    image.paste(raw.resize((448, 448)), (x, y+42))
    version = geom.parse_observation_version(snapshot["view"]["frame"])
    draw.text((x+10,y+6), f"{title} | {camera} | seq {version['seq']}", fill="white", font=font)


def _jpeg(image):
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def spatial_image(snapshot, selection, targets, cards):
    image, draw, font = _canvas(2, 1)
    current = snapshot["view"]["frame"]
    historical = bool(selection and selection["source_frame"] != current)
    measured = snapshot["view"]["measured_end_effector"]
    for col, camera in enumerate(CAMERAS):
        x, y = col*448, 0
        _panel(image, draw, font, snapshot, camera, x, y, "SPATIAL")
        calibration = snapshot["calibration"][camera]
        def pixel(p):
            uv = project(p, calibration)
            return (x+uv[0]*447, y+42+uv[1]*447) if uv else None
        for p in (selection or {}).get("points", []):
            at = pixel(p)
            if at:
                draw.ellipse((at[0]-1,at[1]-1,at[0]+1,at[1]+1), fill="#ffe16b")
        if selection and not historical and selection["surface"] == camera:
            for region in selection["regions"]:
                spec = region["region"]
                if spec["kind"] == "box":
                    draw.rectangle((x+spec["u0"]*447,42+spec["v0"]*447,
                                    x+spec["u1"]*447,42+spec["v1"]*447), outline="#ffe16b", width=2)
                elif spec["kind"] == "polygon":
                    points = [(x+u*447,42+v*447) for u,v in spec["points"]]
                    draw.line(points+points[:1], fill="#ffe16b", width=2)
                elif spec["kind"] == "point":
                    cx, cy = x+spec["u"]*447, 42+spec["v"]*447
                    radius = spec["radius_px"]*448/calibration["img_size"]
                    draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline="#ffe16b",width=2)
        for card in cards:
            if card.get("center") is not None:
                at = pixel(xyz(card["center"]))
                if at:
                    color = "#ffe16b" if card.get("valid") else "#ff6b6b"
                    draw.ellipse((at[0]-5,at[1]-5,at[0]+5,at[1]+5), outline=color, width=2)
                    draw.text((at[0]+6,at[1]+4), str(card.get("name", "ref"))[:20], fill=color, font=font)
        poses = [("EEF", xyz(measured["fingertip_position"]), xyz(measured["approach"]),
                  xyz(measured["opening"]), "#45f4ff")]
        poses += [(f"T{i+1}", t["position"],t["approach"],t["opening"],"#ff7bf0")
                  for i,t in enumerate(targets) if t]
        for label, center, approach, opening, color in poses:
            at = pixel(center)
            if at:
                draw.line((at[0]-6,at[1],at[0]+6,at[1]),fill=color,width=3)
                draw.line((at[0],at[1]-6,at[0],at[1]+6),fill=color,width=3)
                # A reached target often overlaps the actual EEF. Keep both labels legible.
                label_y = at[1]+8 if label == "EEF" else at[1]-20
                draw.text((at[0]+7,label_y), label,fill=color,font=font)
                for axis, length, width in ((approach,.04,3),(opening,.025,1)):
                    tip = pixel([p+length*d for p,d in zip(center,axis)])
                    if tip:
                        draw.line((at,tip),fill=color,width=width)
    draw.text((10,500), "Cyan: measured EEF | Magenta: requested targets (not predicted path)", fill="white", font=font)
    draw.text((10,526), "Yellow: selected cloud / reference " + ("(HISTORICAL; not tracked)" if historical else "(identity assigned by model)"), fill="#ffe16b", font=font)
    return _jpeg(image)


def temporal_image(before, after):
    image, draw, font = _canvas(2, 2)
    for row,camera in enumerate(CAMERAS):
        for col,(label,snapshot) in enumerate((("BEFORE",before),("AFTER",after))):
            _panel(image,draw,font,snapshot,camera,col*448,row*490,label)
    draw.text((10,990), "Same camera labels; wrist camera moves with robot. No object correspondence inferred.", fill="white", font=font)
    return _jpeg(image)


async def finish(ctx, payload, mode, before=None, selection=None, cards=(), errors=()):
    """Feedback failure is explicit and never discards an acknowledged motion."""
    result = {"variant": mode, "identity": "model_assigned_not_verified", "errors": list(errors)}
    images = []
    view = payload.get("observation") or payload.get("observation_before")
    try:
        if not view:
            raise ValueError("no_observation_available")
        after = await capture(ctx, view)
        result["spatial"] = {
            "status": "paired", "image_frame": view["frame"],
            "selection_source_frame": (selection or {}).get("source_frame"),
            "selection_historical": bool(selection and selection["source_frame"] != view["frame"]),
            "selection_surface": (selection or {}).get("surface"),
            "regions": (selection or {}).get("regions", []),
            "references": [{"name": c.get("name"),
                            "fit_valid": None if c.get("source", "").startswith("explicit_model") else c.get("valid"),
                            "source": c.get("source", "sensor_fit"),
                            "source_frame": c.get("bound_frame", c.get("from_frame")),
                            "historical": c.get("bound_frame", c.get("from_frame")) != view["frame"],
                            "identity_verified": False} for c in cards],
            "note": "selected cloud returns are not fit inliers; historical points are not current object location"}
        images.append(("Spatial feedback: selected evidence, requested targets, measured EEF", spatial_image(
            after,selection,payload.get("compiled_targets",[]),cards)))
        if mode == "paired":
            result["temporal"] = temporal_evidence(before,after)
            if result["temporal"]["status"] == "paired":
                images.append(("Temporal feedback: real before/after RGB endpoints",temporal_image(before,after)))
    except Exception as exc:
        result.setdefault("spatial", {"status": "unknown"})
        result["errors"].append(type(exc).__name__+": "+str(exc))
        if mode == "paired":
            result.setdefault("temporal", {"status": "unknown", "reason": "feedback_capture_failed"})
    payload["feedback"] = result
    return images
