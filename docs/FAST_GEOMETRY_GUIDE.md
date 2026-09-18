# Robot control: fast geometry tools

You operate a simulated robot arm with a two-finger gripper over a tabletop, and
you complete the task you were given. This guide is the whole guide for this
interface — the six tools below are the only ones the server answers, so there is
no per-waypoint nudging, no click-to-teleport, no camera control and no separate
gripper toggle. Anything else you may have seen documented elsewhere comes back
`Unknown tool`.

    fg_look    one screenshot of the UI and the current frame id
    fg_bind    fit the objects you point at, once per object, and name them
    fg_check   re-measure named objects against a newer frame
    fg_run     execute a short sequence of stages, bounded
    fg_state   what is currently known — no new measurement, no images
    end_episode

## What you are looking at

`fg_look` returns the browser UI. Two raw camera feeds on the left — a
third-person `agentview` and a `wrist` view — and the 3-D point cloud on the
right, reconstructed from depth cameras. The agentview is the one to use for
telling objects apart; the wrist view is close-up. The point cloud is where the
fits happen.

Coordinates are the robot frame in metres: +x away from the base, +y left, +z up.
Regions are fractions of whichever surface you name, `u` across from 0.0 at the
left, `v` down from 0.0 at the top.

You act autonomously: do not ask questions. When you believe the task is done,
call `end_episode`. Success is judged by the benchmark, separately from every
measurement these tools report — a satisfied relation is not a satisfied task, and
nothing here reads the simulator's object poses, its segmentation, or its success
flag. What you know is what the cloud, the cameras and the robot's own telemetry
show.

## The loop

Look, bind everything you need in one call, run a few stages, read the residuals,
correct if needed.

Regions must come from the view you have **just looked at**. Below is the shape of
a request, not a set of coordinates to reuse: `u`/`v` here are placeholders, and
every object sits somewhere different in every episode. Read the objects off your
own `fg_look` image (or a camera feed) and put your own numbers in.

```
fg_look
                                    # regions read off THIS episode's agentview
fg_bind  {"surface": "agentview", "objects": [
           {"name": "bowl",  "shape": "ring",
            "region": {"kind": "box", "u0": <bowl>, "v0": <bowl>, "u1": <bowl>, "v1": <bowl>}},
           {"name": "plate", "shape": "disc",
            "region": {"kind": "box", "u0": <plate>, "v0": <plate>, "u1": <plate>, "v1": <plate>}}]}
```

Name a `shape`. Without one you get `blob`, which reports a centroid and a box and
offers **no grasp candidates**, so `approach_grasp` has nothing to aim at. `ring`
suits a bowl or a cup — anything whose graspable feature is a rim — and `disc`
suits a plate or a lid. Read the returned card before acting on it: `valid`, the
`center`, `extent_m`, `top_z`, and the `grasp_candidates` you are about to name.

Then a grasp, in six stages:

```
fg_run   {"stages": [
           {"move": {"relation": "approach_grasp", "ref": "bowl#1",
                     "grasp": "rim_y+", "standoff_m": 0.05}},   # clearance above the rim
           {"move": {"relation": "approach_grasp", "ref": "bowl#1",
                     "grasp": "rim_y+", "standoff_m": 0.0}},    # onto the candidate itself
           {"gripper": "close"},
           {"check": {"refs": ["bowl#1"]}},                     # the closed frame
           {"move": {"relation": "retreat", "distance_m": 0.05}},
           {"calibrate_attachment": {"ref": "bowl#1"}}]}
```

Why it is shaped that way:

- Two `approach_grasp` stages, not one. The first stops at a standoff along the
  approach axis so the fingers are clear of the rim; the second goes to the
  candidate point itself, which is where the jaws have to be to close on
  anything. A close issued from the standoff closes on air.
- `close` is its own stage. Every gripper stage is its own physical waypoint.
- The `check` comes **before** the retreat, and the retreat before the
  calibration. Calibration needs two distinct paired frames measured *while the
  gripper is closed*, with real motion between them — it moves nothing itself. The
  check supplies the first, the retreat moves the arm and mints the second, and
  the calibration fits the offset from the pair. Reverse the check and the
  retreat and the calibration sees one observation and refuses for want of
  evidence.

This sequence is a **proposal**, not a result. It is what asking for a physical
grasp looks like; whether the fingers actually closed on the bowl is decided by
what comes back — the approach stage's `arrival`, and then the calibration's
verdict. Read those before you plan the placement.

Only once `calibrate_attachment` reports `attached` do `align_over` and
`descend_to` mean anything, so the placement is a **later** request:

```
fg_run   {"stages": [
           {"move": {"relation": "align_over", "ref": "plate#2", "height_m": 0.08}},
           {"move": {"relation": "descend_to", "ref": "plate#2", "height_m": 0.01}},
           {"gripper": "open"}]}
```

Bind both objects in one call when you can: two objects fitted from the same frame
describe the same moment, and a relation between them is then a comparison of two
objects rather than of two times.

## Regions

`surface` is `canvas` (the 3-D point cloud) or `agentview` / `wrist` (the camera
feeds). Coordinates are fractions of that surface, so `u0: 0.43` is 43% across.

    {"kind": "box",     "u0":, "v0":, "u1":, "v1":}
    {"kind": "point",   "u":, "v":, "radius_px":}
    {"kind": "polygon", "points": [[u, v], ...]}          up to 12 vertices

A region says *where to look*, not what is there. The fit uses the cloud points
inside it, so a box that covers two objects may return the wrong one, and a box on
the stove returns the stove. `name` is your label and is never verified — if the
returned `center`, `extent_m` or `top_z` does not describe the object you meant,
re-bind with a tighter region rather than proceeding.

`shape` is `ring`, `disc`, `plane_patch` or `blob` (the default). What each one
actually establishes:

- `ring` and `disc` fit a circle in the **horizontal** plane through the
  cluster's upper band. That presumes the object is upright: `up` is reported as
  world up, an *assumption* that is then tested against the tilt of the band's own
  plane, and the card is invalid if the band is too tilted for the fit to mean
  anything. It is not a fitted 3-D orientation. These two are the shapes that
  offer `grasp_candidates` — rim points with the jaws across the rim.
- `plane_patch` is the one shape that fits a normal: a least-squares plane, with
  the normal reported and its residual gated.
- `blob` claims a centroid, a bounding box and a principal axis. No residual, no
  fitted axis, no grasp candidates.

Each object comes back as a card: `ref` (use this id everywhere), `valid`,
`center`, `base_center`, `top_z`, `extent_m`, `up` with an `up_source` saying
where it came from, `grasp_candidates`, and a `fit` block with `object_points` and
`residual_rms_m`. A residual is the spread of the returns about the fitted shape —
how well the shape explains what was seen, not how accurately the object was
located. An invalid card carries `reasons` — read them, they say what to change.

## Relations

Each `move` stage takes a relation and a reference:

| relation | what it positions | knob |
|---|---|---|
| `approach_grasp` | fingers at a grasp candidate, backed off along the approach | `standoff_m` [0, 0.2] |
| `above` | gripper above the object | `height_m` [-0.02, 0.3] |
| `align_over` | held object above the target | `height_m` |
| `descend_to` | held object lowered to the target | `height_m` |
| `retreat` | gripper away along its own approach | `distance_m` [0.01, 0.2] |

`align_over` and `descend_to` are about the object you are holding, so they need a
confirmed attachment. Optional per-move: `grasp` (a candidate id, default the
nearest), `tolerance_m` [0.002, 0.05] default 0.01, `refine` (correct against the
re-measured residual, up to 2 attempts).

## What "arrived" means

Two different numbers, reported separately, because they answer different
questions:

- `arrival` — how close the **gripper** got to the pose that was asked for.
- `relation_residual` — how close the **held object** got to the target geometry.

`verified_by` tells you which one is the verdict. `approach_grasp`, `above` and
`retreat` position the gripper, so their verdict is the arrival — comparing them
against the reference they were derived from would be comparing a number with
itself. `align_over` and `descend_to` are judged by the object, because a gripper
in the right place holding the bowl 3 cm off-centre has not aligned anything.

Both are measured from a frame observed *after* the motion. `status: "measured"`
means a real observation backed it; anything else means the number is not
evidence. Neither is a task verdict — the benchmark decides success, separately.

## Attachment

Closing the gripper does not create a hold. A hold is *measured*:
`calibrate_attachment` moves nothing and checks whether the object displaced with
the gripper, across two closed paired frames with motion between them. Only then
can `align_over` and `descend_to` be planned, and a placement is refused outright
if the hold has been lost.

The hold is re-verified at every later measurement boundary, including between the
segments of one long move. If the object stops moving with the gripper, or the
fingers are seen open while the stored offset still predicts the object, the
attachment is cleared and the reply says so.

`attachment_not_confirmed` is the third case and is not the same as a lost hold:
the frame does not report the gripper as closed, so nothing on it shows the object
is in the jaws. A placement is refused — before the first segment, or at the
boundary where the evidence stopped — but the stored offset is **kept**, because a
frame that says nothing is not a frame that says the object was released. One
`fg_check` on a frame that does report the state is enough; no re-grasp is implied.

## `fg_run` is bounded

Up to 8 stages. The whole request is validated before anything moves, so a
malformed stage 5 stops stage 1 from executing. Default budget 12 waypoints and
110 s (max 24 / 150 s), settable per call via `budget`.

A stage whose verdict is unknown or out of tolerance stops the stages that
depended on it, and the reply lists what did not run. That is not a failure to
route around — it means the measurement the next stage needed does not exist.
Re-measure with `fg_check`, then send a corrected sequence.

## Reading a reply

A success is a number; a failure is an explanation. Successful stages come back
compact — the effective ref, the measured error, the counts. Refusals and aborts
carry the reason, the message and the limits. Call `fg_state` when you want the
full stored picture without spending a measurement.

`status` values: `ok`, `rejected` (the request was malformed, nothing moved),
`unknown` (nothing could be measured), `aborted` (some stages ran, the reply says
which). `aborted` also covers a required move that could not be shown to have
arrived, even when it was the last stage: the reply names it and keeps everything
that did happen.

`unknown / same_frame_no_new_evidence` from `fg_check` is not an error: no new
observation has arrived since that reference was last measured, so there is
nothing new to say and the stored fit was left alone. It is also what you get for
a reference `fg_run` already re-measured at the same boundary.

`size_mismatch` or `ambiguous_two_candidates` means re-association refused rather
than guess. Common cause: the arm is now sitting in front of the object, so the
visible cluster is not the one that was bound. Move the arm out of the way and
check again, or re-bind.

## What these tools cannot tell you

Fits use the visible, downsampled cloud. A residual says how well the shape
explains the returns that were *seen* — an occluded or clipped surface can fit
well and still be wrong. There is no collision proof, no contact dynamics, and no
verified object identity. Nothing here measures grasp quality: `attached` says the
object moved with the gripper, not that it is held securely. Where the fingers
close is your decision, and `top_z` with `extent_m` is what tells you whether they
will have surface to hold or will close near the very top edge of the object.
