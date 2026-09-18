# Geometry reference and proxy tools

These tools are available in addition to the robot tools described above. They
are read-only: they observe, summarize and compare. `execute_waypoint` remains
the only way anything physically moves, and `gripper_toggle` still gets its own
executed waypoint.

## What the tools see

Only three sources, all of them ordinary observations you already have:

* the live RGB-D point cloud shown in the 3-D panel, in the robot frame;
* the camera calibration behind the two camera feeds;
* the robot's own telemetry — the measured fingertip position and orientation,
  and a binary open/closed classification derived from observed gripper width.

They do not see object poses, object identities, segmentation labels, or whether
the task has succeeded. A label you pass in is stored verbatim as `model_label`
and is never treated as a verified identity of anything.

## Observation versions and pairing

Every reply carries an `observation_version` and a `pairing` block. Both come
from the simulator's own numbering of its **observations**, not from the contents
of the cloud and not from the messages that carry them: the version advances when
the simulator takes a new reading of its sensors, and it also says which
observation produced the cloud that is currently in the browser. Two replies with
the same `observation_version` read the same observation.

A message is not an observation. While nothing moves, the simulator keeps
redelivering the one snapshot it already took — the same cloud bytes, the same
camera images, the same proprioception — so the version does not move either, and
a reference bound at one tool call is still evaluable at the next. Delivery is
counted separately in `frame_meta.delivery_seq`, with `frame_meta.redelivery` set
on a repeat, and neither appears in any version: a redelivery is not evidence that
anything was measured again, so treating it as a new observation would retire
references that nothing had invalidated. Redelivered frames do not touch the
virtual target or the completed-waypoint count.

While a waypoint executes the simulator streams frames that carry a fresh end
effector and fresh camera images but **resend the previous cloud**. Those frames
are new observations, so the version changes, but their cloud is older than the
rest of them. `pairing.status` is `paired` only when the cloud, the telemetry and (for
a click resolved on a camera feed) the image actually on screen were all measured
in the same frame. Anything else is `pairing.status='unknown'` with a reason, and
the tool then binds nothing, refreshes nothing and evaluates no corridor.

Moving the camera, editing the virtual target, hovering and taking a screenshot
do not go through the simulator at all, so they cannot change the version or
manufacture a paired frame. A re-read of an unchanged observation costs a tool
call and returns the same version. Only the simulator stepping — an executed
waypoint — ends an observation's life.

Extraction spans several browser reads. If a new frame arrives partway through,
the reply is `unknown`, nothing is stored, and both versions are reported — retry
rather than treating the result as a partial observation. After
`execute_waypoint`, refresh anything you intend to rely on.

A UI old enough not to report frame metadata gets `observation_version: null` and
`pairing.reason='no_frame_metadata'`: freshness is then unverifiable, and no
observation is treated as current.

## `geometry_bind_reference`

Pick a point with the same `(u, v)` convention as `hover` — on the 3-D canvas or
on a camera feed — and bind a local geometry reference there.

You get the local centre and axis-aligned extents in metres, the point support
behind them, two extreme cloud returns as keypoints, and a principal axis with
its extent **only** when the neighbourhood is genuinely elongated. When support
is too thin the reply is `status: "unknown"` with no coordinates: a sparse return
is evidence of occlusion or sampling, not of a small object. The screenshot marks
the extracted centre rather than your raw click, so you can check the two agree
before trusting the numbers.

## `geometry_refresh_reference`

Re-extracts a bound reference from the current observation. Called while the
simulator is still idle it is a re-read of the same observation the reference was
extracted from, and `same_observation_as_stored_geometry` says so; that is not
tracking and it is the case where the reference stays usable. Once a waypoint has
executed, the reference's observation is gone and no refresh can revive it.

* `status: "ok"` — a local match was found within the stated motion and extent
  gates; the reply gives the new centre and how far it moved.
* `status: "unknown"` — occlusion, too few returns, an ambiguous match, motion
  beyond the gate, a frame whose parts cannot be shown to have been measured
  together, or a new frame arriving mid-extraction. The stored coordinates come
  back explicitly labelled stale. They are the last place the reference was seen,
  not where anything is now.

Every `unknown` retires the reference. A refresh that could not produce current
geometry — including one that could not run at all — leaves the old numbers under
`stale_*` names only, and the proxy refuses the reference from then on. Bind a new
one visually; nothing local can re-establish the old association.

Matching is deliberately conservative: it re-extracts near the last known centre.
It will not follow an object across a large displacement, and a similar-looking
neighbour nearby can make a match ambiguous. When that happens, look at the
images and re-bind rather than assuming continuity.

A successful refresh also reports `region_comparison`: how far the *neighbourhood*
centre moved between the two extractions, with `identity: "unknown"`. It is a
region observation, not object tracking and not an endpoint error.

## `geometry_proxy_check`

A bounded geometric check of a candidate pose. It takes the same inputs and obeys
the same limits as `edit_target` (0.1 m per position component, 90 degrees of
orientation change), but changes nothing at all — not the robot, not the virtual
target. It approves nothing: it reports what the sampled cloud along the candidate
segment does and does not show. No verdict from this check is a clearance
guarantee or a safety recommendation. To act on a candidate
anyway, call `edit_target` and then `execute_waypoint` — that decision is yours,
not the tool's.

It reports:

* the candidate pose and the **measured** end effector separately, plus the
  translation and approach angle between them. These are different things: the
  candidate is a hypothesis, the measurement is what the robot is actually doing.
* `min_sampled_point_distance_m`: the distance to the nearest **sampled** cloud
  return along a straight segment from the measured fingertip to the candidate.
  Read the name literally. The cloud is downsampled, so the real surface can be
  nearer than the nearest sampled point; this is not a clearance bound and not a
  collision proof. The segment is also a first-order approximation of the
  controller's interpolation, not the trajectory it will execute.
* which samples have sampled returns inside the swept radius.
* which samples are **unknown**, for either of two reasons: no returns nearby
  (unobserved — empty space and unseen space look identical in a point cloud), or
  inside the fingertip self-exclusion sphere.
* optionally, `reference_span_along_opening_m` against the jaw span. Read this one
  literally too: it is the extent of the reference's **sampled** points along your
  candidate opening axis, measured from the one viewpoint that saw them. A face
  turned away from every depth camera contributes nothing, so the real object can
  be wider than the span reported, and `within_jaw_span` is a comparison of two
  numbers rather than a graspability claim.
* optionally, the single-frame cues bearing on whether a referenced object moves
  with the gripper. This always comes back `unknown`, and nothing in this tool set
  changes that. A closed gripper next to an object looks the same as a held object
  in one frame; co-displacement across two observations would be evidence, but a
  refresh on a later observation does not supply it — it retires the reference and
  returns `geometry: null`, precisely because re-extracting near the old centre
  cannot show the same object is there. Whether something is held is a judgement
  you make from the images.

Cloud points close to the measured fingertip are dropped, since they are likely
the gripper's own surface. That sphere is blunt and deletes real objects too, so
samples inside it are reported unknown rather than clear — no verdict from this
tool means "clear", "safe" or "collision-free".

## Predictions get checked

Every candidate `geometry_proxy_check` evaluates is recorded, with or without a
reference. The comparison happens at `execute_waypoint`: that call reads the target
the controller actually received and reports the endpoint check with its own reply.
That is the only place an endpoint error is computed, and it is computed once.

The check reports `not_verifiable` — never an error figure — unless all of the
following hold: the executed target matches the candidate to round-trip precision
in both position and orientation; no other `execute_waypoint` call intervened; the
execution started from the very frame the candidate was evaluated against; the
observation after it is a strictly later frame of the same episode; and exactly one
waypoint completed across it. Anything else and the two things being subtracted are
not the same waypoint.

`geometry_refresh_reference` afterwards repeats that stored result, marked with its
source, or says no execution has been associated with the prediction. It does not
recompute an endpoint error of its own. A prediction is never reported as confirmed
just because it was written down. If you want to know whether something worked,
execute and then look — at the endpoint check, the refreshed geometry, and fresh
camera images.

## Limits, stated plainly

The tools give you coarse geometry on currently visible surfaces. They cannot
prove a path is collision-free, cannot predict contact or force, cannot tell you
that an object is held, and cannot see behind anything. They are a way to read
the cloud numerically instead of by eye — the judgment stays yours.
