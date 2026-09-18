# Direct geometric control: grasp-only development diagnostic

Grasp the bowl, lift it at least 5 cm above its initial support, hold it for
inspection using actual simulation steps, then end the episode. This diagnostic
ends with the bowl held; the environment's original bowl-on-plate reward is a
separate outcome.

Use the visible proxy/point cloud to author a short coarse policy that brings the
gripper near your chosen part of the bowl at your chosen orientation. Return from
that policy before the final approach and grasp. Inspect its fresh image, wrist
depth and measured pose, then author the fine adjustment, close, lift and hold.
Aim for two or three policy calls when the evidence supports it. If more calls
are necessary, use them and report the actual outcome; do not call an uncertain
grasp successful to meet the call target.

Every geometric choice and recovery is yours. The proxy is a fit from its bound
frame and does not track an object after it moves. Local depth includes visible
gripper and background returns as well as objects: its nearest point, centroid
and spread do not identify the bowl or measure contact. Combine the image with
the metric samples to decide which surfaces they describe.

Use only the exposed robot tools as environmental evidence. Do not inspect
simulator files, saved state, recordings, source, object poses or hidden success
data. Offline state and video are for evaluation after the episode. End when the
task is achieved or when you judge that it cannot be completed in this attempt.

The proxy is an editable geometric reference, not an actuator or an object-state
truth. An explicit policy executes real robot controller commands. You choose all
poses, orientations, gripper actions and observation boundaries. Supplied camera
intrinsics/extrinsics and measured robot proprioception are allowed; final task
evaluation returned by end_episode is terminal, not guidance for further actions.
