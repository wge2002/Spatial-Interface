# Grasp and hold diagnostic

Grasp the bowl, lift it at least 5 cm above its initial support, and hold it in
the air for at least 3 seconds using actual simulation steps. End the episode
while still holding the bowl. The environment's original bowl-on-plate score is
a separate outcome and is not the success criterion for this grasp diagnostic.

Choose the geometry, poses, orientation, action sequence, observation boundaries
and any recovery yourself. Use as many tool calls as the evidence requires.
Use only the exposed robot tools as environmental evidence: RGB-D, point clouds,
supplied camera calibration and robot proprioception. Do not inspect simulator
files, source, recordings, hidden object states, contacts or success endpoints.
The final evaluation returned by end_episode ends the attempt; do not use it to
generate further actions. Report uncertainty honestly.
