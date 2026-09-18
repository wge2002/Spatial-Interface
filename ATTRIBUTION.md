# Source attribution and migration

Spatial Interface is derived from [VIA by Hengyuan Hu](https://github.com/hengyuan-hu/via)
and subsequent local work on visual-geometric robot interfaces. Upstream robot
simulation, browser interaction, and agent harness code remain part of this work.
This repository does not claim exclusive authorship of the inherited code.

The initial implementation was imported from the local DG-r5 integration source
`6f8c6b40a15258408616903f278269c11419c901`, with method anchor
`01fe3a2c48f69519c0663e72d1ada5644423fc6c`. Those identifiers describe provenance;
their Git objects and original experiment history are not included here.
Historical manifests, version tags and results remain in the separate source archive.

The new Python namespace is `spatial_interface`. New portable setup, experiment
entrypoints, occupied-port refusal and root-commit identity handling form the new
`si-r1` release. Its identity is distinct from DG-r5; historical evaluation scores
are not presented as evaluations of this new repository.

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) remains a pinned
third-party submodule, with its own authorship and license files. Dependencies
retain their respective terms. No new blanket license is added by this migration.
