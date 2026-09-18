import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from scipy.spatial.transform import Rotation
from spatial_interface import dg_feedback as f


def view(seq=1, epoch="smoke", steps=10):
    return dict(frame=f"e{epoch}-s{seq}-c{seq}", sim_steps_used=steps,
                measured_end_effector=dict(fingertip_position=dict(x=0.,y=0.,z=1.),
                    approach=dict(x=0,y=0,z=-1), opening=dict(x=0,y=1,z=0),
                    gripper_width_m=.006, gripper_state_class="closed"))


class Feedback(unittest.TestCase):
    def test_projection_uses_camera_extrinsics_and_rejects_invisible_points(self):
        e=np.eye(4); e[:3,:3]=Rotation.from_euler("xyz",[.2,.7,-.4]).as_matrix()
        e[:3,3]=[.3,-.1,1.2]
        calibration=dict(E=e.flatten().tolist(),K=[100,0,100,0,100,100,0,0,1],img_size=200)
        for local,expected in (([0,0,1,1],(.5,.5)),([.2,-.4,1,1],(.6,.3)),
                               ([0,0,-1,1],None),([3,0,1,1],None)):
            result=f.project((e@local)[:3],calibration)
            if expected is None:
                self.assertIsNone(result)
            else:
                np.testing.assert_allclose(result,expected,atol=1e-12)

    def test_temporal_rejects_old_frame_cross_episode_and_nonadvancing_sim(self):
        before=dict(view=view())
        for after,status in ((dict(view=view()),"no_new_frame"),
                             (dict(view=view(2,"other",30)),"unknown"),
                             (dict(view=view(2,steps=10)),"unknown")):
            self.assertEqual(f.temporal_evidence(before,after)["status"],status)
        after=dict(view=view(2,steps=30)); after["view"]["measured_end_effector"]["fingertip_position"]["z"]+=.1
        evidence=f.temporal_evidence(before,after)
        self.assertEqual(evidence["measured_eef_delta_m"]["z"],.1)
        self.assertNotIn("grasped",evidence)

    def test_capture_rejects_frame_race(self):
        ctx=SimpleNamespace(observe_geometry=AsyncMock(side_effect=[{},{}]),
                            page=SimpleNamespace(evaluate=AsyncMock(return_value={})))
        with patch.object(f.geom,"observation_version",side_effect=["esmoke-s1-c1","esmoke-s2-c2"]), \
             patch.object(f.geom,"pairing_state",return_value={"status":"paired"}):
            with self.assertRaisesRegex(ValueError,"changed_during_capture"):
                asyncio.run(f.capture(ctx,view()))

    def test_feedback_failure_preserves_acknowledged_execution(self):
        payload=dict(status="completed",executed=[dict(command_id="p/0",completed=True)],observation=view(2,steps=40))
        original=copy.deepcopy(payload)
        with patch.object(f,"capture",AsyncMock(side_effect=ValueError("stale camera"))):
            self.assertEqual(asyncio.run(f.finish(None,payload,"paired")),[])
        self.assertEqual({k:payload[k] for k in original},original)
        self.assertEqual(payload["feedback"]["temporal"]["status"],"unknown")

    def test_native_mcp_forwards_variant_and_bad_variant_fails(self):
        from spatial_interface import codex_harness as h
        with patch.dict("os.environ",VIA_CONTROL_INTERFACE="direct_geometry",VIA_DG_FEEDBACK="paired"):
            self.assertIn('VIA_DG_FEEDBACK="paired"'," ".join(h.codex_mcp_overrides(9040)))
        with patch.dict("os.environ",VIA_DG_FEEDBACK="typo"):
            with self.assertRaises(ValueError):
                f.variant()


if __name__ == "__main__":
    unittest.main()
