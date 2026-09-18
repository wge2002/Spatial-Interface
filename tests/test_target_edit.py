"""Target geometry contracts; these run without the simulator or model APIs."""

import math
import unittest

from spatial_interface.target_edit import control_interface, prepare_edit, validate_edit


def pose():
    return {
        "robot_position": dict(x=0.2, y=-0.1, z=1.0),
        "ui_position": dict(x=2.0, y=2.0, z=1.0),
        "robot_approach": dict(x=0.0, y=0.0, z=-1.0),
        "robot_opening": dict(x=0.0, y=-1.0, z=0.0),
        "gripper_open": True,
    }


class TargetGeometryTests(unittest.TestCase):
    def test_position_and_orientation_map_to_fingertip_frame(self):
        result = prepare_edit(validate_edit({
            "frame": "robot", "delta_position": [0.03, -0.02, 0.04],
            "approach": [0, 0, -1], "opening": [1, 0, 0],
        }), pose(), 0.8)
        for actual, expected in zip(result["position"], [2.3, 2.4, 1.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["basis"], [[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    def test_absolute_and_relative_positions_agree(self):
        a = prepare_edit(validate_edit({"frame": "robot", "position": [0.23, -0.1, 1]}), pose(), 0.8)
        b = prepare_edit(validate_edit({"frame": "robot", "delta_position": [0.03, 0, 0]}), pose(), 0.8)
        self.assertEqual(a, b)

    def test_tilted_orientation_stays_orthonormal(self):
        s = math.sqrt(0.5)
        result = prepare_edit(validate_edit({"frame": "robot", "approach": [s, 0, -s],
                                             "opening": [0, -1, 0]}), pose(), 0.8)
        basis = result["basis"]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(sum(x * y for x, y in zip(basis[i], basis[j])), i == j)

    def test_invalid_requests_are_rejected(self):
        bad = [
            {}, {"frame": "camera", "position": [0, 0, 1]}, {"frame": "robot"},
            {"frame": "robot", "delta_position": [False, 0, 0]},
            {"frame": "robot", "delta_position": [float("nan"), 0, 0]},
            {"frame": "robot", "delta_position": [float("inf"), 0, 0]},
            {"frame": "robot", "delta_position": ["0", 0, 0]},
            {"frame": "robot", "delta_position": [0, 0]},
            {"frame": "robot", "delta_position": [0.1001, 0, 0]},
            {"frame": "robot", "position": [0, 0, 1], "delta_position": [0, 0, 0]},
            {"frame": "robot", "approach": [0, 0, -1]},
            {"frame": "robot", "approach": [0, 0, -1], "opening": [0, 0, 1]},
            {"frame": "robot", "approach": [0, 0, -2], "opening": [0, 1, 0]},
            {"frame": "robot", "position": [0, 0, 1], "execute": True},
            {"frame": "robot", "position": [0, 0, 1], "gripper_open": False},
        ]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_edit(request)

    def test_absolute_step_and_rotation_limits(self):
        for request in (
            {"frame": "robot", "position": [1, 0, 1]},
            {"frame": "robot", "approach": [0, 0, 1], "opening": [0, -1, 0]},
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                prepare_edit(validate_edit(request), pose(), 0.8)

    def test_only_small_rounding_error_is_corrected(self):
        edit = validate_edit({"frame": "robot", "approach": [0, 0, -1.0001],
                              "opening": [0, -1, 0.0001]})
        self.assertEqual(edit["approach"], [0, 0, -1])
        self.assertEqual(edit["opening"], [0, -1, 0])

    def test_mode_requires_explicit_opt_in(self):
        self.assertEqual(control_interface({}), "legacy")
        self.assertEqual(control_interface({"VIA_CONTROL_INTERFACE": "compact"}), "compact")
        with self.assertRaises(ValueError):
            control_interface({"VIA_CONTROL_INTERFACE": "typo"})


if __name__ == "__main__":
    unittest.main()
