from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The UI module imports imgui-bundle (the optional ``ui`` extra). Skip this whole module
# when it is not installed so the core suite still runs with just ``pip install .[dev]``.
pytest.importorskip("imgui_bundle")

from quick_mag.quick_mag_ui import (  # noqa: E402
    AXIS_WIDGET_ARM_PX,
    AXIS_WIDGET_LABEL_SCALE,
    AXIS_WIDGET_MARGIN_PX,
    DEFAULT_STRUCTURE_ROTATION,
    SCREEN_TURN_AXES,
    SCREEN_TURN_AXIS_TOOLTIPS,
    SPIN_RADIUS_REFERENCE_MOMENT,
    STRUCTURE_TURN_STEP_DEGREES,
    plot_axis_directions,
    view_axis_alignment_sign,
    rotation_after_alignment_step,
    rotation_after_drag,
    rotation_after_screen_turn,
    spin_moment_magnitudes,
    spin_scaled_render_radii,
    view_rotation_for_axis,
)


CUBIC_LIMITS = (-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)


def rotate(quaternion, vector: np.ndarray) -> np.ndarray:
    """Apply an (x, y, z, w) quaternion to a vector, independently of ImPlot3D."""
    x, y, z, w = quaternion
    axis = np.array([x, y, z], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return (
        vector * (w * w - axis @ axis)
        + 2.0 * axis * (axis @ vector)
        + 2.0 * w * np.cross(axis, vector)
    )


class PlotAxisDirectionsTests(unittest.TestCase):
    def test_fractional_mode_is_the_identity(self):
        lattice = np.array([[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]])
        directions = plot_axis_directions(lattice, False, CUBIC_LIMITS)
        np.testing.assert_allclose(directions, np.eye(3), atol=1e-12)

    def test_cartesian_mode_normalises_the_lattice_rows(self):
        lattice = np.array([[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]])
        directions = plot_axis_directions(lattice, True, CUBIC_LIMITS)
        np.testing.assert_allclose(directions, np.eye(3), atol=1e-12)

    def test_monoclinic_axis_tilts_away_from_the_box_axis(self):
        # b is 100 degrees off a in the ab plane, so it must not come out as (0, 1, 0).
        angle = np.deg2rad(100.0)
        lattice = np.array(
            [
                [5.0, 0.0, 0.0],
                [7.0 * np.cos(angle), 7.0 * np.sin(angle), 0.0],
                [0.0, 0.0, 11.0],
            ]
        )
        directions = plot_axis_directions(lattice, True, CUBIC_LIMITS)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(directions[0], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            directions[1], [np.cos(angle), np.sin(angle), 0.0], atol=1e-12
        )
        # The lattice angle survives the trip into the box.
        self.assertAlmostEqual(float(directions[0] @ directions[1]), np.cos(angle), places=12)

    def test_degenerate_row_comes_back_as_zero_rather_than_nan(self):
        lattice = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 11.0]])
        directions = plot_axis_directions(lattice, True, CUBIC_LIMITS)
        self.assertFalse(np.isnan(directions).any())
        np.testing.assert_allclose(directions[1], np.zeros(3))

    def test_non_cubic_box_rescales_each_axis(self):
        # A box twice as long in y squashes a 45-degree in-plane vector towards x.
        lattice = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        directions = plot_axis_directions(lattice, True, (-1.0, 1.0, -2.0, 2.0, -1.0, 1.0))
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_allclose(
            directions[0], expected / np.linalg.norm(expected), atol=1e-12
        )


class ViewRotationTests(unittest.TestCase):
    def assert_unit_quaternion(self, quaternion):
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(quaternion))), 1.0, places=12)

    def test_each_axis_swings_onto_the_viewer(self):
        directions = np.eye(3)
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                quaternion = view_rotation_for_axis(directions, axis_index)
                self.assert_unit_quaternion(quaternion)
                # +z is out of the screen: the chosen axis points at the viewer.
                np.testing.assert_allclose(
                    rotate(quaternion, directions[axis_index]), [0.0, 0.0, 1.0], atol=1e-12
                )

    def test_c_points_up_the_screen(self):
        directions = np.eye(3)
        for axis_index in (0, 1):
            with self.subTest(axis=axis_index):
                quaternion = view_rotation_for_axis(directions, axis_index)
                np.testing.assert_allclose(
                    rotate(quaternion, directions[2]), [0.0, 1.0, 0.0], atol=1e-12
                )

    def test_looking_down_c_gives_the_ab_plane_with_a_to_the_right(self):
        directions = np.eye(3)
        quaternion = view_rotation_for_axis(directions, 2)
        np.testing.assert_allclose(
            rotate(quaternion, directions[0]), [1.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            rotate(quaternion, directions[1]), [0.0, 1.0, 0.0], atol=1e-12
        )

    def test_looking_down_b_puts_a_to_the_left(self):
        # Forced, not chosen: with c up and b at the viewer, a right-handed cell has
        # nowhere else to put a.
        directions = np.eye(3)
        quaternion = view_rotation_for_axis(directions, 1)
        np.testing.assert_allclose(
            rotate(quaternion, directions[0]), [-1.0, 0.0, 0.0], atol=1e-12
        )

    def test_triclinic_rotation_stays_orthonormal(self):
        lattice = np.array([[5.1, 0.0, 0.0], [1.3, 6.2, 0.0], [0.7, -1.1, 9.4]])
        directions = plot_axis_directions(lattice, True, CUBIC_LIMITS)
        quaternion = view_rotation_for_axis(directions, 1)
        self.assert_unit_quaternion(quaternion)
        np.testing.assert_allclose(
            rotate(quaternion, directions[1]), [0.0, 0.0, 1.0], atol=1e-12
        )
        # The rotation is rigid: angles between the lattice vectors are untouched.
        rotated = np.vstack([rotate(quaternion, row) for row in directions])
        np.testing.assert_allclose(rotated @ rotated.T, directions @ directions.T, atol=1e-12)

    def test_collinear_axes_fall_back_without_nan(self):
        directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        quaternion = view_rotation_for_axis(directions, 0)
        self.assertFalse(np.isnan(np.asarray(quaternion)).any())
        self.assert_unit_quaternion(quaternion)
        np.testing.assert_allclose(
            rotate(quaternion, directions[0]), [0.0, 0.0, 1.0], atol=1e-12
        )

    def test_degenerate_axis_is_the_identity(self):
        directions = np.zeros((3, 3))
        self.assertEqual(view_rotation_for_axis(directions, 0), (0.0, 0.0, 0.0, 1.0))


class AxisFlipTests(unittest.TestCase):
    def test_the_opposite_sign_looks_at_the_other_face(self):
        directions = np.eye(3)
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                quaternion = view_rotation_for_axis(directions, axis_index, -1)
                # The axis now points away from the viewer: it is the far face on screen.
                np.testing.assert_allclose(
                    rotate(quaternion, directions[axis_index]),
                    [0.0, 0.0, -1.0],
                    atol=1e-12,
                )

    def test_c_still_points_up_when_looking_from_the_other_side(self):
        directions = np.eye(3)
        for axis_index in (0, 1):
            with self.subTest(axis=axis_index):
                quaternion = view_rotation_for_axis(directions, axis_index, -1)
                np.testing.assert_allclose(
                    rotate(quaternion, directions[2]), [0.0, 1.0, 0.0], atol=1e-12
                )

    def test_first_press_aligns_and_the_second_flips(self):
        directions = np.eye(3)
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                first = view_axis_alignment_sign(
                    DEFAULT_STRUCTURE_ROTATION, directions, axis_index
                )
                self.assertEqual(first, 1)
                aligned = view_rotation_for_axis(directions, axis_index, first)
                second = view_axis_alignment_sign(aligned, directions, axis_index)
                self.assertEqual(second, -1)
                # And a third press comes back to the near face.
                flipped = view_rotation_for_axis(directions, axis_index, second)
                self.assertEqual(
                    view_axis_alignment_sign(flipped, directions, axis_index), 1
                )

    def test_a_different_button_always_aligns_rather_than_flipping(self):
        directions = np.eye(3)
        aligned_to_a = view_rotation_for_axis(directions, 0, 1)
        for axis_index in (1, 2):
            with self.subTest(axis=axis_index):
                self.assertEqual(
                    view_axis_alignment_sign(aligned_to_a, directions, axis_index), 1
                )

    def test_rotating_well_away_makes_the_button_align_again(self):
        directions = np.eye(3)
        rotation = view_rotation_for_axis(directions, 2, 1)
        # A deliberate drag, not a nudge.
        for _ in range(20):
            rotation = rotation_after_drag(rotation, 12.0, 0.0)
        self.assertEqual(view_axis_alignment_sign(rotation, directions, 2), 1)

    def test_a_tiny_nudge_still_counts_as_aligned(self):
        directions = np.eye(3)
        rotation = rotation_after_drag(view_rotation_for_axis(directions, 1, 1), 1.0, 0.0)
        self.assertEqual(view_axis_alignment_sign(rotation, directions, 1), -1)

    def test_non_orthogonal_lattice_flips_on_the_lattice_vector(self):
        lattice = np.array([[5.1, 0.0, 0.0], [1.3, 6.2, 0.0], [0.7, -1.1, 9.4]])
        directions = plot_axis_directions(lattice, True, CUBIC_LIMITS)
        aligned = view_rotation_for_axis(directions, 1, 1)
        self.assertEqual(view_axis_alignment_sign(aligned, directions, 1), -1)
        flipped = view_rotation_for_axis(directions, 1, -1)
        np.testing.assert_allclose(
            rotate(flipped, directions[1]), [0.0, 0.0, -1.0], atol=1e-12
        )


class DefaultRotationTests(unittest.TestCase):
    def test_the_starting_pose_is_an_exact_rotation(self):
        # Composed onto by every drag and roll, so anything off unit length would let the
        # view creep on the first interaction.
        self.assertAlmostEqual(
            float(np.linalg.norm(np.asarray(DEFAULT_STRUCTURE_ROTATION))), 1.0, places=15
        )


class ScreenTurnAxisTests(unittest.TestCase):
    """x is right, y is up and z is *into* the screen, which is what the buttons say."""

    def test_the_axes_are_right_up_and_into_the_screen(self):
        np.testing.assert_allclose(SCREEN_TURN_AXES[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(SCREEN_TURN_AXES[1], [0.0, 1.0, 0.0])
        # ImPlot3D's view frame has z out of the screen, so "into" is its negative.
        np.testing.assert_allclose(SCREEN_TURN_AXES[2], [0.0, 0.0, -1.0])

    def test_each_axis_is_a_unit_vector(self):
        for index in range(3):
            with self.subTest(axis=index):
                self.assertAlmostEqual(
                    float(np.linalg.norm(np.asarray(SCREEN_TURN_AXES[index]))),
                    1.0,
                    places=12,
                )

    def test_there_is_one_axis_per_button_and_one_tooltip_each(self):
        self.assertEqual(len(SCREEN_TURN_AXES), 3)
        self.assertEqual(len(SCREEN_TURN_AXIS_TOOLTIPS), 3)


class ScreenTurnTests(unittest.TestCase):
    IDENTITY = (0.0, 0.0, 0.0, 1.0)

    def turned(self, axis_index, degrees, point):
        return rotate(
            rotation_after_screen_turn(
                self.IDENTITY, SCREEN_TURN_AXES[axis_index], degrees
            ),
            point,
        )

    def test_turning_about_z_spins_the_picture_clockwise(self):
        # z points away from the viewer, so the right-hand rule about it reads clockwise
        # on screen: a point out to the right drops towards the bottom.
        moved = self.turned(2, 5.0, [1.0, 0.0, 0.0])
        self.assertLess(moved[1], 0.0)
        self.assertAlmostEqual(float(moved[2]), 0.0, places=12)
        # And the negative button undoes it.
        self.assertGreater(self.turned(2, -5.0, [1.0, 0.0, 0.0])[1], 0.0)

    def test_turning_about_x_tips_the_top_towards_the_viewer(self):
        self.assertGreater(self.turned(0, 5.0, [0.0, 1.0, 0.0])[2], 0.0)
        self.assertLess(self.turned(0, -5.0, [0.0, 1.0, 0.0])[2], 0.0)

    def test_turning_about_y_swings_the_near_face_right(self):
        self.assertGreater(self.turned(1, 5.0, [0.0, 0.0, 1.0])[0], 0.0)
        self.assertLess(self.turned(1, -5.0, [0.0, 0.0, 1.0])[0], 0.0)

    def test_the_axis_itself_is_left_where_it_is(self):
        # What "turning about an axis" means: everything else moves, the axis does not.
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                axis = np.asarray(SCREEN_TURN_AXES[axis_index], dtype=np.float64)
                turned = rotation_after_screen_turn(self.IDENTITY, axis, 5.0)
                np.testing.assert_allclose(rotate(turned, axis), axis, atol=1e-12)

    def test_turning_about_z_leaves_the_axis_you_are_looking_down_alone(self):
        # Which is why the screen normal is the default: it composes with an alignment
        # rather than undoing it.
        directions = np.eye(3)
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                aligned = view_rotation_for_axis(directions, axis_index)
                turned = rotation_after_screen_turn(aligned, SCREEN_TURN_AXES[2], 5.0)
                np.testing.assert_allclose(
                    rotate(turned, directions[axis_index]), [0.0, 0.0, 1.0], atol=1e-12
                )
                self.assertEqual(
                    view_axis_alignment_sign(turned, directions, axis_index), -1
                )

    def test_a_full_turn_of_steps_comes_back_to_the_start(self):
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                rotation = DEFAULT_STRUCTURE_ROTATION
                for _ in range(int(round(360.0 / STRUCTURE_TURN_STEP_DEGREES))):
                    rotation = rotation_after_screen_turn(
                        rotation,
                        SCREEN_TURN_AXES[axis_index],
                        STRUCTURE_TURN_STEP_DEGREES,
                    )
                # The same pose, though the quaternion may be its own negative.
                np.testing.assert_allclose(
                    np.abs(np.asarray(rotation)),
                    np.abs(np.asarray(DEFAULT_STRUCTURE_ROTATION)),
                    atol=1e-9,
                )

    def test_opposite_presses_cancel(self):
        axis = SCREEN_TURN_AXES[0]
        there = rotation_after_screen_turn(DEFAULT_STRUCTURE_ROTATION, axis, 5.0)
        back = rotation_after_screen_turn(there, axis, -5.0)
        np.testing.assert_allclose(
            np.asarray(back), np.asarray(DEFAULT_STRUCTURE_ROTATION), atol=1e-12
        )


class AxisWidgetLayoutTests(unittest.TestCase):
    def test_the_triad_is_inset_far_enough_not_to_clip(self):
        # The draw list is clipped to the plot rect, so the origin has to sit at least a
        # full label reach in from the corner or an arm aimed at it loses its letter.
        self.assertGreater(
            AXIS_WIDGET_MARGIN_PX, AXIS_WIDGET_ARM_PX * AXIS_WIDGET_LABEL_SCALE
        )


class TrackballTests(unittest.TestCase):
    """The point of owning the rotation: no orientation where a drag stops working."""

    def test_no_motion_is_no_rotation(self):
        self.assertEqual(
            rotation_after_drag(DEFAULT_STRUCTURE_ROTATION, 0.0, 0.0),
            DEFAULT_STRUCTURE_ROTATION,
        )

    def test_dragging_right_swings_the_near_face_right(self):
        identity = (0.0, 0.0, 0.0, 1.0)
        # The near face is at +z; dragging right must carry it towards +x.
        turned = rotate(rotation_after_drag(identity, 20.0, 0.0), [0.0, 0.0, 1.0])
        self.assertGreater(turned[0], 0.0)
        self.assertAlmostEqual(float(turned[1]), 0.0, places=12)

    def test_dragging_down_tips_the_near_face_down(self):
        identity = (0.0, 0.0, 0.0, 1.0)
        # Mouse dy points down the screen, so the near face must go towards -y.
        turned = rotate(rotation_after_drag(identity, 0.0, 20.0), [0.0, 0.0, 1.0])
        self.assertLess(turned[1], 0.0)
        self.assertAlmostEqual(float(turned[0]), 0.0, places=12)

    def test_every_axis_aligned_view_still_turns_in_both_directions(self):
        # The regression this replaces ImPlot3D's turntable for: looking straight down an
        # axis, a horizontal and a vertical drag must move the view in different ways.
        directions = np.eye(3)
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                aligned = view_rotation_for_axis(directions, axis_index)
                near = np.array([0.0, 0.0, 1.0])
                horizontal = rotate(rotation_after_drag(aligned, 25.0, 0.0), near)
                vertical = rotate(rotation_after_drag(aligned, 0.0, 25.0), near)
                # Both drags move the near face, and they move it somewhere different.
                self.assertGreater(float(np.linalg.norm(horizontal - near)), 0.05)
                self.assertGreater(float(np.linalg.norm(vertical - near)), 0.05)
                self.assertGreater(float(np.linalg.norm(horizontal - vertical)), 0.05)

    def test_rotation_stays_normalised_over_a_long_drag(self):
        rotation = DEFAULT_STRUCTURE_ROTATION
        for _ in range(500):
            rotation = rotation_after_drag(rotation, 7.0, -3.0)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(rotation))), 1.0, places=9)


class AlignmentAnimationTests(unittest.TestCase):
    def test_a_step_moves_towards_the_target(self):
        target = view_rotation_for_axis(np.eye(3), 2)
        stepped, arrived = rotation_after_alignment_step(
            DEFAULT_STRUCTURE_ROTATION, target, 1.0 / 60.0
        )
        self.assertFalse(arrived)
        before = abs(float(np.asarray(DEFAULT_STRUCTURE_ROTATION) @ np.asarray(target)))
        after = abs(float(np.asarray(stepped) @ np.asarray(target)))
        self.assertGreater(after, before)

    def test_the_swing_finishes_and_reports_arrival(self):
        target = view_rotation_for_axis(np.eye(3), 0)
        rotation = DEFAULT_STRUCTURE_ROTATION
        for _ in range(200):
            rotation, arrived = rotation_after_alignment_step(
                rotation, target, 1.0 / 60.0
            )
            if arrived:
                break
        self.assertTrue(arrived)
        self.assertEqual(rotation, target)

    def test_a_slow_frame_covers_as_much_ground_as_several_fast_ones(self):
        target = view_rotation_for_axis(np.eye(3), 1)
        slow, _ = rotation_after_alignment_step(DEFAULT_STRUCTURE_ROTATION, target, 0.1)
        fast = DEFAULT_STRUCTURE_ROTATION
        for _ in range(6):
            fast, _ = rotation_after_alignment_step(fast, target, 0.1 / 6.0)
        np.testing.assert_allclose(np.abs(np.asarray(slow)), np.abs(np.asarray(fast)), atol=1e-6)


class SpinMomentMagnitudeTests(unittest.TestCase):
    def test_scalar_moments_use_absolute_value(self):
        magnitudes = spin_moment_magnitudes(np.array([5.0, -5.0, 0.0]), 3)
        np.testing.assert_allclose(magnitudes, [5.0, 5.0, 0.0])

    def test_vector_moments_use_the_row_norm(self):
        moments = np.array([[0.0, 0.0, -4.0], [3.0, 4.0, 0.0]])
        np.testing.assert_allclose(spin_moment_magnitudes(moments, 2), [4.0, 5.0])

    def test_short_arrays_are_zero_padded(self):
        np.testing.assert_allclose(
            spin_moment_magnitudes(np.array([5.0, -5.0]), 5), [5.0, 5.0, 0.0, 0.0, 0.0]
        )

    def test_long_arrays_are_truncated(self):
        np.testing.assert_allclose(
            spin_moment_magnitudes(np.array([5.0, -5.0, 1.0]), 2), [5.0, 5.0]
        )

    def test_none_and_unrecognised_shapes(self):
        self.assertIsNone(spin_moment_magnitudes(None, 4))
        self.assertIsNone(spin_moment_magnitudes(np.zeros((2, 4)), 2))


class SpinScaledRadiiTests(unittest.TestCase):
    def test_ratio_of_moments_is_the_ratio_of_radii(self):
        radii = np.array([0.6, 0.6])
        scaled = spin_scaled_render_radii(radii, np.array([5.0, 1.0]))
        self.assertAlmostEqual(float(scaled[0] / scaled[1]), 5.0, places=12)

    def test_reference_moment_keeps_the_element_radius(self):
        radii = np.array([0.645])
        scaled = spin_scaled_render_radii(
            radii, np.array([SPIN_RADIUS_REFERENCE_MOMENT])
        )
        np.testing.assert_allclose(scaled, radii)

    def test_zero_moment_is_hidden(self):
        scaled = spin_scaled_render_radii(np.array([0.6, 1.3]), np.array([4.0, 0.0]))
        self.assertEqual(float(scaled[1]), 0.0)
        self.assertAlmostEqual(float(scaled[0]), 0.6 * 4.0 / SPIN_RADIUS_REFERENCE_MOMENT)

    def test_moments_above_the_reference_are_not_clamped(self):
        scaled = spin_scaled_render_radii(np.array([0.94]), np.array([7.0]))
        self.assertGreater(float(scaled[0]), 0.94)

    def test_all_zero_moments_leave_the_view_alone(self):
        radii = np.array([0.6, 1.3])
        np.testing.assert_allclose(
            spin_scaled_render_radii(radii, np.zeros(2)), radii
        )

    def test_missing_or_mismatched_magnitudes_leave_the_view_alone(self):
        radii = np.array([0.6, 1.3])
        np.testing.assert_allclose(spin_scaled_render_radii(radii, None), radii)
        np.testing.assert_allclose(
            spin_scaled_render_radii(radii, np.array([5.0])), radii
        )


if __name__ == "__main__":
    unittest.main()
