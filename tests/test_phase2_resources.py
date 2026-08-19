import mujoco
import numpy as np

from seqgrasp.experiments.resource_components import PALM_REFERENCE_TO_COMPILED, _point_inside_geom, occupied_fingers


def test_occupied_finger_threshold_is_strict_and_ordered():
    count, mask = occupied_fingers([0.21, 0.20, 0.0, 1.0], 0.20)
    assert count == 2
    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_actual_box_and_capsule_point_occupancy():
    xml = """<mujoco><worldbody>
      <geom name='box' type='box' size='1 2 3'/>
      <geom name='cap' type='capsule' size='.5 1' pos='4 0 0'/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cap")
    np.testing.assert_array_equal(_point_inside_geom(model, data, box, np.array([[0, 0, 0], [2, 0, 0]])), [True, False])
    np.testing.assert_array_equal(_point_inside_geom(model, data, cap, np.array([[4, 0, 1.4], [4, 0, 1.6]])), [True, False])


def test_PI_reference_box_axis_mapping_is_a_volume_preserving_rotation():
    np.testing.assert_allclose(PALM_REFERENCE_TO_COMPILED @ PALM_REFERENCE_TO_COMPILED.T, np.eye(3))
    assert np.isclose(np.linalg.det(PALM_REFERENCE_TO_COMPILED), 1.0)
    np.testing.assert_allclose(PALM_REFERENCE_TO_COMPILED @ [1.0, 2.0, 3.0], [-3.0, 2.0, 1.0])
