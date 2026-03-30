import numpy as np
import pytest

from nsbutils.plotting_pyvista import _prepare_vertex_scalars, _rh_view_swap


def test_prepare_vertex_scalars_vertexwise_no_rois():
    n = 5
    data = np.arange(n)
    scalars, roi_labels = _prepare_vertex_scalars(data, rois=None, n_verts=n)
    assert roi_labels is None
    assert scalars.shape == (n,)
    assert np.allclose(scalars, data)


def test_prepare_vertex_scalars_roiwise_expands_and_masks_zero():
    rois = np.array([0, 1, 1, 2, 2])
    data_roi = np.array([10.0, 20.0])
    scalars, roi_labels = _prepare_vertex_scalars(data_roi, rois=rois, n_verts=rois.size)

    assert roi_labels is not None
    assert scalars.shape == (rois.size,)
    assert np.isnan(scalars[0])
    assert np.allclose(scalars[1:3], 10.0)
    assert np.allclose(scalars[3:5], 20.0)


def test_prepare_vertex_scalars_raises_on_bad_shapes():
    rois = np.array([1, 1, 2, 2])
    with pytest.raises(ValueError):
        _prepare_vertex_scalars(np.array([1, 2, 3]), rois=None, n_verts=rois.size)

    with pytest.raises(ValueError):
        _prepare_vertex_scalars(np.array([1, 2, 3]), rois=rois, n_verts=rois.size)


@pytest.mark.parametrize(
    "view,expected",
    [
        ("lateral", "medial"),
        ("medial", "lateral"),
        ("dorsal", "ventral"),
        ("ventral", "dorsal"),
        ("anterior", "posterior"),
        ("posterior", "anterior"),
        ("other", "other"),
    ],
)
def test_rh_view_swap(view, expected):
    assert _rh_view_swap(view) == expected
