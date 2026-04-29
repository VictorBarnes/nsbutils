import numpy as np
import pytest

from nsbutils.plotting_pyvista import (
    _auto_gradient_vector_max_length,
    _clip_vectors_to_max_length,
    _prepare_triangle_vectors,
    _normalize_maps_clim,
    _normalize_video_clim,
    _prepare_timeseries_scalars,
    _prepare_vertex_scalars,
    _rh_view_swap,
    _validate_clim,
    downsample_triangle_vectors_by_vertex_mask,
)


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


def test_validate_clim_accepts_tuple():
    assert _validate_clim((-1, 1)) == (-1.0, 1.0)


def test_validate_clim_accepts_numpy_array():
    assert _validate_clim(np.array([-1, 1])) == (-1.0, 1.0)


@pytest.mark.parametrize("clim", [(1, 1), (2, -1), (np.nan, 1), (0, np.inf)])
def test_validate_clim_raises_on_invalid(clim):
    with pytest.raises(ValueError):
        _validate_clim(clim)


def test_prepare_timeseries_scalars_vertexwise_masks_medial_wall():
    rois = np.array([0, 1, 1, 2, 2])
    ts = np.arange(rois.size * 3, dtype=float).reshape(rois.size, 3)
    vts, roi_labels = _prepare_timeseries_scalars(ts, rois=rois, n_verts=rois.size)

    assert roi_labels is not None
    assert vts.shape == ts.shape
    assert np.allclose(vts[1:, :], ts[1:, :])
    assert np.all(np.isnan(vts[0, :]))


def test_prepare_timeseries_scalars_roiwise_expands():
    # ROI labels: {0,1,1,2,2}
    rois = np.array([0, 1, 1, 2, 2])
    # ROI-wise timeseries: 2 ROIs x 3 frames
    ts_roi = np.array(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ]
    )

    vts, roi_labels = _prepare_timeseries_scalars(ts_roi, rois=rois, n_verts=rois.size)
    assert roi_labels is not None
    assert vts.shape == (rois.size, 3)
    assert np.all(np.isnan(vts[0, :]))
    assert np.allclose(vts[1:3, :], ts_roi[0, :])
    assert np.allclose(vts[3:5, :], ts_roi[1, :])


def test_prepare_timeseries_scalars_vertexwise_3d_masks_medial_wall():
    rois = np.array([0, 1, 1, 2, 2])
    ts = np.arange(rois.size * 3 * 2, dtype=float).reshape(rois.size, 3, 2)
    vts, roi_labels = _prepare_timeseries_scalars(ts, rois=rois, n_verts=rois.size)

    assert roi_labels is not None
    assert vts.shape == ts.shape
    assert np.allclose(vts[1:, :, :], ts[1:, :, :])
    assert np.all(np.isnan(vts[0, :, :]))


def test_prepare_timeseries_scalars_roiwise_3d_expands():
    rois = np.array([0, 1, 1, 2, 2])
    ts_roi = np.array(
        [
            [[10.0, 100.0], [11.0, 101.0], [12.0, 102.0]],
            [[20.0, 200.0], [21.0, 201.0], [22.0, 202.0]],
        ]
    )
    vts, roi_labels = _prepare_timeseries_scalars(ts_roi, rois=rois, n_verts=rois.size)
    assert roi_labels is not None
    assert vts.shape == (rois.size, 3, 2)
    assert np.all(np.isnan(vts[0, :, :]))
    assert np.allclose(vts[1:3, :, :], ts_roi[0, :, :])
    assert np.allclose(vts[3:5, :, :], ts_roi[1, :, :])


def test_normalize_video_clim_none_uses_global_symmetric():
    # vertex_ts: (n_verts, n_frames)
    vertex_ts = np.array(
        [
            [-2.0, 1.0],
            [3.0, -4.0],
        ]
    )
    clim = _normalize_video_clim(None, n_frames=2, vertex_ts=vertex_ts)
    assert clim == (-4.0, 4.0)


def test_normalize_video_clim_tuple_is_fixed():
    vertex_ts = np.zeros((3, 4), dtype=float)
    clim = _normalize_video_clim((-1, 2), n_frames=4, vertex_ts=vertex_ts)
    assert clim == (-1.0, 2.0)


def test_normalize_video_clim_per_frame_array_validates():
    vertex_ts = np.zeros((2, 3), dtype=float)
    clim_in = np.array(
        [
            [-1.0, 1.0],
            [0.0, 2.0],
            [-2.0, 3.0],
        ]
    )
    clim = _normalize_video_clim(clim_in, n_frames=3, vertex_ts=vertex_ts)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (3, 2)
    assert np.allclose(clim, clim_in)


def test_normalize_video_clim_per_frame_shape_mismatch_raises():
    vertex_ts = np.zeros((2, 3), dtype=float)
    clim_in = np.array([[-1.0, 1.0], [0.0, 2.0]])
    with pytest.raises(ValueError):
        _normalize_video_clim(clim_in, n_frames=3, vertex_ts=vertex_ts)


def test_normalize_video_clim_degenerate_row_expands_and_nan_row_falls_back():
    vertex_ts = np.zeros((2, 2), dtype=float)
    clim_in = np.array(
        [
            [1.0, 1.0],
            [np.nan, np.nan],
        ]
    )
    clim = _normalize_video_clim(clim_in, n_frames=2, vertex_ts=vertex_ts)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (2, 2)
    assert clim[0, 0] < clim[0, 1]
    assert np.allclose(clim[1], [-1.0, 1.0])


def test_normalize_video_clim_partially_nonfinite_row_raises():
    vertex_ts = np.zeros((2, 1), dtype=float)
    clim_in = np.array([[np.nan, 1.0]])
    with pytest.raises(ValueError):
        _normalize_video_clim(clim_in, n_frames=1, vertex_ts=vertex_ts)


def test_normalize_video_clim_fixed_tuple_broadcasts_for_multi_map():
    vertex_ts = np.zeros((2, 3, 4), dtype=float)
    clim = _normalize_video_clim((-1, 2), n_frames=3, n_maps=4, vertex_ts=vertex_ts)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (3, 4, 2)
    assert np.allclose(clim[0, :, :], np.array([[-1.0, 2.0]] * 4))


def test_normalize_video_clim_per_frame_broadcasts_for_multi_map():
    vertex_ts = np.zeros((2, 3, 2), dtype=float)
    clim_in = np.array([[-1.0, 1.0], [0.0, 2.0], [-2.0, 3.0]])
    clim = _normalize_video_clim(clim_in, n_frames=3, n_maps=2, vertex_ts=vertex_ts)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (3, 2, 2)
    assert np.allclose(clim[:, 0, :], clim_in)
    assert np.allclose(clim[:, 1, :], clim_in)


def test_normalize_video_clim_per_map_broadcasts_for_multi_frame():
    vertex_ts = np.zeros((2, 5, 3), dtype=float)
    clim_in = np.array([[-1.0, 1.0], [0.0, 2.0], [-2.0, 3.0]])
    clim = _normalize_video_clim(clim_in, n_frames=5, n_maps=3, vertex_ts=vertex_ts)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (5, 3, 2)
    assert np.allclose(clim[0, :, :], clim_in)
    assert np.allclose(clim[-1, :, :], clim_in)


def test_normalize_video_clim_per_frame_per_map_validates_shape():
    vertex_ts = np.zeros((2, 2, 3), dtype=float)
    clim_in = np.array(
        [
            [[-1.0, 1.0], [0.0, 2.0], [-2.0, 3.0]],
            [[-1.0, 1.0], [0.0, 2.0], [-2.0, 3.0]],
        ]
    )
    clim = _normalize_video_clim(clim_in, n_frames=2, n_maps=3, vertex_ts=vertex_ts)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (2, 3, 2)
    assert np.allclose(clim, clim_in)


def test_normalize_maps_clim_none_passthrough():
    assert _normalize_maps_clim(None, n_maps=2) is None


def test_normalize_maps_clim_fixed_pair():
    clim = _normalize_maps_clim((-2, 3), n_maps=4)
    assert clim == (-2.0, 3.0)


def test_normalize_maps_clim_per_map_array():
    clim_in = np.array([[-1, 1], [0, 2], [-2, 5]], dtype=float)
    clim = _normalize_maps_clim(clim_in, n_maps=3)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (3, 2)
    assert np.allclose(clim, clim_in)


def test_normalize_maps_clim_shape_mismatch_raises():
    clim_in = np.array([[-1, 1], [0, 2]], dtype=float)
    with pytest.raises(ValueError):
        _normalize_maps_clim(clim_in, n_maps=3)


def test_normalize_maps_clim_degenerate_row_expands():
    clim_in = np.array([[1.0, 1.0]], dtype=float)
    clim = _normalize_maps_clim(clim_in, n_maps=1)
    assert isinstance(clim, np.ndarray)
    assert clim.shape == (1, 2)
    assert clim[0, 0] < clim[0, 1]


def test_prepare_triangle_vectors_accepts_shape_and_allows_nan():
    grads = np.array([[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]], dtype=float)
    out = _prepare_triangle_vectors(grads, n_triangles=2)
    assert out is not None
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [1.0, 0.0, 0.0])
    assert np.isnan(out[1]).all()


def test_prepare_triangle_vectors_raises_on_bad_shape():
    with pytest.raises(ValueError):
        _prepare_triangle_vectors(np.zeros((3,)), n_triangles=1)
    with pytest.raises(ValueError):
        _prepare_triangle_vectors(np.zeros((2, 2)), n_triangles=2)
    with pytest.raises(ValueError):
        _prepare_triangle_vectors(np.zeros((3, 3)), n_triangles=2)


def test_auto_gradient_vector_max_length_uses_bbox_diagonal_fraction():
    class DummyMesh:
        # bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        bounds = (0.0, 2.0, -1.0, 1.0, 0.0, 0.0)

    max_len = _auto_gradient_vector_max_length(DummyMesh())
    diag = np.sqrt((2.0 - 0.0) ** 2 + (1.0 - (-1.0)) ** 2 + (0.0 - 0.0) ** 2)
    assert np.isfinite(max_len)
    assert np.allclose(max_len, 0.05 * diag)


def test_clip_vectors_to_max_length_scales_down_only():
    vecs = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    out = _clip_vectors_to_max_length(vecs, max_length=2.0)
    assert out.shape == vecs.shape
    assert np.allclose(out[0], [2.0, 0.0, 0.0])
    assert np.allclose(out[1], [0.0, 2.0, 0.0])
    assert np.allclose(out[2], [0.0, 0.0, 0.0])


def test_clip_vectors_to_max_length_raises_on_invalid_max_length():
    with pytest.raises(ValueError):
        _clip_vectors_to_max_length(np.zeros((1, 3)), max_length=0.0)
    with pytest.raises(ValueError):
        _clip_vectors_to_max_length(np.zeros((1, 3)), max_length=float("nan"))


def test_downsample_triangle_vectors_by_vertex_mask_selects_unique_cells():
    # Simple square split into two triangles
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    surf = (verts, faces)

    gradients = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    vertex_mask = np.array([True, False, True, False])

    ds, cell_ids = downsample_triangle_vectors_by_vertex_mask(gradients, surf, vertex_mask)
    assert ds.shape == gradients.shape
    assert cell_ids.ndim == 1
    assert np.all(cell_ids >= 0)
    assert np.all(cell_ids < gradients.shape[0])
    assert np.unique(cell_ids).size == cell_ids.size

    # Only selected rows are non-NaN
    keep = np.zeros(gradients.shape[0], dtype=bool)
    keep[cell_ids] = True
    assert np.all(np.isfinite(ds[keep]).all(axis=1))
    assert np.isnan(ds[~keep]).all()

    # Selected rows match the original values
    assert np.allclose(ds[keep], gradients[keep])


def test_downsample_triangle_vectors_by_vertex_mask_empty_mask_returns_all_nan():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    surf = (verts, faces)

    gradients = np.array([[1.0, 2.0, 3.0]], dtype=float)
    vertex_mask = np.array([False, False, False])

    ds, cell_ids = downsample_triangle_vectors_by_vertex_mask(gradients, surf, vertex_mask)
    assert ds.shape == (1, 3)
    assert np.isnan(ds).all()
    assert cell_ids.size == 0


def test_downsample_triangle_vectors_by_vertex_mask_raises_on_shape_mismatch():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    surf = (verts, faces)

    with pytest.raises(ValueError):
        downsample_triangle_vectors_by_vertex_mask(np.zeros((2, 3)), surf, np.array([True, False, False]))

    with pytest.raises(ValueError):
        downsample_triangle_vectors_by_vertex_mask(np.zeros((1, 3)), surf, np.array([True, False]))
