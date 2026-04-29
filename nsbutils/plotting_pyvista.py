"""PyVista-based surface plotting utilities.

This module provides `plot_surf_single` and `plot_surf` functions that mirror the
behavior of the Plotly-based API the project uses elsewhere, but renders using
PyVista/VTK.

PyVista is treated as an optional dependency and is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_DEFAULT_PANEL_SIZE: Tuple[int, int] = (400, 300)  # (width, height) in pixels

# Gradient-vector overlay: clip displayed vectors to a fixed fraction of the mesh
# bounding-box diagonal (proxy for overall brain size). This prevents a few
# extreme triangles from producing unreadably large arrows.
_GRADIENT_VECTOR_MAXLEN_FRAC_BBOX_DIAG: float = 0.02


def _enable_pyvista_off_screen() -> None:
    """Best-effort setup for robust off-screen rendering in headless environments."""

    try:
        pv.OFF_SCREEN = True
    except Exception:
        pass

    try:
        # TODO: `pv.start_xvfb` is deprecated. Install vtk with osmesa instead
        pv.start_xvfb(wait=0.05)
    except Exception:
        pass


def _validate_clim(clim: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if clim is None:
        return None
    try:
        arr = np.asarray(clim, dtype=float)
    except Exception as e:
        raise ValueError("`clim` must be a (vmin, vmax) pair.") from e
    if arr.ndim != 1 or arr.size != 2:
        raise ValueError("`clim` must be a (vmin, vmax) pair.")
    vmin = float(arr[0])
    vmax = float(arr[1])
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        raise ValueError("`clim` values must be finite.")
    if not (vmin < vmax):
        raise ValueError("`clim` must satisfy vmin < vmax.")
    return vmin, vmax


def _normalize_maps_clim(
    clim: Any,
    *,
    n_maps: int,
) -> Union[None, Tuple[float, float], np.ndarray]:
    """Normalize `clim` for `plot_surf`.

    Accepts:
    - None: pass through (PyVista chooses range per mesh)
    - (vmin, vmax): fixed limits for all maps
    - array-like of shape (n_maps, 2): per-map limits

    Returns:
    - None
    - (vmin, vmax)
    - ndarray of shape (n_maps, 2)
    """

    if clim is None:
        return None

    if n_maps <= 0:
        raise ValueError("`n_maps` must be > 0.")

    def _expand_if_degenerate(vmin: float, vmax: float) -> Tuple[float, float]:
        if vmin < vmax:
            return vmin, vmax
        if vmin != vmax:
            return vmin, vmax
        eps = max(1e-6, abs(vmin) * 1e-3)
        return vmin - eps, vmax + eps

    arr = np.asarray(clim, dtype=float)

    # Fixed (vmin, vmax)
    if arr.ndim == 1 and arr.size == 2:
        vmin, vmax = float(arr[0]), float(arr[1])
        vmin, vmax = _expand_if_degenerate(vmin, vmax)
        validated = _validate_clim((vmin, vmax))
        assert validated is not None
        return validated

    # Per-map (n_maps, 2)
    if arr.ndim == 2 and arr.shape == (n_maps, 2):
        out = np.empty((n_maps, 2), dtype=float)
        for idx in range(n_maps):
            vmin = float(arr[idx, 0])
            vmax = float(arr[idx, 1])

            vmin_finite = np.isfinite(vmin)
            vmax_finite = np.isfinite(vmax)
            if vmin_finite and vmax_finite:
                vmin, vmax = _expand_if_degenerate(vmin, vmax)
                validated = _validate_clim((vmin, vmax))
                assert validated is not None
                out[idx, 0], out[idx, 1] = validated
                continue

            if (not vmin_finite) and (not vmax_finite):
                out[idx, 0], out[idx, 1] = (-1.0, 1.0)
                continue

            raise ValueError(
                "Per-map `clim` must contain finite (vmin, vmax) pairs for each map; "
                "got a partially non-finite row."
            )
        return out

    raise ValueError("`clim` must be None, a (vmin, vmax) pair, or an array of shape (n_maps, 2).")


def _normalize_video_clim(
    clim: Any,
    *,
    n_frames: int,
    n_maps: int = 1,
    vertex_ts: np.ndarray,
) -> Union[Tuple[float, float], np.ndarray]:
    """Normalize `clim` for `plot_surf_video`.

    Accepts:
    - None: automatic global symmetric limits across all frames (current behavior)
    - (vmin, vmax): fixed limits across all frames (and maps)
    - array-like of shape (n_frames, 2): per-frame limits (broadcast across maps)
    - array-like of shape (n_maps, 2): per-map limits (broadcast across frames)
    - array-like of shape (n_frames, n_maps, 2): per-frame per-map limits

    Returns:
    - If n_maps == 1: either a single (vmin, vmax) tuple, or a float array of shape (n_frames, 2).
    - If n_maps > 1: a float array of shape (n_frames, n_maps, 2).
    """

    if n_frames <= 0:
        raise ValueError("`n_frames` must be > 0.")
    if n_maps <= 0:
        raise ValueError("`n_maps` must be > 0.")

    def _expand_if_degenerate(vmin: float, vmax: float) -> Tuple[float, float]:
        if vmin < vmax:
            return vmin, vmax
        if vmin != vmax:
            # Covers vmin > vmax; let _validate_clim raise with a clearer error.
            return vmin, vmax
        eps = max(1e-6, abs(vmin) * 1e-3)
        return vmin - eps, vmax + eps

    def _broadcast_fixed(validated: Tuple[float, float]) -> Union[Tuple[float, float], np.ndarray]:
        if n_maps == 1:
            return validated
        base = np.asarray(validated, dtype=float).reshape(1, 1, 2)
        return np.broadcast_to(base, (n_frames, n_maps, 2)).copy()

    if clim is None:
        abs_max = float(np.nanmax(np.abs(vertex_ts)))
        if not np.isfinite(abs_max) or abs_max == 0:
            clim_use: Tuple[float, float] = (-1.0, 1.0)
        else:
            clim_use = (-abs_max, abs_max)
        validated = _validate_clim(clim_use)
        assert validated is not None
        return _broadcast_fixed(validated)

    arr = np.asarray(clim, dtype=float)

    # Fixed (vmin, vmax)
    if arr.ndim == 1 and arr.size == 2:
        vmin = float(arr[0])
        vmax = float(arr[1])
        vmin, vmax = _expand_if_degenerate(vmin, vmax)
        validated = _validate_clim((vmin, vmax))
        assert validated is not None
        return _broadcast_fixed(validated)

    def _validate_pairs(pairs_2d: np.ndarray, *, error_prefix: str) -> np.ndarray:
        out = np.empty_like(pairs_2d, dtype=float)
        for idx in range(pairs_2d.shape[0]):
            vmin = float(pairs_2d[idx, 0])
            vmax = float(pairs_2d[idx, 1])

            vmin_finite = np.isfinite(vmin)
            vmax_finite = np.isfinite(vmax)
            if vmin_finite and vmax_finite:
                vmin, vmax = _expand_if_degenerate(vmin, vmax)
                validated = _validate_clim((vmin, vmax))
                assert validated is not None
                out[idx, 0], out[idx, 1] = validated
                continue

            if (not vmin_finite) and (not vmax_finite):
                out[idx, 0], out[idx, 1] = (-1.0, 1.0)
                continue

            raise ValueError(
                f"{error_prefix} must contain finite (vmin, vmax) pairs; got a partially non-finite row."
            )
        return out

    # Per-frame (n_frames, 2)
    if arr.ndim == 2 and arr.shape == (n_frames, 2):
        out = _validate_pairs(arr, error_prefix="Per-frame `clim`")
        if n_maps == 1:
            return out
        return np.broadcast_to(out[:, np.newaxis, :], (n_frames, n_maps, 2)).copy()

    # Per-map (n_maps, 2)
    if arr.ndim == 2 and arr.shape == (n_maps, 2):
        out_maps = _validate_pairs(arr, error_prefix="Per-map `clim`")
        if n_maps == 1:
            validated = _validate_clim((float(out_maps[0, 0]), float(out_maps[0, 1])))
            assert validated is not None
            return validated
        return np.broadcast_to(out_maps[np.newaxis, :, :], (n_frames, n_maps, 2)).copy()

    # Per-frame per-map (n_frames, n_maps, 2)
    if arr.ndim == 3 and arr.shape == (n_frames, n_maps, 2):
        out3 = np.empty((n_frames, n_maps, 2), dtype=float)
        for f_idx in range(n_frames):
            out3[f_idx, :, :] = _validate_pairs(arr[f_idx, :, :], error_prefix="Per-frame per-map `clim`")
        return out3

    raise ValueError(
        "`clim` must be None, a (vmin, vmax) pair, or an array of shape (n_frames, 2), (n_maps, 2), "
        "or (n_frames, n_maps, 2)."
    )


@dataclass(frozen=True)
class _CameraView:
    direction: Tuple[float, float, float]
    viewup: Tuple[float, float, float]


# Coordinate convention matches the Plotly version you provided:
# - lateral/medial look along +/- x with +z as up
# - dorsal/ventral look along +/- z with +y as up
# - anterior/posterior look along +/- y with +z as up
camera_views: Dict[str, _CameraView] = {
    "lateral": _CameraView(direction=(-1.0, 0.0, 0.0), viewup=(0.0, 0.0, 1.0)),
    "medial": _CameraView(direction=(1.0, 0.0, 0.0), viewup=(0.0, 0.0, 1.0)),
    "dorsal": _CameraView(direction=(0.0, 0.0, 1.0), viewup=(0.0, 1.0, 0.0)),
    "ventral": _CameraView(direction=(0.0, 0.0, -1.0), viewup=(0.0, 1.0, 0.0)),
    "anterior": _CameraView(direction=(0.0, 1.0, 0.0), viewup=(0.0, 0.0, 1.0)),
    "posterior": _CameraView(direction=(0.0, -1.0, 0.0), viewup=(0.0, 0.0, 1.0)),
}


def _apply_camera_headlight(plotter: Any, intensity: float = 1.0) -> None:
    """Configure a camera-aligned light ("headlight") on the active renderer.

    This makes lighting appear to come straight out of the screen.
    """

    try:
        renderer = plotter.renderer
    except Exception:
        return

    try:
        renderer.remove_all_lights()
        light = pv.Light(light_type="headlight")
        light.intensity = float(intensity)
        renderer.add_light(light)
    except Exception:
        # If lighting is unavailable (backend-specific), fail silently.
        return


def _draw_image_on_axis(ax: Axes, image: np.ndarray) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _as_pathlike(obj: Any) -> Optional[Path]:
    if isinstance(obj, Path):
        return obj
    if isinstance(obj, str):
        return Path(obj)
    return None


def _try_read_surf_nsbutils(surf: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Try to load a surface via nsbutils.io.read_surf.

    Returns (verts, faces) or None if nsbutils is unavailable or fails.
    """

    try:
        from nsbutils.io import read_surf  # type: ignore
    except Exception:
        return None

    try:
        mesh = read_surf(surf)
        verts = np.asarray(mesh.v)
        faces = np.asarray(mesh.t)
        return verts, faces
    except Exception:
        return None


def _polydata_from_verts_faces(pv: Any, verts: np.ndarray, faces: np.ndarray) -> Any:
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected verts shape (n_verts, 3), got {verts.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shape (n_faces, 3), got {faces.shape}.")

    faces_i64 = faces.astype(np.int64, copy=False)
    # PyVista expects a flattened face array with a leading vertex count per face.
    faces_pv = np.hstack([np.full((faces_i64.shape[0], 1), 3, dtype=np.int64), faces_i64]).ravel()
    return pv.PolyData(verts, faces_pv)


def _load_surface(surf: Any) -> Tuple[Any, int]:
    """Load a surface into a PyVista PolyData and return (polydata, n_verts)."""

    # In-memory `(verts, faces)`
    if isinstance(surf, (tuple, list)) and len(surf) == 2:
        poly = _polydata_from_verts_faces(pv, surf[0], surf[1])
        return poly, poly.n_points

    # Dict-like `{v, t}` (neuromodes-style)
    if isinstance(surf, Mapping) and "v" in surf and "t" in surf:
        poly = _polydata_from_verts_faces(pv, np.asarray(surf["v"]), np.asarray(surf["t"]))
        return poly, poly.n_points

    # Dict-like `{vertices, faces}` (common nsbutils/tutorial style)
    if isinstance(surf, Mapping) and "vertices" in surf and "faces" in surf:
        poly = _polydata_from_verts_faces(pv, np.asarray(surf["vertices"]), np.asarray(surf["faces"]))
        return poly, poly.n_points

    # Try neuromodes first for known gifti-like extensions
    path = _as_pathlike(surf)
    if path is not None and path.suffix.lower() == ".gii":
        nm = _try_read_surf_nsbutils(surf)
        if nm is not None:
            verts, faces = nm
            poly = _polydata_from_verts_faces(pv, verts, faces)
            return poly, poly.n_points

    # File paths: try PyVista reader
    if path is not None:
        try:
            poly = pv.read(str(path))
            # Ensure PolyData (some readers return UnstructuredGrid)
            poly = poly.extract_surface().triangulate()
            return poly, poly.n_points
        except Exception:
            nm = _try_read_surf_nsbutils(surf)
            if nm is not None:
                verts, faces = nm
                poly = _polydata_from_verts_faces(pv, verts, faces)
                return poly, poly.n_points
            raise

    # Fallback: try neuromodes for non-path objects (GiftiImage, lapy mesh, etc.)
    nm = _try_read_surf_nsbutils(surf)
    if nm is not None:
        verts, faces = nm
        poly = _polydata_from_verts_faces(pv, verts, faces)
        return poly, poly.n_points

    raise ValueError(
        "Unsupported `surf` input. Provide a file path, a (verts, faces) tuple, a dict with keys "
        "{'v','t'} or {'vertices','faces'}, or an object supported by neuromodes.io.read_surf (if installed)."
    )


def _prepare_vertex_scalars(
    data: Optional[np.ndarray],
    rois: Optional[np.ndarray],
    n_verts: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate and normalize scalars.

    Returns (vertex_data, roi_labels). If rois is provided, masked vertices (roi==0)
    are set to NaN in vertex_data.
    """

    roi_labels = None
    if rois is not None:
        roi_labels = np.asarray(rois)
        if roi_labels.shape != (n_verts,):
            raise ValueError(f"ROIs shape {roi_labels.shape} does not match mesh of shape (n_verts,) = ({n_verts},).")

    if data is None:
        return None, roi_labels

    data_arr = np.asarray(data)

    # Vertex-wise data
    if data_arr.shape == (n_verts,):
        vertex_data = data_arr.astype(float, copy=False)
    else:
        # ROI-wise data: if rois provided and data matches number of ROIs
        if roi_labels is None:
            raise ValueError(
                f"Data shape {data_arr.shape} does not match mesh of shape (n_verts,) = ({n_verts},)."
            )
        n_rois = int(np.nanmax(roi_labels))
        if data_arr.shape == (n_rois,):
            vertex_data = np.zeros(n_verts, dtype=float)
            for roi_id in range(1, n_rois + 1):
                vertex_data[roi_labels == roi_id] = float(data_arr[roi_id - 1])
        else:
            raise ValueError(
                f"Data shape {data_arr.shape} does not match mesh vertices ({n_verts},) or number of ROIs ({n_rois},)."
            )

    if roi_labels is not None:
        vertex_data = vertex_data.astype(float, copy=False)
        vertex_data = vertex_data.copy()
        vertex_data[roi_labels == 0] = np.nan

    return vertex_data, roi_labels


def _prepare_triangle_vectors(
    gradients: Optional[np.ndarray],
    *,
    n_triangles: int,
) -> Optional[np.ndarray]:
    """Validate and normalize triangle-wise vectors.

    Parameters
    ----------
    gradients
        Triangle-wise vectors of shape ``(n_triangles, 3)``.
        Rows may contain NaNs; these will be skipped during rendering.
    n_triangles
        Expected number of triangles/cells in the mesh.

    Returns
    -------
    np.ndarray | None
        Array of shape ``(n_triangles, 3)`` (float) or None.
    """

    if gradients is None:
        return None

    arr = np.asarray(gradients, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"`gradients` must have shape (n_triangles, 3); got {arr.shape}.")
    if arr.shape[0] != int(n_triangles):
        raise ValueError(
            f"`gradients` first dimension must match n_triangles={n_triangles}; got {arr.shape[0]}."
        )
    return arr


def _auto_gradient_vector_max_length(mesh: Any) -> float:
    """Compute an automatic cap for displayed gradient-vector lengths.

    Uses a fixed fraction of the mesh axis-aligned bounding-box diagonal.
    If the diagonal is degenerate or non-finite, returns np.inf (no clipping).
    """

    try:
        bounds = getattr(mesh, "bounds")
        b = np.asarray(bounds, dtype=float).reshape(-1)
        if b.size != 6:
            raise ValueError
        dx = float(b[1] - b[0])
        dy = float(b[3] - b[2])
        dz = float(b[5] - b[4])
    except Exception:
        # Fallback: compute bounds from points if available.
        try:
            pts = np.asarray(getattr(mesh, "points"), dtype=float)
        except Exception:
            return float("inf")
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return float("inf")
        mins = np.nanmin(pts, axis=0)
        maxs = np.nanmax(pts, axis=0)
        dx = float(maxs[0] - mins[0])
        dy = float(maxs[1] - mins[1])
        dz = float(maxs[2] - mins[2])

    diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if (not np.isfinite(diag)) or diag <= 0:
        return float("inf")

    frac = float(_GRADIENT_VECTOR_MAXLEN_FRAC_BBOX_DIAG)
    if (not np.isfinite(frac)) or frac <= 0:
        return float("inf")

    return frac * diag


def _clip_vectors_to_max_length(vecs: np.ndarray, *, max_length: float) -> np.ndarray:
    """Row-wise clip so that each vector has norm <= max_length."""

    if not np.isfinite(float(max_length)):
        raise ValueError("`max_length` must be finite.")
    if float(max_length) <= 0:
        raise ValueError("`max_length` must be > 0.")

    arr = np.asarray(vecs, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected vecs shape (n, 3), got {arr.shape}.")

    norms = np.linalg.norm(arr, axis=1)
    out = arr.copy()
    nonzero = norms > 0
    scale = np.ones_like(norms)
    scale[nonzero] = np.minimum(1.0, float(max_length) / norms[nonzero])
    out *= scale[:, np.newaxis]
    return out


def downsample_triangle_vectors_by_vertex_mask(
    gradients: np.ndarray,
    surf: Any,
    vertex_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample triangle vectors using vertex sample locations.

    This helper is intentionally kept outside of `plot_surf`.

    Strategy (Option A)
    -------------------
    - Treat `vertex_mask` as a set of sample locations on the surface.
    - For each selected vertex, pick the closest triangle (cell) via
      `mesh.find_closest_cell(mesh.points[vertex_mask])`.
    - Deduplicate the resulting cell ids.
    - Return a NaN-masked triangle-vector array of the same shape as the input.

    Parameters
    ----------
    gradients
        Triangle-wise vectors of shape ``(n_triangles, 3)``.
    surf
        Surface mesh in any of the formats accepted by `plot_surf`/`plot_surf_single`
        (e.g., ``(verts, faces)`` tuple, dict with ``{"v","t"}`` or
        ``{"vertices","faces"}``, file path, etc). You may also pass a
        `pyvista.PolyData` directly.
    vertex_mask
        Boolean array of shape ``(n_vertices,)`` selecting sample vertices.

    Returns
    -------
    gradients_ds, cell_ids
        `gradients_ds` has shape ``(n_triangles, 3)`` with non-selected rows set to NaN.
        `cell_ids` are the unique selected triangle ids.
    """

    if surf is None:
        raise ValueError("`surf` must be provided.")

    # Accept a pre-loaded PyVista mesh-like object, otherwise load from `surf`.
    if hasattr(surf, "n_cells") and hasattr(surf, "n_points") and hasattr(surf, "points"):
        mesh = surf
    else:
        mesh, _ = _load_surface(surf)

    grads = np.asarray(gradients, dtype=float)
    if grads.ndim != 2 or grads.shape[1] != 3:
        raise ValueError(f"`gradients` must have shape (n_triangles, 3); got {grads.shape}.")
    n_triangles = int(getattr(mesh, "n_cells", -1))
    if n_triangles <= 0:
        raise ValueError("Loaded mesh must have a positive number of cells.")
    if grads.shape[0] != n_triangles:
        raise ValueError(
            f"`gradients` first dimension must match mesh.n_cells={n_triangles}; got {grads.shape[0]}."
        )

    vmask = np.asarray(vertex_mask, dtype=bool)
    n_points = int(getattr(mesh, "n_points", -1))
    if n_points <= 0:
        raise ValueError("Loaded mesh must have a positive number of points.")
    if vmask.shape != (n_points,):
        raise ValueError(f"`vertex_mask` must have shape (mesh.n_points,) = ({n_points},); got {vmask.shape}.")

    if not np.any(vmask):
        out = np.full_like(grads, np.nan, dtype=float)
        return out, np.asarray([], dtype=np.int64)

    sample_points = np.asarray(mesh.points)[vmask]
    try:
        cell_ids = mesh.find_closest_cell(sample_points)
    except Exception as e:
        raise RuntimeError("Failed to query closest cells from mesh; ensure `mesh` is a valid PyVista PolyData.") from e

    cell_ids_arr = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    cell_ids_arr = cell_ids_arr[cell_ids_arr >= 0]
    cell_ids_arr = np.unique(cell_ids_arr)

    out = np.full_like(grads, np.nan, dtype=float)
    if cell_ids_arr.size:
        out[cell_ids_arr] = grads[cell_ids_arr]
    return out, cell_ids_arr


def _prepare_timeseries_scalars(
    data_timeseries: np.ndarray,
    rois: Optional[np.ndarray],
    n_verts: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate and normalize timeseries scalars.

    Accepts either vertex-wise timeseries (n_verts, n_frames) / (n_verts, n_frames, n_maps)
    or ROI-wise timeseries (n_rois, n_frames) / (n_rois, n_frames, n_maps) when `rois` is provided.

    Returns (vertex_timeseries, roi_labels). If roi_labels is provided, masked
    vertices (roi==0) are set to NaN across all frames.
    """

    roi_labels: Optional[np.ndarray] = None
    if rois is not None:
        roi_labels = np.asarray(rois)
        if roi_labels.shape != (n_verts,):
            raise ValueError(
                f"ROIs shape {roi_labels.shape} does not match mesh of shape (n_verts,) = ({n_verts},)."
            )

    arr = np.asarray(data_timeseries)
    if arr.ndim not in (2, 3):
        raise ValueError(
            "`data_timeseries` must be a 2D array (n_verts, n_frames)/(n_rois, n_frames) or a 3D array "
            "(n_verts, n_frames, n_maps)/(n_rois, n_frames, n_maps)."
        )

    if arr.shape[0] == n_verts:
        vertex_ts = arr.astype(float, copy=False)
    else:
        if roi_labels is None:
            raise ValueError(
                f"`data_timeseries` shape {arr.shape} does not match mesh vertices ({n_verts},)."
            )
        n_rois = int(np.nanmax(roi_labels))
        if arr.shape[0] != n_rois:
            raise ValueError(
                f"`data_timeseries` shape {arr.shape} does not match number of ROIs ({n_rois}, n_frames)."
            )

        n_frames = arr.shape[1]
        n_maps = 1 if arr.ndim == 2 else arr.shape[2]
        out_shape = (n_verts, n_frames) if arr.ndim == 2 else (n_verts, n_frames, n_maps)
        vertex_ts = np.zeros(out_shape, dtype=float)
        for roi_id in range(1, n_rois + 1):
            vertex_ts[roi_labels == roi_id, ...] = arr[roi_id - 1, ...]

    if roi_labels is not None:
        vertex_ts = vertex_ts.copy()
        vertex_ts[roi_labels == 0, ...] = np.nan

    return vertex_ts, roi_labels


def _rh_view_swap(view: str) -> str:
    rh_view_swap = {
        "lateral": "medial",
        "medial": "lateral",
        "dorsal": "ventral",
        "ventral": "dorsal",
        "anterior": "posterior",
        "posterior": "anterior",
    }
    return rh_view_swap.get(view, view)


def _default_scalar_bar_args() -> Dict[str, Any]:
    # Compact defaults; caller can override by passing `scalar_bar_args`.
    return {
        # Vertical bar to the right of each subplot
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.10,
        "width": 0.12,
        "height": 0.80,
        "n_labels": 3,
        "fmt": "%.2g",
    }


def _set_camera_named_view(plotter: Any, view: str) -> None:
    if view not in camera_views:
        raise ValueError(f"Invalid view '{view}'. Valid options are: {list(camera_views.keys())}.")

    cam = camera_views[view]
    # view_vector expects the direction the camera points *from* towards the focal point.
    # Using `view_vector(direction)` matches the Plotly convention for named views.
    plotter.view_vector(cam.direction, viewup=cam.viewup)


def _finalize_camera(plotter: Any, view: str, zoom: float) -> None:
    # Orientation -> fit -> orientation (robust) -> parallel projection -> optional zoom
    _set_camera_named_view(plotter, view)
    plotter.reset_camera()
    _set_camera_named_view(plotter, view)
    plotter.enable_parallel_projection()
    if zoom != 1.0:
        if zoom <= 0:
            raise ValueError("`zoom` must be > 0.")
        plotter.camera.zoom(float(zoom))


def _polydata_from_nan_separated_segments(pv: Any, xe: np.ndarray, ye: np.ndarray, ze: np.ndarray) -> Any:
    xe = np.asarray(xe)
    ye = np.asarray(ye)
    ze = np.asarray(ze)
    if xe.size == 0:
        return pv.PolyData()

    points: List[List[float]] = []
    lines: List[int] = []
    line_idx = 0

    # Expected format: x0, x1, nan repeating (same for y/z)
    for idx in range(0, len(xe), 3):
        if idx + 1 >= len(xe):
            break
        x0, x1 = xe[idx], xe[idx + 1]
        y0, y1 = ye[idx], ye[idx + 1]
        z0, z1 = ze[idx], ze[idx + 1]
        if np.isnan(x0) or np.isnan(x1) or np.isnan(y0) or np.isnan(y1) or np.isnan(z0) or np.isnan(z1):
            continue

        points.append([float(x0), float(y0), float(z0)])
        points.append([float(x1), float(y1), float(z1)])
        lines.extend([2, line_idx, line_idx + 1])
        line_idx += 2

    if not points:
        return pv.PolyData()

    # Creating PolyData with only points may implicitly create vertex cells,
    # which can render as visible dots at segment endpoints when zooming.
    # Ensure this polydata contains only line cells.
    poly = pv.PolyData(np.asarray(points))
    poly.lines = np.asarray(lines, dtype=np.int64)
    try:
        poly.verts = np.empty((0,), dtype=np.int64)
    except Exception:
        pass
    return poly


def _add_surface_to_plotter(
    plotter: Any,
    surf: Any,
    data: Optional[np.ndarray],
    rois: Optional[np.ndarray],
    gradients: Optional[np.ndarray],
    *,
    cbar: bool,
    cmap: Union[str, Any],
    mesh_edges: bool,
    roi_outlines: bool,
    gradient_scale: float,
    scalar_bar_args: Optional[Dict[str, Any]],
    clim: Optional[Tuple[float, float]],
) -> Tuple[Any, Optional[Any], Optional[np.ndarray]]:
    """Add a surface mesh to a plotter and return (mesh_used, actor, roi_labels)."""

    _apply_camera_headlight(plotter)

    mesh, n_verts = _load_surface(surf)
    vertex_data, roi_labels = _prepare_vertex_scalars(data, rois, n_verts)
    triangle_vectors = _prepare_triangle_vectors(gradients, n_triangles=int(mesh.n_cells))
    if not np.isfinite(float(gradient_scale)):
        raise ValueError("`gradient_scale` must be finite.")
    if float(gradient_scale) < 0:
        raise ValueError("`gradient_scale` must be >= 0.")
    validated_clim = _validate_clim(clim)

    show_scalar_bar = bool(cbar and vertex_data is not None)
    sb_args = {**_default_scalar_bar_args(), **(scalar_bar_args or {})} if show_scalar_bar else {}

    actor = None
    mesh_used = mesh
    if vertex_data is None:
        plotter.add_mesh(
            mesh,
            color="lightgrey",
            smooth_shading=True,
            show_edges=mesh_edges,
            edge_color="black",
            line_width=0.5,
            ambient=0.01,
            diffuse=1,
            specular=0.1,
            roughness=1e-6,
        )
    else:
        mesh_used = mesh.copy(deep=False)
        mesh_used.point_data["scalars"] = vertex_data
        actor = plotter.add_mesh(
            mesh_used,
            scalars="scalars",
            cmap=cmap,
            nan_color="lightgrey",
            smooth_shading=True,
            show_edges=mesh_edges,
            edge_color="black",
            line_width=0.5,
            show_scalar_bar=False,
            clim=validated_clim,
            ambient=0.01,
            diffuse=1,
            specular=0.1,
            roughness=1e-6,
        )

        if show_scalar_bar:
            user_title = None
            if "title" in sb_args:
                user_title = sb_args.get("title")
                sb_args = {k: v for k, v in sb_args.items() if k != "title"}

            renderer_idx = getattr(getattr(plotter, "renderers", None), "active_index", 0)
            internal_title = f"cbar-{renderer_idx}"
            scalar_bar = plotter.add_scalar_bar(title=internal_title, mapper=actor.mapper, **sb_args)
            if user_title is None:
                scalar_bar.SetTitle("")
            else:
                scalar_bar.SetTitle(str(user_title))

    if roi_outlines and roi_labels is not None:
        xe, ye, ze = compute_roi_midline_edges(mesh_used.points, mesh_used.faces.reshape(-1, 4)[:, 1:], roi_labels)
        outline_poly = _polydata_from_nan_separated_segments(pv, xe, ye, ze)
        if outline_poly.n_points:
            plotter.add_mesh(outline_poly, color="black", line_width=1.7)

    if triangle_vectors is not None:
        centers = np.asarray(mesh_used.cell_centers().points)
        vecs = triangle_vectors
        finite = np.isfinite(centers).all(axis=1) & np.isfinite(vecs).all(axis=1)
        if np.any(finite):
            csel = centers[finite]
            vsel = vecs[finite]
            vec_disp = float(gradient_scale) * vsel
            max_len = _auto_gradient_vector_max_length(mesh_used)
            if np.isfinite(float(max_len)):
                vec_disp = _clip_vectors_to_max_length(vec_disp, max_length=float(max_len))

            endpoints = csel + vec_disp

            # Build nan-separated segments arrays: [x0, x1, nan] repeated
            xe = np.empty(csel.shape[0] * 3, dtype=float)
            ye = np.empty_like(xe)
            ze = np.empty_like(xe)
            xe[0::3], xe[1::3], xe[2::3] = csel[:, 0], endpoints[:, 0], np.nan
            ye[0::3], ye[1::3], ye[2::3] = csel[:, 1], endpoints[:, 1], np.nan
            ze[0::3], ze[1::3], ze[2::3] = csel[:, 2], endpoints[:, 2], np.nan

            vec_poly = _polydata_from_nan_separated_segments(pv, xe, ye, ze)
            if vec_poly.n_points:
                plotter.add_mesh(vec_poly, color="black", line_width=2.0)

    return mesh_used, actor, roi_labels


def plot_surf_single(
    surf: Any,
    data: Optional[np.ndarray] = None,
    rois: Optional[np.ndarray] = None,
    plotter: Optional[Any] = None,
    subplot: Optional[Tuple[int, int]] = None,
    view: str = "lateral",
    zoom: float = 1.0,
    size: Optional[Tuple[int, int]] = None,
    cbar: bool = False,
    cmap: Union[str, Any] = "turbo",
    mesh_edges: bool = False,
    roi_outlines: bool = False,
    gradients: Optional[np.ndarray] = None,
    *,
    ax: Optional[Axes] = None,
    scale: float = 1.0,
    clim: Optional[Union[Tuple[float, float], np.ndarray]] = None,
    gradient_scale: float = 1.0,
    scalar_bar_args: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Render a single surface into a PyVista subplot or embed into a Matplotlib axis.

    - If `ax` is provided: renders off-screen and draws an image into `ax`, returning None.
    - Else: draws into `plotter` (or creates one) and returns the plotter.

    Parameters are intentionally aligned with the Plotly version, except `plot_surf_single`
    does not take Plotly-specific `fig/row/col`.

    Notes
    -----
    When `gradients` is provided, the displayed gradient vectors are automatically
    clipped to a fixed fraction (currently 5%) of the surface mesh bounding-box
    diagonal, to prevent extreme outliers from producing unreadably large arrows.
    Use `gradient_scale` to tune overall vector visibility.
    """

    if ax is not None:
        _enable_pyvista_off_screen()
        panel_size = size or _DEFAULT_PANEL_SIZE
        if scale <= 0:
            raise ValueError("`scale` must be > 0.")
        window_size = (int(panel_size[0] * scale), int(panel_size[1] * scale))
        p = pv.Plotter(off_screen=True, window_size=window_size)
        try:
            plot_surf_single(
                surf=surf,
                data=data,
                rois=rois,
                plotter=p,
                subplot=None,
                view=view,
                zoom=zoom,
                size=panel_size,
                cbar=cbar,
                cmap=cmap,
                mesh_edges=mesh_edges,
                roi_outlines=roi_outlines,
                gradients=gradients,
                clim=clim,
                gradient_scale=gradient_scale,
                scalar_bar_args=scalar_bar_args,
            )
            image = p.screenshot(return_img=True, transparent_background=False)
            _draw_image_on_axis(ax, image)
        finally:
            p.close()
        return None

    created_plotter = False
    if plotter is None:
        panel_size = size or _DEFAULT_PANEL_SIZE
        plotter = pv.Plotter(window_size=panel_size)
        created_plotter = True

    if subplot is not None:
        plotter.subplot(subplot[0], subplot[1])

    _add_surface_to_plotter(
        plotter,
        surf=surf,
        data=data,
        rois=rois,
        gradients=gradients,
        cbar=cbar,
        cmap=cmap,
        mesh_edges=mesh_edges,
        roi_outlines=roi_outlines,
        gradient_scale=gradient_scale,
        scalar_bar_args=scalar_bar_args,
        clim=clim,
    )

    plotter.hide_axes()
    _finalize_camera(plotter, view=view, zoom=zoom)

    return plotter if (created_plotter or plotter is not None) else None


def plot_surf(
    surf: Mapping[str, Any],
    data: Optional[Mapping[str, np.ndarray]] = None,
    rois: Optional[Mapping[str, np.ndarray]] = None,
    gradients: Optional[Mapping[str, np.ndarray]] = None,
    views: List[str] = ["lateral"],
    layout_indiv: str = "row",
    layout_group: str = "row",
    zoom: float = 1.0,
    size: Tuple[int, int] = _DEFAULT_PANEL_SIZE,
    cbar: bool = False,
    cmap: Union[str, Any] = "turbo",
    mesh_edges: bool = False,
    roi_outlines: bool = False,
    *,
    ax: Optional[Axes] = None,
    off_screen: bool = False,
    scale: float = 1.0,
    clim: Optional[Union[Tuple[float, float], np.ndarray]] = None,
    gradient_scale: float = 1.0,
    scalar_bar_args: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Plot surface data across hemispheres, views, and (optionally) multiple maps.

    Parameters
    ----------
    clim
        Color limits. Accepted forms:

        - None: let PyVista choose limits per mesh (default)
        - (vmin, vmax): fixed limits applied to every map
        - array-like of shape (n_maps, 2): per-map limits, one (vmin, vmax) pair per map
    off_screen
        If True, force off-screen rendering for PyVista.

    Notes
    -----
    When `gradients` is provided, the displayed gradient vectors are automatically
    clipped to a fixed fraction (currently 5%) of the surface mesh bounding-box
    diagonal. Use `gradient_scale` to tune overall vector visibility.
    """

    hemis = list(surf.keys())
    n_hemi = len(hemis)
    n_views = len(views)

    # Determine number of maps and ensure hemi data is 2D (n_verts, n_maps)
    if data is None:
        n_maps = 1
    else:
        data2: Dict[str, np.ndarray] = {}
        for hemi in hemis:
            if hemi not in data:
                raise ValueError(f"Missing data for hemisphere '{hemi}'.")
            arr = np.asarray(data[hemi])
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            data2[hemi] = arr
        n_maps = data2[hemis[0]].shape[1]
        for hemi in hemis:
            if data2[hemi].shape[1] != n_maps:
                raise ValueError("All hemispheres must have the same number of maps.")
        data = data2

    if gradients is not None and n_maps != 1:
        raise ValueError(
            "Gradient overlay currently supports only a single map (n_maps == 1). "
            "Provide single-map `data` or call `plot_surf` once per map."
        )
    if gradients is not None:
        for hemi in hemis:
            if hemi not in gradients:
                raise ValueError(f"Missing gradients for hemisphere '{hemi}'.")

    clim_norm = _normalize_maps_clim(clim, n_maps=n_maps)
    clim_fixed: Optional[Tuple[float, float]] = None
    clim_per_map: Optional[np.ndarray] = None
    if isinstance(clim_norm, tuple):
        clim_fixed = clim_norm
    elif isinstance(clim_norm, np.ndarray):
        clim_per_map = clim_norm

    # Individual block size (hemis × views) for a single map
    if layout_indiv == "row":
        indiv_rows, indiv_cols = 1, n_hemi * n_views
    elif layout_indiv == "col":
        indiv_rows, indiv_cols = n_hemi * n_views, 1
    elif layout_indiv == "grid":
        indiv_rows, indiv_cols = n_views, n_hemi
    else:
        raise ValueError("`layout_indiv` must be one of 'row', 'col', or 'grid'.")

    # Group layout: tile the individual block across maps
    if layout_group == "row":
        rows, cols = indiv_rows, indiv_cols * n_maps
    elif layout_group == "col":
        rows, cols = indiv_rows * n_maps, indiv_cols
    else:
        raise ValueError("`layout_group` must be one of 'row' or 'col'.")

    panel_w, panel_h = size
    window_size = (int(panel_w * cols), int(panel_h * rows))

    off_screen_use = (ax is not None) or bool(off_screen)
    if off_screen_use:
        _enable_pyvista_off_screen()

    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=window_size,
        off_screen=off_screen_use,
        border=False,
    )

    mirror_rh = ("lh" in hemis and "rh" in hemis)
    reverse_rh_view_order = mirror_rh and (layout_indiv in ("row", "col"))

    for map_idx in range(n_maps):
        clim_use = (
            clim_fixed
            if clim_fixed is not None
            else (None if clim_per_map is None else (float(clim_per_map[map_idx, 0]), float(clim_per_map[map_idx, 1])))
        )

        if layout_group == "row":
            row_offset, col_offset = 0, map_idx * indiv_cols
        else:
            row_offset, col_offset = map_idx * indiv_rows, 0

        for h_idx, hemi in enumerate(hemis):
            hemi_views = list(views)[::-1] if (hemi == "rh" and reverse_rh_view_order) else list(views)

            for v_idx, view in enumerate(hemi_views):
                camera_view = _rh_view_swap(view) if hemi == "rh" else view

                if layout_indiv == "row":
                    r0, c0 = 0, h_idx * n_views + v_idx
                elif layout_indiv == "col":
                    r0, c0 = h_idx * n_views + v_idx, 0
                else:  # grid
                    r0, c0 = v_idx, h_idx

                r, c = r0 + row_offset, c0 + col_offset

                plot_surf_single(
                    surf=surf[hemi],
                    data=None if data is None else data[hemi][:, map_idx],
                    rois=None if rois is None else rois[hemi],
                    plotter=plotter,
                    subplot=(r, c),
                    view=camera_view,
                    zoom=zoom,
                    size=size,
                    cbar=cbar,
                    cmap=cmap,
                    mesh_edges=mesh_edges,
                    roi_outlines=roi_outlines,
                    gradients=None if gradients is None else gradients[hemi],
                    clim=clim_use,
                    gradient_scale=gradient_scale,
                    scalar_bar_args=scalar_bar_args,
                )

    if ax is not None:
        try:
            image = plotter.screenshot(return_img=True, transparent_background=False, scale=scale)
            _draw_image_on_axis(ax, image)
        finally:
            plotter.close()
        return None

    return plotter


def plot_surf_video(
    surf: Any,
    data_timeseries: Union[np.ndarray, Mapping[str, np.ndarray]],
    *,
    rois: Optional[Union[np.ndarray, Mapping[str, np.ndarray]]] = None,
    filename: Union[str, Path] = "brain_animation.mp4",
    framerate: int = 10,
    view: Union[str, Sequence[str]] = "lateral",
    views: Optional[List[str]] = None,
    layout_indiv: str = "row",
    layout_group: str = "row",
    zoom: float = 1.0,
    size: Tuple[int, int] = (800, 608),
    cmap: Union[str, Any] = "plasma",
    mesh_edges: bool = False,
    roi_outlines: bool = False,
    cbar: bool = False,
    clim: Optional[Union[Tuple[float, float], np.ndarray]] = None,
    title_template: Optional[str] = "Time: {:.1f} ms",
    scalar_bar_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Create an MP4 video of surface activity over time using PyVista.

    Notes
    -----
    MP4 output via `pyvista.Plotter.open_movie` typically requires `ffmpeg`.

    Parameters
    ----------
    surf
        Either a single surface input accepted by `plot_surf_single`, or a hemisphere mapping
        (e.g., ``{"lh": <surf>, "rh": <surf>}``) like `plot_surf`.
    data_timeseries
        For a single surface: a 2D array ``(n_verts, n_frames)`` or a 3D array
        ``(n_verts, n_frames, n_maps)``.

        If `rois` is provided, ROI-wise variants ``(n_rois, n_frames)`` and
        ``(n_rois, n_frames, n_maps)`` are also accepted.

        For hemisphere mappings: provide a dict with matching hemisphere keys.
    rois
        Optional ROI/medial-wall labels. For a single surface: array of shape ``(n_verts,)``.
        For hemisphere mappings: dict of arrays keyed by hemisphere. Vertices with label 0 are masked.
    filename
        Output video filename.
    framerate
        Frames per second.
    clim
        Color limits. Accepted forms:

        - None: uses symmetric global limits across all frames (and maps) (default)
        - (vmin, vmax): fixed limits across all frames (and maps)
        - array-like of shape (n_frames, 2): per-frame limits (broadcast across maps)
        - array-like of shape (n_maps, 2): per-map limits (broadcast across frames)
        - array-like of shape (n_frames, n_maps, 2): per-frame per-map limits

        When using time-varying limits, the color scale (and scalar bars, if enabled) will change
        over time.
    title_template
        Optional per-frame title template formatted with time in ms.

    Returns
    -------
    str
        Path to the created video file.
    """

    if framerate <= 0:
        raise ValueError("`framerate` must be > 0.")

    views_use: List[str]
    if views is None:
        if isinstance(view, (list, tuple)):
            views_use = [str(v) for v in view]
        else:
            views_use = [str(view)]
    else:
        views_use = list(views)
        if isinstance(view, (list, tuple)) or (isinstance(view, str) and view != "lateral"):
            raise ValueError("Pass either `view` or `views` (not both).")
    if len(views_use) == 0:
        raise ValueError("`views` must contain at least one view.")

    # Disambiguate hemisphere dict vs single-surface dict.
    is_single_surface_dict = (
        isinstance(surf, Mapping)
        and (("v" in surf and "t" in surf) or ("vertices" in surf and "faces" in surf))
    )
    is_hemi_mapping = isinstance(surf, Mapping) and (not is_single_surface_dict)

    if is_hemi_mapping:
        surf_by_hemi: Mapping[str, Any] = surf
        if not isinstance(data_timeseries, Mapping):
            raise ValueError(
                "When `surf` is a hemisphere mapping, `data_timeseries` must be a dict with matching hemisphere keys."
            )
        data_by_hemi = data_timeseries
        if rois is not None and (not isinstance(rois, Mapping)):
            raise ValueError("When `surf` is a hemisphere mapping, `rois` must be a dict (or None).")
        rois_by_hemi: Mapping[str, Optional[np.ndarray]] = (
            {k: np.asarray(v) for k, v in rois.items()} if isinstance(rois, Mapping) else {}
        )
    else:
        surf_by_hemi = {"surf": surf}
        data_by_hemi = {"surf": np.asarray(data_timeseries)}
        rois_by_hemi = {"surf": None if rois is None else np.asarray(rois)}

    hemis = list(surf_by_hemi.keys())
    if len(hemis) == 0:
        raise ValueError("`surf` must contain at least one hemisphere/surface.")

    # Load surfaces + normalize timeseries per hemi.
    vertex_ts_by_hemi: Dict[str, np.ndarray] = {}
    n_frames: Optional[int] = None
    n_maps: Optional[int] = None
    abs_max = 0.0

    for hemi in hemis:
        if hemi not in data_by_hemi:
            raise ValueError(f"Missing data for hemisphere '{hemi}'.")
        poly, n_verts = _load_surface(surf_by_hemi[hemi])
        arr_ts, _ = _prepare_timeseries_scalars(
            data_by_hemi[hemi],
            rois=None if rois is None else rois_by_hemi.get(hemi, None),
            n_verts=n_verts,
        )
        if arr_ts.shape[1] == 0:
            raise ValueError("`data_timeseries` must have at least one frame.")
        if arr_ts.ndim == 2:
            arr_ts = arr_ts[:, :, np.newaxis]
        if n_frames is None:
            n_frames = int(arr_ts.shape[1])
        elif int(arr_ts.shape[1]) != n_frames:
            raise ValueError("All hemispheres must have the same number of frames.")

        if n_maps is None:
            n_maps = int(arr_ts.shape[2])
        elif int(arr_ts.shape[2]) != n_maps:
            raise ValueError("All hemispheres must have the same number of maps.")

        vertex_ts_by_hemi[hemi] = arr_ts
        hemi_abs = float(np.nanmax(np.abs(arr_ts)))
        if np.isfinite(hemi_abs):
            abs_max = max(abs_max, hemi_abs)

    assert n_frames is not None
    assert n_maps is not None

    clim_norm = _normalize_video_clim(
        clim,
        n_frames=n_frames,
        n_maps=n_maps,
        vertex_ts=np.asarray([abs_max], dtype=float),
    )
    clim_fixed: Optional[Tuple[float, float]] = None
    clim_array: Optional[np.ndarray] = None
    if isinstance(clim_norm, tuple):
        clim_fixed = clim_norm
    else:
        clim_array = clim_norm

    # Layout matches `plot_surf`.
    n_hemi = len(hemis)
    n_views = len(views_use)

    if layout_indiv == "row":
        indiv_rows, indiv_cols = 1, n_hemi * n_views
    elif layout_indiv == "col":
        indiv_rows, indiv_cols = n_hemi * n_views, 1
    elif layout_indiv == "grid":
        indiv_rows, indiv_cols = n_views, n_hemi
    else:
        raise ValueError("`layout_indiv` must be one of 'row', 'col', or 'grid'.")

    if layout_group == "row":
        rows, cols = indiv_rows, indiv_cols * n_maps
    elif layout_group == "col":
        rows, cols = indiv_rows * n_maps, indiv_cols
    else:
        raise ValueError("`layout_group` must be one of 'row' or 'col'.")

    panel_w, panel_h = size
    window_size = (int(panel_w * cols), int(panel_h * rows))
    out_path = str(filename)

    mirror_rh = ("lh" in hemis and "rh" in hemis)
    reverse_rh_view_order = mirror_rh and (layout_indiv in ("row", "col"))

    plotter = pv.Plotter(shape=(rows, cols), off_screen=True, window_size=window_size, border=False)
    try:
        # Each cell stores: (hemi, map_idx, scalars_array, actor)
        cell_refs: List[Tuple[str, int, np.ndarray, Optional[Any]]] = []

        for map_idx in range(n_maps):
            if layout_group == "row":
                row_offset, col_offset = 0, map_idx * indiv_cols
            else:
                row_offset, col_offset = map_idx * indiv_rows, 0

            for h_idx, hemi in enumerate(hemis):
                hemi_views = list(views_use)[::-1] if (hemi == "rh" and reverse_rh_view_order) else list(views_use)

                for v_idx, view_name in enumerate(hemi_views):
                    camera_view = _rh_view_swap(view_name) if hemi == "rh" else view_name

                    if layout_indiv == "row":
                        r0, c0 = 0, h_idx * n_views + v_idx
                    elif layout_indiv == "col":
                        r0, c0 = h_idx * n_views + v_idx, 0
                    else:  # grid
                        r0, c0 = v_idx, h_idx

                    r, c = r0 + row_offset, c0 + col_offset
                    plotter.subplot(r, c)

                    if clim_fixed is not None:
                        clim_first = clim_fixed
                    else:
                        assert clim_array is not None
                        if n_maps == 1:
                            clim_first = (float(clim_array[0, 0]), float(clim_array[0, 1]))
                        else:
                            clim_first = (float(clim_array[0, map_idx, 0]), float(clim_array[0, map_idx, 1]))

                    mesh_used, actor, _ = _add_surface_to_plotter(
                        plotter,
                        surf=surf_by_hemi[hemi],
                        data=vertex_ts_by_hemi[hemi][:, 0, map_idx],
                        rois=None if rois is None else rois_by_hemi.get(hemi, None),
                        cbar=cbar,
                        cmap=cmap,
                        mesh_edges=mesh_edges,
                        roi_outlines=roi_outlines,
                        scalar_bar_args=scalar_bar_args,
                        clim=clim_first,
                    )

                    plotter.hide_axes()
                    _finalize_camera(plotter, view=camera_view, zoom=zoom)

                    if "scalars" not in mesh_used.point_data:
                        raise RuntimeError("Internal error: expected 'scalars' point_data on the rendered mesh.")
                    scalars_arr = mesh_used.point_data["scalars"]
                    cell_refs.append((hemi, map_idx, scalars_arr, actor))

        try:
            plotter.open_movie(out_path, framerate=int(framerate))
        except Exception as e:
            raise RuntimeError(
                "Failed to open movie writer. PyVista typically requires ffmpeg for MP4 output; install ffmpeg and retry."
            ) from e

        plotter.show(auto_close=False)

        for frame_idx in range(n_frames):
            for hemi, map_idx, scalars_arr, actor in cell_refs:
                scalars_arr[:] = vertex_ts_by_hemi[hemi][:, frame_idx, map_idx]

                if clim_fixed is not None:
                    clim_use = clim_fixed
                else:
                    assert clim_array is not None
                    if n_maps == 1:
                        clim_use = (float(clim_array[frame_idx, 0]), float(clim_array[frame_idx, 1]))
                    else:
                        clim_use = (
                            float(clim_array[frame_idx, map_idx, 0]),
                            float(clim_array[frame_idx, map_idx, 1]),
                        )

                if actor is not None:
                    try:
                        actor.mapper.scalar_range = clim_use
                    except Exception:
                        pass

            if title_template:
                time_ms = frame_idx * (1000.0 / float(framerate))
                try:
                    plotter.remove_actor("time_text")
                except Exception:
                    pass
                plotter.add_text(
                    title_template.format(time_ms),
                    position="upper_left",
                    font_size=16,
                    name="time_text",
                )

            plotter.write_frame()

    finally:
        plotter.close()

    return out_path


def compute_roi_midline_edges(verts: np.ndarray, faces: np.ndarray, labeling: np.ndarray, verbose: bool = False):
    """
    Compute ROI boundary line segments on a triangular mesh. The boundary is approximated 
    within each triangle using midpoints of edges whose incident vertices belong to 
    different ROI labels.

    Parameters
    ----------
    verts : array_like 
        Vertex coordinates of shape (n_vertices, 3).
    faces : array_like
        Triangle indices into ``verts`` of shape (n_faces, 3).
    labeling : array_like 
        Integer ROI label per vertex of shape (n_vertices,). ``0`` is treated as 
        background/mask (e.g., medial wall). NaNs are converted to 0.
    verbose : bool, optional
        If True, print a message when no boundaries are found.

    Returns
    -------
    xe, ye, ze : np.ndarray
        1D float arrays of equal length encoding the polyline(s) for Plotly
        ``Scatter3d``. Each line segment is represented by two points followed by
        a ``np.nan`` separator (i.e., ``[x0, x1, nan, x0, x1, nan, ...]``).
        If no boundaries are found, all three arrays are empty.

    Notes
    -----
    - If a triangle contains exactly two unique labels (including the common case
      ``{0, X}`` for medial-wall vs ROI), two of its edges will cross a label
      boundary; the function adds a segment connecting the midpoints of those two
      edges.
    - If a triangle contains three unique labels, the function treats it as a
      three-way junction and adds three segments from the triangle centroid to
      the midpoint of each edge.

    """

    labeling = np.asarray(labeling)
    labeling = np.nan_to_num(labeling, nan=0).astype(int)

    tri_labels = labeling[faces]
    tri_coords = verts[faces]

    line_segments = []
    for lbls, coords in zip(tri_labels, tri_coords):
        unique_lbls = np.unique(lbls)

        # Skip all-zero triangles (pure medial wall)
        if np.all(unique_lbls == 0):
            continue

        edges = [(0, 1), (1, 2), (2, 0)]

        # Two or more distinct labels — boundary triangle
        if len(unique_lbls) == 2:
            # Includes case {0, X}
            diff_edges = [e for e in edges if lbls[e[0]] != lbls[e[1]]]
            if len(diff_edges) == 2:
                mids = [coords[list(e)].mean(axis=0) for e in diff_edges]
                line_segments.append(np.vstack(mids))

        elif len(unique_lbls) == 3:
            # Three-way junction: draw centroid-to-midpoint lines
            centroid = coords.mean(axis=0)
            mids = [coords[list(e)].mean(axis=0) for e in edges]
            for m in mids:
                line_segments.append(np.vstack([centroid, m]))

    if not line_segments:
        if verbose:
            print("No ROI boundaries found.")
        return np.array([]), np.array([]), np.array([])

    segs = np.stack(line_segments)
    n = len(segs)
    xe = np.empty(n * 3)
    ye = np.empty_like(xe)
    ze = np.empty_like(xe)

    xe[0::3] = segs[:, 0, 0]
    xe[1::3] = segs[:, 1, 0]
    xe[2::3] = np.nan
    ye[0::3] = segs[:, 0, 1]
    ye[1::3] = segs[:, 1, 1]
    ye[2::3] = np.nan
    ze[0::3] = segs[:, 0, 2]
    ze[1::3] = segs[:, 1, 2]
    ze[2::3] = np.nan

    return xe, ye, ze
