"""PyVista-based surface plotting utilities.

This module provides `plot_surf_single` and `plot_surf` functions that mirror the
behavior of the Plotly-based API the project uses elsewhere, but renders using
PyVista/VTK.

PyVista is treated as an optional dependency and is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


_DEFAULT_PANEL_SIZE: Tuple[int, int] = (400, 300)  # (width, height) in pixels


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


def _try_read_surf_neuromodes(surf: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Try to load a surface via neuromodes.io.read_surf.

    Returns (verts, faces) or None if neuromodes is unavailable or fails.
    """

    try:
        from neuromodes.io import read_surf  # type: ignore
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

    # Try neuromodes first for known gifti-like extensions
    path = _as_pathlike(surf)
    if path is not None and path.suffix.lower() == ".gii":
        nm = _try_read_surf_neuromodes(surf)
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
            nm = _try_read_surf_neuromodes(surf)
            if nm is not None:
                verts, faces = nm
                poly = _polydata_from_verts_faces(pv, verts, faces)
                return poly, poly.n_points
            raise

    # Fallback: try neuromodes for non-path objects (GiftiImage, lapy mesh, etc.)
    nm = _try_read_surf_neuromodes(surf)
    if nm is not None:
        verts, faces = nm
        poly = _polydata_from_verts_faces(pv, verts, faces)
        return poly, poly.n_points

    raise ValueError(
        "Unsupported `surf` input. Provide a file path, a (verts, faces) tuple, a dict with keys "
        "{'v','t'}, or an object supported by neuromodes.io.read_surf (if installed)."
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
    *,
    ax: Optional[Axes] = None,
    scale: float = 1.0,
    scalar_bar_args: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Render a single surface into a PyVista subplot or embed into a Matplotlib axis.

    - If `ax` is provided: renders off-screen and draws an image into `ax`, returning None.
    - Else: draws into `plotter` (or creates one) and returns the plotter.

    Parameters are intentionally aligned with the Plotly version, except `plot_surf_single`
    does not take Plotly-specific `fig/row/col`.
    """

    if ax is not None:
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

    _apply_camera_headlight(plotter)

    mesh, n_verts = _load_surface(surf)
    vertex_data, roi_labels = _prepare_vertex_scalars(data, rois, n_verts)

    # Base mesh + optional scalars
    show_scalar_bar = bool(cbar and vertex_data is not None)
    sb_args = {**_default_scalar_bar_args(), **(scalar_bar_args or {})} if show_scalar_bar else {}

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
        mesh = mesh.copy(deep=False)
        mesh.point_data["scalars"] = vertex_data
        actor = plotter.add_mesh(
            mesh,
            scalars="scalars",
            cmap=cmap,
            nan_color="lightgrey",
            smooth_shading=True,
            show_edges=mesh_edges,
            edge_color="black",
            line_width=0.5,
            # Scalar bars are handled manually below to avoid overwriting across subplots.
            show_scalar_bar=False,
            ambient=0.01,
            diffuse=1,
            specular=0.1,
            roughness=1e-6,
        )

        if show_scalar_bar:
            # PyVista keys scalar bars by their title, and repeated titles overwrite.
            # Give each subplot renderer a unique internal key while keeping the
            # displayed title blank unless the user provided one.
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

    # ROI outlines
    if roi_outlines and roi_labels is not None:
        xe, ye, ze = compute_roi_midline_edges(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:], roi_labels)
        outline_poly = _polydata_from_nan_separated_segments(pv, xe, ye, ze)
        if outline_poly.n_points:
            plotter.add_mesh(outline_poly, color="black", line_width=1.7)

    plotter.hide_axes()
    _finalize_camera(plotter, view=view, zoom=zoom)

    return plotter if (created_plotter or plotter is not None) else None


def plot_surf(
    surf: Mapping[str, Any],
    data: Optional[Mapping[str, np.ndarray]] = None,
    rois: Optional[Mapping[str, np.ndarray]] = None,
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
    scale: float = 1.0,
    scalar_bar_args: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Plot surface data across hemispheres, views, and (optionally) multiple maps.

    Mirrors the Plotly API you provided, but returns a `pyvista.Plotter` when `ax is None`.
    If `ax` is provided, renders off-screen and embeds a rasterized image.
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

    off_screen = ax is not None
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=window_size,
        off_screen=off_screen,
        border=False,
    )

    mirror_rh = ("lh" in hemis and "rh" in hemis)
    reverse_rh_view_order = mirror_rh and (layout_indiv in ("row", "col"))

    for map_idx in range(n_maps):
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


def compute_roi_midline_edges(verts: np.ndarray, faces: np.ndarray, labeling: np.ndarray, verbose: bool = False):
    """Compute ROI boundaries using midpoints between label boundaries.

    Matches MATLAB findROIboundaries.m behavior, including medial wall borders.

    This function is copied from the Plotly implementation and is backend-agnostic.
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
