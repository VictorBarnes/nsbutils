"""
Module for loading surface meshes and maps, as well as setting up caching.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from lapy import TriaMesh
from nibabel.gifti.gifti import GiftiImage

fs_extensions = ('.white', '.pial', '.inflated', '.orig', '.sphere', '.smoothwm', '.qsphere',
                 '.fsaverage')

def read_surf(
    surf: str | Path | GiftiImage | TriaMesh | dict
) -> TriaMesh:
    """Load a triangular surface mesh.

    Parameters
    ----------
    surf : str, Path, GiftiImage, lapy.TriaMesh, or dict
        Surface mesh specified as a file path (``str`` or ``Path``) to a VTK (``.vtk``), GIFTI
        (``.gii``), or FreeSurfer file (``.white``, ``.pial``, ``.inflated``, ``.orig``,
        ``.sphere``, ``.smoothwm``, ``.qsphere``, ``.fsaverage``), an instance of
        ``nibabel.GiftiImage`` or ``lapy.TriaMesh``, or a dictionary with ``'vertices'`` and
        ``'faces'`` keys, referencing arrays of shapes ``(n_verts, 3)`` and ``(n_trias, 3)``,
        respectively.

    Returns
    -------
    lapy.TriaMesh
        Surface mesh with vertices and faces.

    Raises
    ------
    ValueError
        If ``surf`` is a path-like string to an unsupported format.
    FileNotFoundError
        If ``surf`` is a path-like string to a file that does not exist.
    """
    if isinstance(surf, TriaMesh):
        return surf
    elif isinstance(surf, GiftiImage):
        vertices=surf.darrays[0].data
        faces=surf.darrays[1].data
    elif isinstance(surf, dict):
        vertices=surf['vertices']
        faces=surf['faces']
    else:
        surf_str = str(surf)
        # check that file exists
        if not Path(surf_str).is_file():
            raise FileNotFoundError(f'File not found: {surf_str}')
        # Handle different file types
        if surf_str.endswith('.vtk'):
            return TriaMesh.read_vtk(surf_str)
        elif surf_str.endswith('.gii'):
            return TriaMesh.read_gifti(surf_str)
        elif surf_str.endswith(fs_extensions):
            return TriaMesh.read_fssurf(surf_str)
        else:
            raise ValueError(
                'surf must be a path-like string to a valid VTK (.vtk), GIFTI (.gii), or '
                f'FreeSurfer file {fs_extensions}, an instance of nibabel.GiftiImage or '
                "lapy.TriaMesh, or a dictionary of 'faces' and 'vertices' with shapes (n_verts, 3) "
                'and (n_trias, 3), respectively.'
                )
        
    return TriaMesh(v=vertices, t=faces)
