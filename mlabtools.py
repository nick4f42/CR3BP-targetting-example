"""Tools for plotting in mayavi."""

__author__ = "Nick OBrien"

import numpy as np

from mayavi import mlab
from tvtk.api import tvtk


def lock_upaxis(on=True, figure=None):
    """Lock the Z-axis to remain upward relative to the view."""
    if figure is None:
        figure = mlab.gcf()
    figure.scene.interactor.interactor_style = (
        tvtk.InteractorStyleTerrain() if on else tvtk.InteractorStyleTrackballCamera())


def _line_indices(size, i):
    return np.column_stack((np.arange(i, i + size - 1),
                            np.arange(i + 1, i + size)))

def _lines_source(points):
    try:
        points[0][0][0]

        lines = []
        i = 0
        for pt in points:
            lines.append(_line_indices(len(pt[0]), i))
            i += len(pt[0])
        src = mlab.pipeline.scalar_scatter(*np.hstack(points))
        src.mlab_source.dataset.lines = np.concatenate(lines)

    except (IndexError, TypeError):
        src = mlab.pipeline.scalar_scatter(*points)
        src.mlab_source.dataset.lines = _line_indices(len(points[0]), 0)

    src.update()
    return src


def surface(*xyz, **kwargs):
    """Add a mayavi surface from a list of points.

    Args:
        *xyz: List of coordinate values, either (x, y, z) or (x, y, z, s).
            s is the optional scalar values associated with each point.
        **kwargs: Keyword arguments are passed to `mlab.pipeline.surface`
            https://docs.enthought.com/mayavi/mayavi/auto/mlab_pipeline_other_functions.html#mayavi.tools.pipeline.surface

    Returns:
        The mayavi surface.
    """
    src = mlab.pipeline.scalar_scatter(*xyz)
    surf = mlab.pipeline.surface(src, **kwargs)
    return surf


def curves(points, **kwargs):
    """Add line segments through each list of points in `points`.

    Args:
        points: Either [x, y, z], [x, y, z, s], or a list of them.
            s is the optional scalar values associated with each point.
        **kwargs: Keyword arguments are passed to `mlab.pipeline.surface`
            https://docs.enthought.com/mayavi/mayavi/auto/mlab_pipeline_other_functions.html#mayavi.tools.pipeline.surface


    Returns:
        The mayavi surface.
    """
    src = _lines_source(points)
    surf = mlab.pipeline.surface(src, **kwargs)
    return surf


def tubes(points, tube_radius=0.05, tube_sides=6, **kwargs):
    """Add tubes through each list of points in `points`.

    Args:
        points: Either [x, y, z], [x, y, z, s], or a list of them.
            s is the optional scalar value associated with each point.
        tube_radius (float, optional): Radius of the tubes.
        tube_sides (int, optional): Number of sides of the tubes.
        **kwargs: Keyword arguments are passed to `mlab.pipeline.surface`
            https://docs.enthought.com/mayavi/mayavi/auto/mlab_pipeline_other_functions.html#mayavi.tools.pipeline.surface

    Returns:
        The mayavi surface.
    """
    src = _lines_source(points)
    tube = mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=tube_sides)
    return mlab.pipeline.surface(tube, **kwargs)


def sphere(radius,
           color=None,
           map_filename=None,
           position=(0, 0, 0),
           orientation=(0, 0, 0),
           resolution=120):
    """Add a colored or textured sphere to the current figure.

    Args:
        radius: The radius of the sphere.
        color (optional): An RGB tuple ranging 0 through 1 for the sphere color.
        map_filename (optional): The filename of the cylindrical map used for
            the sphere. Note: only png and jpg files are currently supported.
        position (tuple, optional): The 3D position of the sphere's center.
        orientation (tuple, optional): The sphere's Euler orientation (degrees).
        resolution (int, optional): The resolution of the sphere mesh.

    Returns:
        The sphere's tvtk Actor.

    Raises:
        ValueError: Did not specify a color or map_filename.
    """
    if map_filename:

        if any(map_filename.endswith(s) for s in ('jpg', 'jpeg')):
            img = tvtk.JPEGReader()
        elif map_filename.endswith('png'):
            img = tvtk.PNGReader()
        else:
            raise NotImplementedError('Only png and jpg files are currently supported.')
        img.file_name = map_filename

        texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)

        sphere = tvtk.TexturedSphereSource(radius=radius,
            theta_resolution=resolution, phi_resolution=resolution)

        sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
        sphere_actor = tvtk.Actor(mapper=sphere_mapper, texture=texture)
        sphere_actor.orientation = orientation

        mlab.gcf().scene.add_actor(sphere_actor)
    elif color:
        p = mlab.points3d(0, 0, 0, color=color, resolution=resolution,
            scale_factor=2 * radius)
        sphere_actor = p.actor.actor
    else:
        raise ValueError("Must specify either color or map_filename.")

    sphere_actor.position = position
    return sphere_actor


def axes(length, pos=(0, 0, 0), labels=None, label_color=None, label_font='times', label_fontsize=24):
    """Add a set of axes to the plot.

    Args:
        length: The length of each axis.

    Kwargs:
        pos: The position of the axes center.
        labels: A list of each axis label.
        label_color: The RGB color of each label.
        label_font: The font family of each axis label.
            Can be either 'times', 'arial', or 'courier'.
        label_fontsize: The font size of each axis label.
    """
    axis_vec = np.zeros((2, 3), float)
    axis_vec[1,0] = length
    for i in range(3):
        vec = pos + np.roll(axis_vec, i, axis=1)
        curves(vec.T, color=tuple(float(i == j) for j in range(3)))
        if labels is not None:
            txt = mlab.text(*vec[1,:2], labels[i], z=vec[1,2])
            txt.property.trait_set(font_family=label_font, font_size=label_fontsize)
            if label_color is not None:
                txt.property.color = label_color
            txt.actor.text_scale_mode = 'none'

