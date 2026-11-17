"""Camera orientation for the 3D view. Kernel-free, and therefore testable.

WHY THIS IS NOT IN `smt/gui/view3d.py`
--------------------------------------
Because it was wrong, and nothing could tell me.

v7 rotated the view with an azimuth and an elevation angle, and got the vertical
sign backwards -- dragging down tipped the part the wrong way. A sign convention
is precisely the sort of thing a test catches instantly and a reading of the
code does not, and it was unreachable by the test suite only because it happened
to live in a module that imports Qt.

So the orientation maths lives here, on plain numpy, next to the transform
solver and the feature classifier for the same reason they are there.

FREE ROTATION, NOT A TURNTABLE
------------------------------
v7 also clamped elevation to plus or minus ninety degrees. That is the standard
MCAD turntable, and the argument for it is real: a turntable keeps "up" meaning
the same thing, so the part never ends up rolled at an angle nobody chose.

The argument against it is that you hit a wall. Tip the part as far as it will
go and it simply stops, which is what a 180 degree limit feels like from the
outside.

This is a free tumble instead. The orientation is carried as an accumulated
rotation MATRIX rather than as a pair of angles, and each drag pre-multiplies a
small rotation expressed in CAMERA axes. There is no pole to hit, no gimbal lock
and no clamp, because there are no Euler angles left to lock up.

The cost is that the part can end up rolled. That is what the named views are
for, and they set the matrix outright rather than nudging it.

SIGN CONVENTION, STATED ONCE
----------------------------
**The surface under the cursor follows the cursor.** Drag down and the face you
are looking at goes down, carrying the top of the part toward you. Drag right
and it goes right.

That is the whole rule, and every sign below is derived from it rather than
guessed:

    p_cam = R @ p_world,  so a rotation D in camera space gives R' = D @ R

    drag right (dx > 0)   front face +Z must move toward +X   -> rotate about
                          camera Y, since Rot_Y(+a) sends Z to X
    drag down  (dy > 0)   front face +Z must move toward -Y   -> rotate about
                          camera X, since Rot_X(+a) sends Z to -Y

Screen Y grows downward, so a downward drag is already a positive dy and needs
no extra negation. v7's bug was an extra one.
"""

from __future__ import annotations

import math

import numpy as np

#: Degrees of rotation per pixel dragged. A drag across a 1000 px viewport is
#: about a full turn, which is roughly where a tumble stops feeling either
#: sluggish or twitchy.
DEGREES_PER_PIXEL = 0.36


def rot_x(angle: float) -> np.ndarray:
    """Right-handed rotation about X. Sends +Y toward +Z, and +Z toward -Y."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Right-handed rotation about Y. Sends +Z toward +X, and +X toward -Z."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


#: World to camera for the front view: screen right is world +X, screen up is
#: world +Z, and world -Y points at the viewer. Z up, which is what every CAD
#: system this imports from uses.
FRONT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])

#: The named viewpoints, as (azimuth about world Z, elevation) in degrees.
NAMED_VIEWS = {
    "iso": (-37.5, 24.0),
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (-90.0, 0.0),
    "right": (90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
}


def from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """A world-to-camera matrix from a turntable azimuth and elevation.

    Used only to build the NAMED views, where angles are the natural way to say
    what is wanted. Free rotation never goes through here, so nothing can lock
    up at the poles: `top` is a matrix, not an elevation of ninety degrees that
    something later has to keep away from.
    """
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return FRONT @ rot_x(el) @ rot_z(az)


def named(name: str) -> np.ndarray:
    az, el = NAMED_VIEWS.get(name, NAMED_VIEWS["iso"])
    return from_angles(az, el)


def tumble(R: np.ndarray, dx: float, dy: float,
           degrees_per_pixel: float = DEGREES_PER_PIXEL) -> np.ndarray:
    """Rotate by a mouse drag of (dx, dy) pixels. Unlimited in every direction.

    `dy` is in screen pixels, so it is POSITIVE DOWNWARD. See the module
    docstring for where each sign comes from; the short version is that the
    surface under the cursor follows the cursor.
    """
    ax = math.radians(dy * degrees_per_pixel)   # about camera X: down-drag
    ay = math.radians(dx * degrees_per_pixel)   # about camera Y: right-drag

    # Camera-space increment, so it pre-multiplies. Applying it the other way
    # round would rotate about the model's own axes, which drifts away from the
    # cursor as soon as the part is not square to the screen.
    delta = rot_y(ay) @ rot_x(ax)
    return orthonormalise(delta @ R)


def orthonormalise(R: np.ndarray) -> np.ndarray:
    """Scrub accumulated floating-point drift out of a rotation matrix.

    Composing thousands of small rotations across a long session slowly loses
    orthogonality, and a matrix that is no longer a rotation shears the model
    instead of turning it. One SVD per drag is nothing, and it makes the drift
    impossible rather than merely slow.
    """
    u, _s, vt = np.linalg.svd(np.asarray(R, dtype=float))
    out = u @ vt
    if np.linalg.det(out) < 0:          # never allow a reflection
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def is_rotation(R: np.ndarray, tol: float = 1e-9) -> bool:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    return bool(
        np.allclose(R @ R.T, np.eye(3), atol=tol)
        and abs(float(np.linalg.det(R)) - 1.0) < tol
    )
