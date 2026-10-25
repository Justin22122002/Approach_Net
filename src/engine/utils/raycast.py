import math
import numpy as np

from engine.utils.vec3 import Vec3


def screen_to_ray(
    mouse_x: int, mouse_y: int,
    width: int, height: int,
    view_mat: np.ndarray,
    proj_mat: np.ndarray,
    cam_pos: Vec3,
) -> tuple[Vec3, Vec3]:
    x_ndc = (2.0 * mouse_x) / max(width, 1) - 1.0
    y_ndc = 1.0 - (2.0 * mouse_y) / max(height, 1)

    inv_vp = np.linalg.inv(proj_mat @ view_mat)

    near = inv_vp @ np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float32)
    far  = inv_vp @ np.array([x_ndc, y_ndc,  1.0, 1.0], dtype=np.float32)
    near /= near[3]
    far  /= far[3]

    direction = Vec3(far[0] - near[0], far[1] - near[1], far[2] - near[2]).normalized()
    return cam_pos, direction


def ray_aabb_intersect(
    origin: Vec3,
    direction: Vec3,
    box_min: Vec3,
    box_max: Vec3,
) -> float | None:
    tmin, tmax = 0.0, math.inf
    for o, d, mn, mx in (
        (origin.x, direction.x, box_min.x, box_max.x),
        (origin.y, direction.y, box_min.y, box_max.y),
        (origin.z, direction.z, box_min.z, box_max.z),
    ):
        if abs(d) < 1e-9:
            if o < mn or o > mx:
                return None
            continue
        t1 = (mn - o) / d
        t2 = (mx - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None
    return tmin

def ray_plane_intersect(
    origin: Vec3,
    direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Vec3 | None:
    denom = direction.dot(plane_normal)
    if abs(denom) < 1e-9:
        return None
    t = (plane_point - origin).dot(plane_normal) / denom
    if t < 0:
        return None
    return origin + direction * t