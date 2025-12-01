import math
from typing import Tuple

Vec3 = Tuple[float, float, float]

# ---------------- Vector utilities ----------------

def dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def mul(a: Vec3, k: float) -> Vec3:
    return (a[0]*k, a[1]*k, a[2]*k)


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---------------- Geometry helpers ----------------

def dist_point_to_segment(P: Vec3, A: Vec3, B: Vec3) -> float:
    """Distance from point P to segment AB in 3D."""
    AB = sub(B, A)
    AP = sub(P, A)
    ab2 = dot(AB, AB)
    if ab2 == 0.0:
        return norm(sub(P, A))
    t = clamp(dot(AP, AB) / ab2, 0.0, 1.0)
    closest = add(A, mul(AB, t))
    return norm(sub(P, closest))

# ---------------- Problem setup ----------------

def heading_to_unit_new(theta_deg: float) -> Tuple[float, float]:
    """New standard: 0 deg along +x, CCW positive (xy-plane). Returns (ux, uy)."""
    th = math.radians(theta_deg)
    return (math.cos(th), math.sin(th))


def missile_state_M1(t: float) -> Vec3:
    """M1 position at time t [s]. Missile heads towards (0,0,0) at 300 m/s."""
    M0 = (20000.0, 0.0, 2000.0)
    v_dir = (-20000.0, 0.0, -2000.0)
    L = math.sqrt(v_dir[0]**2 + v_dir[1]**2 + v_dir[2]**2)
    u = (v_dir[0]/L, v_dir[1]/L, v_dir[2]/L)
    v = (u[0]*300.0, u[1]*300.0, u[2]*300.0)
    return (M0[0] + v[0]*t, M0[1] + v[1]*t, M0[2] + v[2]*t)


def compute_explosion_point_FY1_new_standard(speed: float = 132.86,
                                             heading_deg: float = 6.63,
                                             t_release: float = 0.12,
                                             t_fuse: float = 0.68,
                                             g: float = 9.8) -> Tuple[Vec3, float, Tuple[float, float]]:
    """
    Compute FY1 explosion point using the NEW heading standard directly.
    Returns (Pe, t_explosion, (ux, uy)).
    """
    FY1 = (17800.0, 0.0, 1800.0)

    ux, uy = heading_to_unit_new(heading_deg)

    # Drone horizontal velocity components
    vx = speed * ux
    vy = speed * uy

    # Release point (constant altitude prior to release)
    R = (FY1[0] + vx*t_release, FY1[1] + vy*t_release, FY1[2])

    # Free fall after release for t_fuse seconds: horizontal continues, vertical under gravity
    x_e = R[0] + vx*t_fuse
    y_e = R[1] + vy*t_fuse
    z_e = R[2] - 0.5*g*(t_fuse**2)

    Pe = (x_e, y_e, z_e)
    t_explosion = t_release + t_fuse
    return Pe, t_explosion, (ux, uy)


def cloud_center_after(Pe: Vec3, t: float, t_explosion: float, sink_speed: float = 3.0) -> Vec3:
    """Cloud center position at time t >= t_explosion. Only z changes (downwards)."""
    dt = max(0.0, t - t_explosion)
    return (Pe[0], Pe[1], Pe[2] - sink_speed*dt)


def effective_obscuration_time_for_M1() -> float:
    """
    Compute total effective obscuration time (seconds) for M1 using the new heading standard
    with given parameters (heading=6.63 deg, speed=132.86 m/s, release at 0.12 s, explode at 0.80 s).
    Criterion: distance from cloud center to segment [M(t), T] <= 10 m within 20 s after explosion.
    """
    # Parameters
    speed = 132.86
    heading_deg = 6.63  # new standard (0 deg = +x, CCW)
    t_release = 0.12
    t_fuse = 0.68
    g = 9.8

    # Target (true target) center of bottom face
    T = (0.0, 200.0, 0.0)

    # Compute explosion point and time
    Pe, t_explosion, _ = compute_explosion_point_FY1_new_standard(speed, heading_deg, t_release, t_fuse, g)

    # Simulation window
    t0 = t_explosion
    t1 = t_explosion + 20.0

    # Numerical integration
    R_eff = 10.0
    dt = 0.001  # 1 ms resolution for good accuracy

    t = t0
    prev_d = None
    total = 0.0

    while t <= t1 + 1e-12:
        C = cloud_center_after(Pe, t, t_explosion, sink_speed=3.0)
        M = missile_state_M1(t)
        d = dist_point_to_segment(C, M, T)

        if prev_d is not None:
            if (prev_d <= R_eff and d <= R_eff):
                total += dt
            elif (prev_d <= R_eff and d > R_eff):
                # leaving interval, add partial within this step
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    if 0.0 < alpha < 1.0:
                        total += alpha * dt
            elif (prev_d > R_eff and d <= R_eff):
                # entering interval, add partial within this step
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    if 0.0 < alpha < 1.0:
                        total += (1.0 - alpha) * dt
                else:
                    total += dt
        prev_d = d
        t += dt

    return total


def main():
    Pe, t_explosion, (ux, uy) = compute_explosion_point_FY1_new_standard()
    total_time = effective_obscuration_time_for_M1()

    print("采用新方位角定义：0°沿+x，逆时针为正（0~360°）")
    print(f"FY1 航向 = 6.63°（新标准），速度 = 132.86 m/s")
    print(f"释放时刻 t_release = 0.12 s，起爆时刻 t_explosion = {t_explosion:.2f} s")
    print(f"起爆点 Pe = ({Pe[0]:.3f}, {Pe[1]:.3f}, {Pe[2]:.3f})")
    print(f"云团中心以 3 m/s 匀速下沉，有效期 20 s，判据：距离[M1(t), T]线段 ≤ 10 m")
    print(f"对 M1 的有效遮蔽总时长 = {total_time:.6f} s")


if __name__ == "__main__":
    main()