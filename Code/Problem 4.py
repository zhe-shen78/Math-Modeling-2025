import math
import argparse
import random
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


def cloud_center_after(Pe: Vec3, t: float, t_explosion: float, sink_speed: float = 3.0) -> Vec3:
    """Cloud center position at time t >= t_explosion. Only z changes (downwards)."""
    dt = max(0.0, t - t_explosion)
    return (Pe[0], Pe[1], Pe[2] - sink_speed*dt)


def effective_obscuration_time_for_M1(speed: float,
                                      heading_deg: float,
                                      t_release: float,
                                      t_fuse: float,
                                      g: float = 9.8,
                                      dt: float = 0.001) -> float:
    """
    Compute total effective obscuration time (seconds) for M1 using given parameters.
    Criterion: distance from cloud center to segment [M(t), T] <= 10 m within 20 s after explosion.
    dt: simulation step size in seconds (use a coarser dt for faster optimization, e.g., 0.02; refine later with 0.001).
    """
    # Target (true target) center of bottom face
    T = (0.0, 200.0, 0.0)

    # Compute explosion point and time
    Pe, t_explosion, _ = compute_explosion_point_FY1_new_standard(speed, heading_deg, t_release, t_fuse, g)

    # Simulation window
    t0 = t_explosion
    t1 = t_explosion + 20.0

    # Numerical integration
    R_eff = 10.0

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
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    if 0.0 < alpha < 1.0:
                        total += alpha * dt
            elif (prev_d > R_eff and d <= R_eff):
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

# ---------------- Optimization: Particle Swarm Optimization (PSO) ----------------

def wrap_angle_deg(a: float) -> float:
    """Wrap angle to [0, 360]."""
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a


def shortest_ang_diff_deg(target: float, current: float) -> float:
    """Return signed shortest angular difference (target - current) in degrees within [-180, 180]."""
    d = (target - current + 180.0) % 360.0 - 180.0
    return d


def cloud_interval_and_closest_M1(Pe: Vec3, te: float, dt: float = 0.001, R_eff: float = 10.0):
    Tt = (0.0, 200.0, 0.0)
    t0 = te
    t1 = te + 20.0
    prev_d = None
    prev_t = t0
    t_enter = None
    t_exit = None
    d_min = float('inf')
    t_at_min = t0

    t = t0
    while t <= t1 + 1e-12:
        C = cloud_center_after(Pe, t, te, sink_speed=3.0)
        M = missile_state_M1(t)
        d = dist_point_to_segment(C, M, Tt)
        # track minimum distance and its time
        if d < d_min:
            d_min = d
            t_at_min = t
        if prev_d is not None:
            if prev_d > R_eff and d <= R_eff:
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    if 0.0 < alpha < 1.0:
                        t_cross = prev_t + alpha * dt
                    else:
                        t_cross = t
                else:
                    t_cross = t
                if t_enter is None:
                    t_enter = t_cross
            elif prev_d <= R_eff and d > R_eff:
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    if 0.0 < alpha < 1.0:
                        t_cross = prev_t + alpha * dt
                    else:
                        t_cross = t
                else:
                    t_cross = t
                t_exit = t_cross
                if t_enter is not None:
                    break
        prev_d = d
        prev_t = t
        t += dt

    if t_enter is not None and t_exit is None:
        t_exit = min(t1, prev_t)
    interval = None if t_enter is None else (t_enter, t_exit, max(0.0, t_exit - t_enter))
    return interval, t_at_min, d_min

def pso_optimize(iters: int = 150,
                 swarm_size: int = 25,
                 seed: int = 42,
                 speed_min: float = 70.0,
                 speed_max: float = 140.0,
                 heading_min: float = 0.0,
                 heading_max: float = 360.0,
                 t_release_min: float = 1e-3,
                 t_release_max: float = 3.0,
                 t_fuse_min: float = 1e-3,
                 t_fuse_max: float = 1.5,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 dt_coarse: float = 0.03) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Particle Swarm Optimization to maximize obscuration time.
    Returns (best_params_tuple, best_value), where params = (speed, heading_deg, t_release, t_fuse).
    Runtime is bounded by swarm_size * iters objective evaluations with coarse dt.
    """
    random.seed(seed)

    # Ranges for clamping and velocity limits
    rng_speed = speed_max - speed_min
    rng_heading = heading_max - heading_min  # assumed 360
    rng_tr = t_release_max - t_release_min
    rng_tf = t_fuse_max - t_fuse_min

    # Velocity limits (keep stable)
    vmax = [rng_speed * 0.25, 90.0, rng_tr * 0.25, rng_tf * 0.25]

    def clamp_param(x, lo, hi):
        return max(lo, min(hi, x))

    def obj(xx):
        return effective_obscuration_time_for_M1(xx[0], xx[1], xx[2], xx[3], dt=dt_coarse)

    # Initialize swarm
    X = []  # positions
    V = []  # velocities
    Pbest = []  # personal best positions
    PbestVal = []

    Gbest = None
    GbestVal = -1e18

    for _ in range(swarm_size):
        s = random.uniform(speed_min, speed_max)
        h = random.uniform(heading_min, heading_max)
        tr = random.uniform(t_release_min, t_release_max)
        tf = random.uniform(t_fuse_min, t_fuse_max)
        x = [s, h, tr, tf]
        # Small initial velocities
        v = [random.uniform(-rng_speed*0.05, rng_speed*0.05),
             random.uniform(-30.0, 30.0),
             random.uniform(-rng_tr*0.05, rng_tr*0.05),
             random.uniform(-rng_tf*0.05, rng_tf*0.05)]
        X.append(x)
        V.append(v)
        val = obj(x)
        Pbest.append(x[:])
        PbestVal.append(val)
        if val > GbestVal:
            GbestVal = val
            Gbest = x[:]

    # PSO main loop
    for it in range(iters):
        for i in range(swarm_size):
            x = X[i]
            v = V[i]

            # Update velocity per dimension
            # speed
            r1 = random.random(); r2 = random.random()
            v0 = (w * v[0]
                  + c1 * r1 * (Pbest[i][0] - x[0])
                  + c2 * r2 * (Gbest[i][0] - x[0]))
            # heading (circular distance)
            r1 = random.random(); r2 = random.random()
            diff_p = shortest_ang_diff_deg(Pbest[i][1], x[1])
            diff_g = shortest_ang_diff_deg(Gbest[i][1], x[1])
            v1 = (w * v[1]
                  + c1 * r1 * diff_p
                  + c2 * r2 * diff_g)
            # t_release
            r1 = random.random(); r2 = random.random()
            v2 = (w * v[2]
                  + c1 * r1 * (Pbest[i][2] - x[2])
                  + c2 * r2 * (Gbest[i][2] - x[2]))
            # t_fuse
            r1 = random.random(); r2 = random.random()
            v3 = (w * v[3]
                  + c1 * r1 * (Pbest[i][3] - x[3])
                  + c2 * r2 * (Gbest[i][3] - x[3]))

            # Velocity clamp
            v0 = max(-vmax[0], min(vmax[0], v0))
            v1 = max(-vmax[1], min(vmax[1], v1))
            v2 = max(-vmax[2], min(vmax[2], v2))
            v3 = max(-vmax[3], min(vmax[3], v3))
            v = [v0, v1, v2, v3]

            # Update position
            x0 = clamp_param(x[0] + v0, speed_min, speed_max)
            x1 = wrap_angle_deg(x[1] + v1)
            x2 = clamp_param(x[2] + v2, t_release_min, t_release_max)
            x3 = clamp_param(x[3] + v3, t_fuse_min, t_fuse_max)
            x = [x0, x1, x2, x3]

            # Save back
            X[i] = x
            V[i] = v

            # Evaluate and update bests
            val = obj(x)
            if val > PbestVal[i]:
                Pbest[i] = x[:]
                PbestVal[i] = val
                if val > GbestVal:
                    GbestVal = val
                    Gbest = x[:]

    return (Gbest[0], Gbest[1], Gbest[2], Gbest[3]), GbestVal


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


# --------- New: Generic explosion point for any UAV (FY) under new heading standard ---------

def compute_explosion_point_generic_new_standard(FY: Vec3,
                                                 speed: float,
                                                 heading_deg: float,
                                                 t_release: float,
                                                 t_fuse: float,
                                                 g: float = 9.8) -> Tuple[Vec3, float, Tuple[float, float]]:
    """Compute explosion point for a UAV starting at FY with given speed/heading.
    Returns (Pe, t_explosion, (ux, uy))."""
    ux, uy = heading_to_unit_new(heading_deg)
    vx = speed * ux
    vy = speed * uy
    R = (FY[0] + vx*t_release, FY[1] + vy*t_release, FY[2])
    x_e = R[0] + vx*t_fuse
    y_e = R[1] + vy*t_fuse
    z_e = R[2] - 0.5*g*(t_fuse**2)
    Pe = (x_e, y_e, z_e)
    t_explosion = t_release + t_fuse
    return Pe, t_explosion, (ux, uy)


# --------- New: Multi-cloud effective obscuration time (union over 3 UAVs) ---------

def effective_obscuration_time_for_M1_three(speed1: float, heading1: float, t_release1: float, t_fuse1: float,
                                            speed2: float, heading2: float, t_release2: float, t_fuse2: float,
                                            speed3: float, heading3: float, t_release3: float, t_fuse3: float,
                                            dt: float = 0.001,
                                            g: float = 9.8) -> float:
    """Total effective obscuration time (seconds) against M1 with three UAVs (FY1, FY2, FY3),
    each dropping one smoke. The union of effective intervals is measured.
    Criterion per cloud i: distance from cloud center to segment [M(t), T] <= 10 m within 20 s after explosion."""
    # Target
    T = (0.0, 200.0, 0.0)

    # UAV initial positions (fixed for this task)
    FY1 = (17800.0, 0.0, 1800.0)
    FY2 = (12000.0, 1400.0, 1400.0)
    FY3 = (6000.0, -3000.0, 700.0)

    # Compute explosion points and windows
    Pe1, te1, _ = compute_explosion_point_generic_new_standard(FY1, speed1, heading1, t_release1, t_fuse1, g)
    Pe2, te2, _ = compute_explosion_point_generic_new_standard(FY2, speed2, heading2, t_release2, t_fuse2, g)
    Pe3, te3, _ = compute_explosion_point_generic_new_standard(FY3, speed3, heading3, t_release3, t_fuse3, g)

    # Time window overall
    t0 = min(te1, te2, te3)
    t1 = max(te1, te2, te3) + 20.0

    R_eff = 10.0
    total = 0.0
    t = t0
    prev_min_d = None

    while t <= t1 + 1e-12:
        # Missile position
        M = missile_state_M1(t)

        # Active clouds and their distances
        dists = []
        if te1 <= t <= te1 + 20.0:
            C1 = cloud_center_after(Pe1, t, te1, sink_speed=3.0)
            dists.append(dist_point_to_segment(C1, M, T))
        if te2 <= t <= te2 + 20.0:
            C2 = cloud_center_after(Pe2, t, te2, sink_speed=3.0)
            dists.append(dist_point_to_segment(C2, M, T))
        if te3 <= t <= te3 + 20.0:
            C3 = cloud_center_after(Pe3, t, te3, sink_speed=3.0)
            dists.append(dist_point_to_segment(C3, M, T))

        if dists:
            dmin = min(dists)
        else:
            dmin = float('inf')

        if prev_min_d is not None:
            # If staying inside
            if prev_min_d <= R_eff and dmin <= R_eff:
                total += dt
            # Exit crossing
            elif prev_min_d <= R_eff and dmin > R_eff:
                denom = (dmin - prev_min_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_min_d) / denom
                    if 0.0 < alpha < 1.0:
                        total += alpha * dt
            # Enter crossing
            elif prev_min_d > R_eff and dmin <= R_eff:
                denom = (dmin - prev_min_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_min_d) / denom
                    if 0.0 < alpha < 1.0:
                        total += (1.0 - alpha) * dt
                else:
                    total += dt
        prev_min_d = dmin
        t += dt

    return total


# --------- New: PSO for three UAVs (12-dimensional search) ---------

# --------- Helper: 单个云团对 M1 的遮蔽时间窗（提前放置以便 main/PSO 调用）---------
# 返回 (t_enter, t_exit, duration)，计算区间 [te, te+20]，当云团中心到[M1(t), T]线段距离 <= R_eff 判为遮蔽
# 若从未有效则返回 None

def single_cloud_interval_M1(Pe: Vec3, te: float, dt: float = 0.001, R_eff: float = 10.0):
    Tt = (0.0, 200.0, 0.0)
    t0 = te
    t1 = te + 20.0
    prev_d = None
    prev_t = t0
    t_enter = None
    t_exit = None

    t = t0
    while t <= t1 + 1e-12:
        C = cloud_center_after(Pe, t, te, sink_speed=3.0)
        M = missile_state_M1(t)
        d = dist_point_to_segment(C, M, Tt)
        if prev_d is not None:
            # crossing detection between prev_t and t
            if prev_d > R_eff and d <= R_eff:
                # entering
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    if 0.0 < alpha < 1.0:
                        t_cross = prev_t + alpha * dt
                    else:
                        t_cross = t
                else:
                    t_cross = t
                if t_enter is None:
                    t_enter = t_cross
            elif prev_d <= R_eff and d > R_eff:
                # exiting
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    if 0.0 < alpha < 1.0:
                        t_cross = prev_t + alpha * dt
                    else:
                        t_cross = t
                else:
                    t_cross = t
                t_exit = t_cross
                if t_enter is not None:
                    break
        prev_d = d
        prev_t = t
        t += dt

    if t_enter is not None and t_exit is None:
        t_exit = min(t1, prev_t)
    if t_enter is None:
        return None
    duration = max(0.0, t_exit - t_enter)
    return (t_enter, t_exit, duration)


def pso_optimize_three(iters: int = 180,
                       swarm_size: int = 36,
                       seed: int = 42,
                       speed_min: float = 70.0,
                       speed_max: float = 140.0,
                       heading_min: float = 0.0,
                       heading_max: float = 360.0,
                       t_release_min: float = 1e-3,
                       t_release_max: float = 60.0,
                       t_fuse_min: float = 1e-3,
                       t_fuse_max: float = 3.0,
                       w: float = 0.7,
                       c1: float = 1.5,
                       c2: float = 1.5,
                       dt_coarse: float = 0.03,
                       t_start1: float = 1.25,
                       t_start2: float = 20.75,
                       t_start3: float = 33.75,
                       d_target1: float = 4.65,
                       d_target2: float = 3.90,
                       d_target3: float = 3.50,
                       w_start: float = 2.5,
                       w_dur: float = 3.0,
                       miss_penalty: float = 25.0,
                       w_close: float = 2.0,
                       w_align: float = 4.0,
                       w_te: float = 3.0,
                       te_offset: float = 0.62) -> Tuple[Tuple[float, ...], float]:
    """PSO to maximize union obscuration time for three UAVs, each dropping one smoke.
    Returns (best_params_tuple(12,), best_value). Params order per UAV i: (si, hi, tri, tfi)."""
    random.seed(seed)

    # Ranges
    rng_speed = speed_max - speed_min
    rng_heading = heading_max - heading_min
    rng_tr = t_release_max - t_release_min
    rng_tf = t_fuse_max - t_fuse_min

    # Velocity limits (replicated for 3 UAVs)
    vmax_one = [rng_speed * 0.25, 90.0, rng_tr * 0.25, rng_tf * 0.25]
    vmax = vmax_one * 3

    def clamp_param(x, lo, hi):
        return max(lo, min(hi, x))

    def obj(xx):
        # 基础目标：三团对M1遮蔽并集总时长（粗dt）
        base = effective_obscuration_time_for_M1_three(
            xx[0], xx[1], xx[2], xx[3],
            xx[4], xx[5], xx[6], xx[7],
            xx[8], xx[9], xx[10], xx[11],
            dt=dt_coarse)
        # 引导惩罚：使各团遮蔽开始时间与持续时长靠近目标；若无窗口，用最近接近度与时间提供软信号；同时约束起爆时刻与目标开始时刻的经验偏移
        FY1 = (17800.0, 0.0, 1800.0)
        FY2 = (12000.0, 1400.0, 1400.0)
        FY3 = (6000.0, -3000.0, 700.0)
        (Pe1, te1, _) = compute_explosion_point_generic_new_standard(FY1, xx[0], xx[1], xx[2], xx[3])
        (Pe2, te2, _) = compute_explosion_point_generic_new_standard(FY2, xx[4], xx[5], xx[6], xx[7])
        (Pe3, te3, _) = compute_explosion_point_generic_new_standard(FY3, xx[8], xx[9], xx[10], xx[11])
        i1, tmin1, dmin1 = cloud_interval_and_closest_M1(Pe1, te1, dt=dt_coarse)
        i2, tmin2, dmin2 = cloud_interval_and_closest_M1(Pe2, te2, dt=dt_coarse)
        i3, tmin3, dmin3 = cloud_interval_and_closest_M1(Pe3, te3, dt=dt_coarse)
        pen = 0.0
        for (interval, ts, dur_target, tmin, dmin, te) in (
            (i1, t_start1, d_target1, tmin1, dmin1, te1),
            (i2, t_start2, d_target2, tmin2, dmin2, te2),
            (i3, t_start3, d_target3, tmin3, dmin3, te3)):
            # te 对齐（经验偏移）
            pen += w_te * abs((te + te_offset) - ts)
            if interval is None:
                pen += miss_penalty
                pen += w_close * max(0.0, dmin - 10.0)
                pen += w_align * abs(tmin - ts)
            else:
                t_enter, t_exit, dur = interval
                pen += w_start * abs(t_enter - ts) + w_dur * abs(dur - dur_target)
        return base - pen

    # Initialize swarm
    X = []
    V = []
    Pbest = []
    PbestVal = []
    Gbest = None
    GbestVal = -1e18

    for _ in range(swarm_size):
        x = []
        v = []
        for _j in range(3):
            s = random.uniform(speed_min, speed_max)
            h = random.uniform(heading_min, heading_max)
            tr = random.uniform(t_release_min, t_release_max)
            tf = random.uniform(t_fuse_min, t_fuse_max)
            x.extend([s, h, tr, tf])
            v.extend([
                random.uniform(-rng_speed*0.05, rng_speed*0.05),
                random.uniform(-30.0, 30.0),
                random.uniform(-rng_tr*0.05, rng_tr*0.05),
                random.uniform(-rng_tf*0.05, rng_tf*0.05)
            ])
        X.append(x)
        V.append(v)
        val = obj(x)
        Pbest.append(x[:])
        PbestVal.append(val)
        if val > GbestVal:
            GbestVal = val
            Gbest = x[:]

    # PSO iterations
    for _it in range(iters):
        for i in range(swarm_size):
            x = X[i]
            v = V[i]

            # Update velocity and position per dimension
            for j in range(3):
                base = 4*j
                # speed
                r1 = random.random(); r2 = random.random()
                v0 = (w * v[base+0]
                      + c1 * r1 * (Pbest[i][base+0] - x[base+0])
                      + c2 * r2 * (Gbest[base+0] - x[base+0]))
                # heading (circular)
                r1 = random.random(); r2 = random.random()
                diff_p = shortest_ang_diff_deg(Pbest[i][base+1], x[base+1])
                diff_g = shortest_ang_diff_deg(Gbest[base+1], x[base+1])
                v1 = (w * v[base+1]
                      + c1 * r1 * diff_p
                      + c2 * r2 * diff_g)
                # t_release
                r1 = random.random(); r2 = random.random()
                v2 = (w * v[base+2]
                      + c1 * r1 * (Pbest[i][base+2] - x[base+2])
                      + c2 * r2 * (Gbest[base+2] - x[base+2]))
                # t_fuse
                r1 = random.random(); r2 = random.random()
                v3 = (w * v[base+3]
                      + c1 * r1 * (Pbest[i][base+3] - x[base+3])
                      + c2 * r2 * (Gbest[base+3] - x[base+3]))

                # Clamp velocities
                v0 = max(-vmax_one[0], min(vmax_one[0], v0))
                v1 = max(-vmax_one[1], min(vmax_one[1], v1))
                v2 = max(-vmax_one[2], min(vmax_one[2], v2))
                v3 = max(-vmax_one[3], min(vmax_one[3], v3))

                # Update pos
                x0 = clamp_param(x[base+0] + v0, speed_min, speed_max)
                x1 = wrap_angle_deg(x[base+1] + v1)
                x2 = clamp_param(x[base+2] + v2, t_release_min, t_release_max)
                x3 = clamp_param(x[base+3] + v3, t_fuse_min, t_fuse_max)

                # Save
                v[base+0], v[base+1], v[base+2], v[base+3] = v0, v1, v2, v3
                x[base+0], x[base+1], x[base+2], x[base+3] = x0, x1, x2, x3

            # Save back
            X[i] = x
            V[i] = v

            # Evaluate and update bests
            val = obj(x)
            if val > PbestVal[i]:
                Pbest[i] = x[:]
                PbestVal[i] = val
                if val > GbestVal:
                    GbestVal = val
                    Gbest = x[:]

    return tuple(Gbest), GbestVal


def main():
    parser = argparse.ArgumentParser(description="Compute or optimize obscuration time for M1 (new heading standard)")
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    # Subcommand: direct evaluation (single UAV)
    p_eval = subparsers.add_parser("eval", help="Direct evaluation with explicit parameters (single UAV)")
    p_eval.add_argument("--speed", type=float, required=True, help="UAV speed in m/s, must be within [70, 140]")
    p_eval.add_argument("--heading", type=float, required=True, help="UAV heading in degrees [0, 360], new standard: 0° along +x, CCW positive")
    p_eval.add_argument("--t_release", type=float, required=True, help="Release time after task received (s), >0")
    p_eval.add_argument("--t_fuse", type=float, required=True, help="Fuse time after release until explosion (s), >0")
    p_eval.add_argument("--dt", type=float, default=0.001, help="Simulation step size in seconds (default 0.001)")

    # Subcommand: optimization via PSO (single UAV)
    p_opt = subparsers.add_parser("opt", help="Optimize parameters using Particle Swarm Optimization (PSO) for single UAV")
    p_opt.add_argument("--iters", type=int, default=150, help="Number of PSO iterations")
    p_opt.add_argument("--swarm", type=int, default=25, help="Swarm size (number of particles)")
    p_opt.add_argument("--seed", type=int, default=42, help="Random seed")
    p_opt.add_argument("--speed_min", type=float, default=70.0)
    p_opt.add_argument("--speed_max", type=float, default=140.0)
    p_opt.add_argument("--heading_min", type=float, default=0.0)
    p_opt.add_argument("--heading_max", type=float, default=360.0)
    p_opt.add_argument("--t_release_max", type=float, default=3.0)
    p_opt.add_argument("--t_fuse_max", type=float, default=1.5)
    p_opt.add_argument("--w", type=float, default=0.7, help="PSO inertia weight")
    p_opt.add_argument("--c1", type=float, default=1.5, help="PSO cognitive coefficient")
    p_opt.add_argument("--c2", type=float, default=1.5, help="PSO social coefficient")
    p_opt.add_argument("--dt_coarse", type=float, default=0.03, help="Coarse dt for search (e.g., 0.02~0.04)")
    p_opt.add_argument("--refine_dt", type=float, default=0.001, help="Refinement dt for final evaluation (e.g., 0.001~0.002)")

    # New Subcommand: direct evaluation (three UAVs, each one smoke)
    p_eval3 = subparsers.add_parser("eval3", help="Direct evaluation with explicit parameters (three UAVs, one smoke each)")
    for i in range(1, 4):
        p_eval3.add_argument(f"--speed{i}", type=float, required=True, help=f"FY{i} speed in m/s, within [70, 140]")
        p_eval3.add_argument(f"--heading{i}", type=float, required=True, help=f"FY{i} heading in degrees [0, 360], new standard")
        p_eval3.add_argument(f"--t_release{i}", type=float, required=True, help=f"FY{i} release time after task received (s), >0")
        p_eval3.add_argument(f"--t_fuse{i}", type=float, required=True, help=f"FY{i} fuse time (s), >0")
    p_eval3.add_argument("--dt", type=float, default=0.001, help="Simulation step size in seconds (default 0.001)")

    # New Subcommand: optimization via PSO (three UAVs)
    p_opt3 = subparsers.add_parser("opt3", help="Optimize parameters for three UAVs (one smoke each) using PSO")
    p_opt3.add_argument("--iters", type=int, default=200, help="Number of PSO iterations")
    p_opt3.add_argument("--swarm", type=int, default=45, help="Swarm size (number of particles)")
    p_opt3.add_argument("--seed", type=int, default=42, help="Random seed")
    p_opt3.add_argument("--speed_min", type=float, default=70.0)
    p_opt3.add_argument("--speed_max", type=float, default=140.0)
    p_opt3.add_argument("--heading_min", type=float, default=0.0)
    p_opt3.add_argument("--heading_max", type=float, default=360.0)
    p_opt3.add_argument("--t_release_min", type=float, default=1e-3)
    p_opt3.add_argument("--t_release_max", type=float, default=60.0)
    p_opt3.add_argument("--t_fuse_min", type=float, default=1e-3)
    p_opt3.add_argument("--t_fuse_max", type=float, default=3.0)
    p_opt3.add_argument("--w", type=float, default=0.7, help="PSO inertia weight")
    p_opt3.add_argument("--c1", type=float, default=1.5, help="PSO cognitive coefficient")
    p_opt3.add_argument("--c2", type=float, default=1.5, help="PSO social coefficient")
    p_opt3.add_argument("--dt_coarse", type=float, default=0.03, help="Coarse dt for search (e.g., 0.03~0.05)")
    p_opt3.add_argument("--miss_penalty", type=float, default=25.0, help="若无遮蔽窗口则施加的惩罚")
    p_opt3.add_argument("--w_close", type=float, default=2.0, help="无遮蔽时，最近距离超出10m的惩罚权重")
    p_opt3.add_argument("--w_align", type=float, default=4.0, help="无遮蔽时，最近接近时刻与目标起始差的惩罚权重")
    p_opt3.add_argument("--w_te", type=float, default=3.0, help="起爆时刻与目标开始时刻(加偏移)对齐的惩罚权重")
    p_opt3.add_argument("--te_offset", type=float, default=0.62, help="经验偏移：t_enter≈te+offset，用于时间对齐引导")
    p_opt3.add_argument("--refine_dt", type=float, default=0.001, help="Refinement dt for final evaluation (e.g., 0.001~0.002)")
    p_opt3.add_argument("--t_start1", type=float, default=1.25, help="FY1 期望遮蔽开始时刻（秒）")
    p_opt3.add_argument("--t_start2", type=float, default=20.75, help="FY2 期望遮蔽开始时刻（秒）")
    p_opt3.add_argument("--t_start3", type=float, default=33.75, help="FY3 期望遮蔽开始时刻（秒）")
    p_opt3.add_argument("--d_target1", type=float, default=4.65, help="FY1 期望遮蔽持续时长（秒）")
    p_opt3.add_argument("--d_target2", type=float, default=3.90, help="FY2 期望遮蔽持续时长（秒）")
    p_opt3.add_argument("--d_target3", type=float, default=3.50, help="FY3 期望遮蔽持续时长（秒）")
    p_opt3.add_argument("--w_start", type=float, default=2.5, help="遮蔽开始时刻偏差权重")
    p_opt3.add_argument("--w_dur", type=float, default=3.0, help="遮蔽持续时长偏差权重")

    args = parser.parse_args()

    if args.cmd is None:
        # 默认执行三架无人机（FY1, FY2, FY3）各投放1枚的PSO优化（放宽释放与引信上限）
        params, best_val_coarse = pso_optimize_three(
            iters=220,
            swarm_size=50,
            seed=42,
            speed_min=70.0,
            speed_max=140.0,
            heading_min=0.0,
            heading_max=360.0,
            t_release_min=1e-3,
            t_release_max=60.0,
            t_fuse_min=1e-3,
            t_fuse_max=3.0,
            w=0.7, c1=1.5, c2=1.5,
            dt_coarse=0.04)
        s1, h1, tr1, tf1, s2, h2, tr2, tf2, s3, h3, tr3, tf3 = params
        final_time = effective_obscuration_time_for_M1_three(s1, h1, tr1, tf1,
                                                             s2, h2, tr2, tf2,
                                                             s3, h3, tr3, tf3,
                                                             dt=0.001)
        FY1 = (17800.0, 0.0, 1800.0)
        FY2 = (12000.0, 1400.0, 1400.0)
        FY3 = (6000.0, -3000.0, 700.0)
        (Pe1, te1, _), (Pe2, te2, _), (Pe3, te3, _) = (
            compute_explosion_point_generic_new_standard(FY1, s1, h1, tr1, tf1),
            compute_explosion_point_generic_new_standard(FY2, s2, h2, tr2, tf2),
            compute_explosion_point_generic_new_standard(FY3, s3, h3, tr3, tf3),
        )
        i1 = single_cloud_interval_M1(Pe1, te1, dt=0.001)
        i2 = single_cloud_interval_M1(Pe2, te2, dt=0.001)
        i3 = single_cloud_interval_M1(Pe3, te3, dt=0.001)

        print("默认执行三机一弹优化（PSO）用于对 M1 的遮蔽（放宽释放/引信上限）")
        print("无人机：FY1(17800,0,1800)、FY2(12000,1400,1400)、FY3(6000,-3000,700)")
        print("新方位角定义：0°沿+x，逆时针为正；速度 70~140 m/s；t_release∈(0,60]；t_fuse∈(0,3]")
        print("粗精度 dt = 0.04，PSO：iters = 220，swarm = 50，seed = 42")
        print("最优参数（粗评价下）：")
        print(f"  FY1: speed = {s1:.6f} m/s, heading = {h1:.6f} °, t_release = {tr1:.6f} s, t_fuse = {tf1:.6f} s")
        print(f"  FY2: speed = {s2:.6f} m/s, heading = {h2:.6f} °, t_release = {tr2:.6f} s, t_fuse = {tf2:.6f} s")
        print(f"  FY3: speed = {s3:.6f} m/s, heading = {h3:.6f} °, t_release = {tr3:.6f} s, t_fuse = {tf3:.6f} s")
        print(f"粗评价下的总有效遮蔽时长（并集）~= {best_val_coarse:.6f} s")
        print("细评估（refine）结果：")
        print(f"  FY1: 起爆时刻 = {te1:.6f} s，起爆点 Pe1 = ({Pe1[0]:.3f}, {Pe1[1]:.3f}, {Pe1[2]:.3f})")
        print(f"  FY2: 起爆时刻 = {te2:.6f} s，起爆点 Pe2 = ({Pe2[0]:.3f}, {Pe2[1]:.3f}, {Pe2[2]:.3f})")
        print(f"  FY3: 起爆时刻 = {te3:.6f} s，起爆点 Pe3 = ({Pe3[0]:.3f}, {Pe3[1]:.3f}, {Pe3[2]:.3f})")
        def fmt_interval(tag, interval):
            if interval is None:
                return f"  {tag}: 无遮蔽窗口"
            t_enter, t_exit, dur = interval
            return f"  {tag}: 遮蔽开始≈{t_enter:.3f} s，持续≈{dur:.3f} s（至 {t_exit:.3f} s）"
        print(fmt_interval("FY1", i1))
        print(fmt_interval("FY2", i2))
        print(fmt_interval("FY3", i3))
        print(f"  最终有效遮蔽总时长（对 M1，三团并集）= {final_time:.6f} s（dt=0.001）")
        return

    if args.cmd == "eval":
        # Validate ranges
        if not (70.0 <= args.speed <= 140.0):
            raise ValueError(f"speed must be within [70, 140], got {args.speed}")
        if not (0.0 <= args.heading <= 360.0):
            raise ValueError(f"heading must be within [0, 360], got {args.heading}")
        if not (args.t_release > 0.0):
            raise ValueError(f"t_release must be > 0, got {args.t_release}")
        if not (args.t_fuse > 0.0):
            raise ValueError(f"t_fuse must be > 0, got {args.t_fuse}")

        Pe, t_explosion, _ = compute_explosion_point_FY1_new_standard(speed=args.speed,
                                                                      heading_deg=args.heading,
                                                                      t_release=args.t_release,
                                                                      t_fuse=args.t_fuse)
        total_time = effective_obscuration_time_for_M1(speed=args.speed,
                                                       heading_deg=args.heading,
                                                       t_release=args.t_release,
                                                       t_fuse=args.t_fuse,
                                                       dt=args.dt)

        print("采用新方位角定义：0°沿+x，逆时针为正（0~360°）")
        print(f"FY1 航向 = {args.heading:.2f}°（新标准），速度 = {args.speed:.2f} m/s")
        print(f"释放时刻 t_release = {args.t_release:.3f} s，起爆时刻 t_explosion = {t_explosion:.3f} s")
        print(f"起爆点 Pe = ({Pe[0]:.3f}, {Pe[1]:.3f}, {Pe[2]:.3f})")
        print("云团中心以 3 m/s 匀速下沉，有效期 20 s，判据：距离[M1(t), T]线段 ≤ 10 m")
        print(f"对 M1 的有效遮蔽总时长 = {total_time:.6f} s")

    elif args.cmd == "opt":
        params, best_val_coarse = pso_optimize(iters=args.iters,
                                               swarm_size=args.swarm,
                                               seed=args.seed,
                                               speed_min=args.speed_min,
                                               speed_max=args.speed_max,
                                               heading_min=args.heading_min,
                                               heading_max=args.heading_max,
                                               t_release_min=1e-3,
                                               t_release_max=args.t_release_max,
                                               t_fuse_min=1e-3,
                                               t_fuse_max=args.t_fuse_max,
                                               w=args.w, c1=args.c1, c2=args.c2,
                                               dt_coarse=args.dt_coarse)
        s, h, tr, tf = params
        # Refine with smaller dt for accurate final time
        final_time = effective_obscuration_time_for_M1(s, h, tr, tf, dt=args.refine_dt)
        Pe, t_explosion, _ = compute_explosion_point_FY1_new_standard(s, h, tr, tf)

        print("优化完成（粒子群算法 PSO，单机）")
        print(f"参数范围：speed∈[{args.speed_min},{args.speed_max}], heading∈[{args.heading_min},{args.heading_max}], t_release∈(0,{args.t_release_max}], t_fuse∈(0,{args.t_fuse_max}]")
        print(f"粗精度 dt = {args.dt_coarse}，PSO：iters = {args.iters}，swarm = {args.swarm}，seed = {args.seed}")
        print("最优参数（粗评价下）：")
        print(f"  speed = {s:.6f} m/s, heading = {h:.6f} °, t_release = {tr:.6f} s, t_fuse = {tf:.6f} s")
        print(f"粗评价下的遮蔽时长 ~= {best_val_coarse:.6f} s")
        print("细评估（refine）结果：")
        print(f"  起爆时刻 = {t_explosion:.6f} s，起爆点 Pe = ({Pe[0]:.3f}, {Pe[1]:.3f}, {Pe[2]:.3f})")
        print(f"  最终有效遮蔽总时长 = {final_time:.6f} s（dt={args.refine_dt}）")

    elif args.cmd == "eval3":
        # Validate ranges for three UAVs
        vals = []
        for i in range(1, 4):
            s = getattr(args, f"speed{i}")
            h = getattr(args, f"heading{i}")
            tr = getattr(args, f"t_release{i}")
            tf = getattr(args, f"t_fuse{i}")
            if not (70.0 <= s <= 140.0):
                raise ValueError(f"speed{i} must be within [70, 140], got {s}")
            if not (0.0 <= h <= 360.0):
                raise ValueError(f"heading{i} must be within [0, 360], got {h}")
            if not (tr > 0.0):
                raise ValueError(f"t_release{i} must be > 0, got {tr}")
            if not (tf > 0.0):
                raise ValueError(f"t_fuse{i} must be > 0, got {tf}")
            vals.extend([s, h, tr, tf])

        s1, h1, tr1, tf1, s2, h2, tr2, tf2, s3, h3, tr3, tf3 = vals
        total_time = effective_obscuration_time_for_M1_three(s1, h1, tr1, tf1,
                                                             s2, h2, tr2, tf2,
                                                             s3, h3, tr3, tf3,
                                                             dt=args.dt)
        FY1 = (17800.0, 0.0, 1800.0)
        FY2 = (12000.0, 1400.0, 1400.0)
        FY3 = (6000.0, -3000.0, 700.0)
        (Pe1, te1, _), (Pe2, te2, _), (Pe3, te3, _) = (
            compute_explosion_point_generic_new_standard(FY1, s1, h1, tr1, tf1),
            compute_explosion_point_generic_new_standard(FY2, s2, h2, tr2, tf2),
            compute_explosion_point_generic_new_standard(FY3, s3, h3, tr3, tf3),
        )
        print("三机一弹评估（对 M1）——新方位角定义：0°沿+x，逆时针为正（0~360°）")
        print(f"FY1: speed={s1:.2f}, heading={h1:.2f}°, t_release={tr1:.3f}s, t_fuse={tf1:.3f}s, te={te1:.3f}s, Pe=({Pe1[0]:.1f},{Pe1[1]:.1f},{Pe1[2]:.1f})")
        print(f"FY2: speed={s2:.2f}, heading={h2:.2f}°, t_release={tr2:.3f}s, t_fuse={tf2:.3f}s, te={te2:.3f}s, Pe=({Pe2[0]:.1f},{Pe2[1]:.1f},{Pe2[2]:.1f})")
        print(f"FY3: speed={s3:.2f}, heading={h3:.2f}°, t_release={tr3:.3f}s, t_fuse={tf3:.3f}s, te={te3:.3f}s, Pe=({Pe3[0]:.1f},{Pe3[1]:.1f},{Pe3[2]:.1f})")
        print("判据：云团中心到[M1(t), T]线段距离≤10 m；每团有效期20 s；云团以 3 m/s 下沉")
        print(f"对 M1 的总有效遮蔽时长（并集）= {total_time:.6f} s（dt={args.dt}）")

    elif args.cmd == "opt3":
        params, best_val_coarse = pso_optimize_three(args.iters,
                                                     args.swarm,
                                                     args.seed,
                                                     args.speed_min,
                                                     args.speed_max,
                                                     args.heading_min,
                                                     args.heading_max,
                                                     args.t_release_min,
                                                     args.t_release_max,
                                                     args.t_fuse_min,
                                                     args.t_fuse_max,
                                                     args.w, args.c1, args.c2,
                                                     args.dt_coarse,
                                                     args.t_start1, args.t_start2, args.t_start3,
                                                     args.d_target1, args.d_target2, args.d_target3,
                                                     args.w_start, args.w_dur, args.miss_penalty,
                                                     args.w_close, args.w_align, args.w_te, args.te_offset)
        s1, h1, tr1, tf1, s2, h2, tr2, tf2, s3, h3, tr3, tf3 = params
        final_time = effective_obscuration_time_for_M1_three(s1, h1, tr1, tf1,
                                                             s2, h2, tr2, tf2,
                                                             s3, h3, tr3, tf3,
                                                             dt=args.refine_dt)
        FY1 = (17800.0, 0.0, 1800.0)
        FY2 = (12000.0, 1400.0, 1400.0)
        FY3 = (6000.0, -3000.0, 700.0)
        (Pe1, te1, _), (Pe2, te2, _), (Pe3, te3, _) = (
            compute_explosion_point_generic_new_standard(FY1, s1, h1, tr1, tf1),
            compute_explosion_point_generic_new_standard(FY2, s2, h2, tr2, tf2),
            compute_explosion_point_generic_new_standard(FY3, s3, h3, tr3, tf3),
        )
        print("优化完成（粒子群算法 PSO，三机一弹，对 M1）")
        print(f"参数范围：speed∈[{args.speed_min},{args.speed_max}], heading∈[{args.heading_min},{args.heading_max}], t_release∈(0,{args.t_release_max}], t_fuse∈(0,{args.t_fuse_max}]")
        print(f"粗精度 dt = {args.dt_coarse}，PSO：iters = {args.iters}，swarm = {args.swarm}，seed = {args.seed}")
        print("最优参数（粗评价下）：")
        print(f"  FY1: speed = {s1:.6f} m/s, heading = {h1:.6f} °, t_release = {tr1:.6f} s, t_fuse = {tf1:.6f} s")
        print(f"  FY2: speed = {s2:.6f} m/s, heading = {h2:.6f} °, t_release = {tr2:.6f} s, t_fuse = {tf2:.6f} s")
        print(f"  FY3: speed = {s3:.6f} m/s, heading = {h3:.6f} °, t_release = {tr3:.6f} s, t_fuse = {tf3:.6f} s")
        print(f"粗评价下的总有效遮蔽时长（并集）~= {best_val_coarse:.6f} s")
        print("细评估（refine）结果：")
        print(f"  FY1: 起爆时刻 = {te1:.6f} s，起爆点 Pe1 = ({Pe1[0]:.3f}, {Pe1[1]:.3f}, {Pe1[2]:.3f})")
        print(f"  FY2: 起爆时刻 = {te2:.6f} s，起爆点 Pe2 = ({Pe2[0]:.3f}, {Pe2[1]:.3f}, {Pe2[2]:.3f})")
        print(f"  FY3: 起爆时刻 = {te3:.6f} s，起爆点 Pe3 = ({Pe3[0]:.3f}, {Pe3[1]:.3f}, {Pe3[2]:.3f})")
        # 计算各团遮蔽窗口（细评估步长）
        i1 = single_cloud_interval_M1(Pe1, te1, dt=args.refine_dt)
        i2 = single_cloud_interval_M1(Pe2, te2, dt=args.refine_dt)
        i3 = single_cloud_interval_M1(Pe3, te3, dt=args.refine_dt)
        def fmt_interval2(tag, interval):
            if interval is None:
                return f"  {tag}: 无遮蔽窗口"
            t_enter, t_exit, dur = interval
            return f"  {tag}: 遮蔽开始≈{t_enter:.3f} s，持续≈{dur:.3f} s（至 {t_exit:.3f} s）"
        print(fmt_interval2("FY1", i1))
        print(fmt_interval2("FY2", i2))
        print(fmt_interval2("FY3", i3))
        print(f"  最终有效遮蔽总时长（对 M1，三团并集）= {final_time:.6f} s（dt={args.refine_dt}）")


if __name__ == "__main__":
    main()