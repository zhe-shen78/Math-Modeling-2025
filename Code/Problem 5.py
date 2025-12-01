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

# ---------------- Added for multi-UAV & multi-missile (Problem 5) ----------------

def missile_state(idx: int, t: float) -> Vec3:
    """Generalized missile state for M1/M2/M3 at time t [s], speed 300 m/s towards origin."""
    if idx == 1:
        M0 = (20000.0, 0.0, 2000.0)
    elif idx == 2:
        M0 = (19000.0, 600.0, 2100.0)
    elif idx == 3:
        M0 = (18000.0, -600.0, 1900.0)
    else:
        raise ValueError("missile idx must be 1,2,3")
    v_dir = (-M0[0], -M0[1], -M0[2])
    L = math.sqrt(v_dir[0]**2 + v_dir[1]**2 + v_dir[2]**2)
    u = (v_dir[0]/L, v_dir[1]/L, v_dir[2]/L)
    v = (u[0]*300.0, u[1]*300.0, u[2]*300.0)
    return (M0[0] + v[0]*t, M0[1] + v[1]*t, M0[2] + v[2]*t)


def compute_explosion_point_generic_new_standard(FY: Vec3,
                                                 speed: float,
                                                 heading_deg: float,
                                                 t_release: float,
                                                 t_fuse: float,
                                                 g: float = 9.8) -> Tuple[Vec3, float]:
    """Compute explosion point for a UAV starting at FY using new heading standard.
    Returns (Pe, t_explosion)."""
    th = math.radians(heading_deg)
    ux, uy = math.cos(th), math.sin(th)
    vx = speed * ux
    vy = speed * uy
    # Release point (constant altitude prior to release)
    Rz = FY[2]
    Rx = FY[0] + vx*t_release
    Ry = FY[1] + vy*t_release
    # Free fall after release for t_fuse seconds
    x_e = Rx + vx*t_fuse
    y_e = Ry + vy*t_fuse
    z_e = Rz - 0.5*g*(t_fuse**2)
    Pe = (x_e, y_e, z_e)
    t_explosion = t_release + t_fuse
    return Pe, t_explosion


def effective_obscuration_time_for_missile_with_params(FY: Vec3,
                                                       speed: float,
                                                       heading_deg: float,
                                                       t_release: float,
                                                       t_fuse: float,
                                                       missile_idx: int,
                                                       dt: float = 0.02) -> float:
    """Effective obscuration time (seconds) for a single bomb against missile_idx.
    Uses same criterion: distance from cloud center to segment [M_i(t), T] <= 10m within 20s after explosion."""
    T = (0.0, 200.0, 0.0)
    Pe, te = compute_explosion_point_generic_new_standard(FY, speed, heading_deg, t_release, t_fuse)
    t0 = te
    t1 = te + 20.0
    R_eff = 10.0
    t = t0
    total = 0.0
    prev_d = None
    while t <= t1 + 1e-12:
        C = cloud_center_after(Pe, t, te, sink_speed=3.0)
        M = missile_state(missile_idx, t)
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


def cloud_intervals_for_missile(Pe: Vec3, te: float, missile_idx: int, dt: float = 0.001, R_eff: float = 10.0):
    """Return list of [t_enter, t_exit] intervals within [te, te+20] where distance<=R_eff against missile_idx."""
    T = (0.0, 200.0, 0.0)
    t0 = te
    t1 = te + 20.0
    intervals = []
    inside = False
    t = t0
    prev_d = None
    t_enter = None
    while t <= t1 + 1e-12:
        C = cloud_center_after(Pe, t, te, sink_speed=3.0)
        M = missile_state(missile_idx, t)
        d = dist_point_to_segment(C, M, T)
        if prev_d is None:
            prev_d = d
            inside = (d <= R_eff)
            if inside:
                t_enter = t
        else:
            if (prev_d > R_eff and d <= R_eff):
                # entering
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    t_enter = t - dt + max(0.0, min(1.0, alpha)) * dt
                else:
                    t_enter = t
                inside = True
            elif (prev_d <= R_eff and d > R_eff) and inside:
                # exiting
                denom = (d - prev_d)
                if abs(denom) > 1e-12:
                    alpha = (R_eff - prev_d) / denom
                    t_exit = t - dt + max(0.0, min(1.0, alpha)) * dt
                else:
                    t_exit = t
                if t_enter is None:
                    t_enter = t - dt
                intervals.append([max(t0, t_enter), min(t1, t_exit)])
                inside = False
                t_enter = None
        prev_d = d
        t += dt
    if inside and t_enter is not None:
        intervals.append([max(t0, t_enter), t1])
    return intervals


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0][:]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def intervals_total_length(intervals) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)

# 补充角度工具函数，供 PSO 使用

def wrap_angle_deg(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a


def shortest_ang_diff_deg(target: float, current: float) -> float:
    d = (target - current + 180.0) % 360.0 - 180.0
    return d

# 为最优指派准备：给定 (FY, s,h,tr,tf) 构造三发弹及其区间

def build_bombs_and_intervals_for_params(FY: Vec3,
                                           s: float,
                                           h: float,
                                           tr: float,
                                           tf: float,
                                           missile_idx: int,
                                           refine_dt: float):
     base_tr = max(1e-3, min(60.0, tr))
     tr_list = [base_tr, min(60.0, base_tr + 1.0), min(60.0, base_tr + 2.0)]
     bombs = []
     all_ivs = []
     ivs_by_bomb = []
     th = math.radians(h)
     vx, vy = s*math.cos(th), s*math.sin(th)
     for trk in tr_list:
         Pe, te = compute_explosion_point_generic_new_standard(FY, s, h, trk, tf)
         R = (FY[0] + vx*trk, FY[1] + vy*trk, FY[2])
         bombs.append((trk, tf, R, Pe, te))
         ivs = cloud_intervals_for_missile(Pe, te, missile_idx, dt=refine_dt)
         ivs_by_bomb.append(ivs)
         all_ivs.extend(ivs)
     return bombs, all_ivs, ivs_by_bomb


def best_for_uav_missile(FY: Vec3,
                           missile_idx: int,
                           iters: int,
                           swarm: int,
                           seed_base: int,
                           dt_coarse: float,
                           refine_dt: float,
                           restarts: int):
     best_len = -1.0
     best_params = None
     best_bombs = None
     best_ivs = None
     best_ivs_by_bomb = None
     for r in range(restarts):
         seed_r = seed_base + 97*r + 31*missile_idx
         params, _ = pso_optimize_single_for_FY_missile(FY, missile_idx, iters=iters, swarm_size=swarm, seed=seed_r, dt_coarse=dt_coarse)
         s, h, tr, tf = params
         bombs, ivs_all, ivs_by_bomb = build_bombs_and_intervals_for_params(FY, s, h, tr, tf, missile_idx, refine_dt)
         merged_len = intervals_total_length(merge_intervals(ivs_all))
         if merged_len > best_len:
             best_len = merged_len
             best_params = params
             best_bombs = bombs
             best_ivs = ivs_all
             best_ivs_by_bomb = ivs_by_bomb
     return best_params, best_bombs, best_ivs, best_ivs_by_bomb, best_len


def pso_optimize_single_for_FY_missile(FY: Vec3,
                                       missile_idx: int,
                                       iters: int = 120,
                                       swarm_size: int = 40,
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
                                       dt_coarse: float = 0.02) -> Tuple[Tuple[float, float, float, float], float]:
    """PSO optimize a single bomb against a specified missile for given FY."""
    random.seed(seed)
    rng_speed = speed_max - speed_min
    rng_heading = heading_max - heading_min
    rng_tr = t_release_max - t_release_min
    rng_tf = t_fuse_max - t_fuse_min
    vmax = [rng_speed * 0.25, 90.0, rng_tr * 0.25, rng_tf * 0.25]

    def clamp_param(x, lo, hi):
        return max(lo, min(hi, x))

    def obj(xx):
        return effective_obscuration_time_for_missile_with_params(FY, xx[0], xx[1], xx[2], xx[3], missile_idx, dt=dt_coarse)

    X = []
    V = []
    Pbest = []
    PbestVal = []
    Gbest = None
    GbestVal = -1e18

    for _ in range(swarm_size):
        s = random.uniform(speed_min, speed_max)
        h = random.uniform(heading_min, heading_max)
        tr = random.uniform(t_release_min, t_release_max)
        tf = random.uniform(t_fuse_min, t_fuse_max)
        x = [s, h, tr, tf]
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

    for _ in range(iters):
        for i in range(swarm_size):
            x = X[i]
            v = V[i]
            r1 = random.random(); r2 = random.random()
            v0 = (w * v[0] + c1 * r1 * (Pbest[i][0] - x[0]) + c2 * r2 * (Gbest[0] - x[0]))
            r1 = random.random(); r2 = random.random()
            diff_p = shortest_ang_diff_deg(Pbest[i][1], x[1])
            diff_g = shortest_ang_diff_deg(Gbest[1], x[1])
            v1 = (w * v[1] + c1 * r1 * diff_p + c2 * r2 * diff_g)
            r1 = random.random(); r2 = random.random()
            v2 = (w * v[2] + c1 * r1 * (Pbest[i][2] - x[2]) + c2 * r2 * (Gbest[2] - x[2]))
            r1 = random.random(); r2 = random.random()
            v3 = (w * v[3] + c1 * r1 * (Pbest[i][3] - x[3]) + c2 * r2 * (Gbest[3] - x[3]))
            v0 = max(-vmax[0], min(vmax[0], v0))
            v1 = max(-vmax[1], min(vmax[1], v1))
            v2 = max(-vmax[2], min(vmax[2], v2))
            v3 = max(-vmax[3], min(vmax[3], v3))
            x0 = clamp_param(x[0] + v0, speed_min, speed_max)
            x1 = wrap_angle_deg(x[1] + v1)
            x2 = clamp_param(x[2] + v2, t_release_min, t_release_max)
            x3 = clamp_param(x[3] + v3, t_fuse_min, t_fuse_max)
            x = [x0, x1, x2, x3]
            X[i] = x
            V[i] = [v0, v1, v2, v3]
            val = obj(x)
            if val > PbestVal[i]:
                Pbest[i] = x[:]
                PbestVal[i] = val
                if val > GbestVal:
                    GbestVal = val
                    Gbest = x[:]
    return (Gbest[0], Gbest[1], Gbest[2], Gbest[3]), GbestVal


def assign_uavs_to_missiles(FY_list: Tuple[Vec3, ...]):
    """Greedy assignment: assign each UAV to the nearest missile (1,2,3)."""
    missiles = [(20000.0, 0.0, 2000.0), (19000.0, 600.0, 2100.0), (18000.0, -600.0, 1900.0)]
    assign = []
    for FY in FY_list:
        dists = [math.sqrt((FY[0]-M[0])**2 + (FY[1]-M[1])**2 + (FY[2]-M[2])**2) for M in missiles]
        idx = 1 + dists.index(min(dists))
        assign.append(idx)
    return assign


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


# --- Wrapper functions for M1-only evaluation and optimization (used by subcommands eval/opt) ---

def effective_obscuration_time_for_M1(speed: float,
                                       heading_deg: float,
                                       t_release: float,
                                       t_fuse: float,
                                       dt: float = 0.001) -> float:
    """Wrapper: evaluate obscuration time for FY1 against M1 using existing generic routine."""
    FY1 = (17800.0, 0.0, 1800.0)
    return effective_obscuration_time_for_missile_with_params(
        FY1, speed, heading_deg, t_release, t_fuse, missile_idx=1, dt=dt
    )


def pso_optimize(iters: int,
                 swarm_size: int,
                 seed: int,
                 speed_min: float,
                 speed_max: float,
                 heading_min: float,
                 heading_max: float,
                 t_release_min: float,
                 t_release_max: float,
                 t_fuse_min: float,
                 t_fuse_max: float,
                 w: float,
                 c1: float,
                 c2: float,
                 dt_coarse: float):
    """Wrapper: run PSO for FY1 against M1 and return (best_params, best_value)."""
    FY1 = (17800.0, 0.0, 1800.0)
    params, best_val = pso_optimize_single_for_FY_missile(
        FY1,
        missile_idx=1,
        iters=iters,
        swarm_size=swarm_size,
        seed=seed,
        speed_min=speed_min,
        speed_max=speed_max,
        heading_min=heading_min,
        heading_max=heading_max,
        t_release_min=t_release_min,
        t_release_max=t_release_max,
        t_fuse_min=t_fuse_min,
        t_fuse_max=t_fuse_max,
        w=w,
        c1=c1,
        c2=c2,
        dt_coarse=dt_coarse,
    )
    return params, best_val


def main():
    parser = argparse.ArgumentParser(description="Compute or optimize obscuration time for M1 (new heading standard)")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Subcommand: direct evaluation
    p_eval = subparsers.add_parser("eval", help="Direct evaluation with explicit parameters")
    p_eval.add_argument("--speed", type=float, required=True, help="UAV speed in m/s, must be within [70, 140]")
    p_eval.add_argument("--heading", type=float, required=True, help="UAV heading in degrees [0, 360], new standard: 0° along +x, CCW positive")
    p_eval.add_argument("--t_release", type=float, required=True, help="Release time after task received (s), >0")
    p_eval.add_argument("--t_fuse", type=float, required=True, help="Fuse time after release until explosion (s), >0")
    p_eval.add_argument("--dt", type=float, default=0.001, help="Simulation step size in seconds (default 0.001)")

    # Subcommand: optimization via PSO
    p_opt = subparsers.add_parser("opt", help="Optimize parameters using Particle Swarm Optimization (PSO)")
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
    p_opt.add_argument("--dt_coarse", type=float, default=0.03, help="Coarse dt for search (e.g., 0.02~0.05)")
    p_opt.add_argument("--refine_dt", type=float, default=0.001, help="Refinement dt for final evaluation (e.g., 0.001~0.002)")

    # 新增子命令：opt5（五架无人机、每架至多三枚弹、多导弹）
    p_opt5 = subparsers.add_parser("opt5", help="五机三弹策略优化（最优指派+多随机重启）")
    p_opt5.add_argument("--iters", type=int, default=160, help="每台无人机PSO迭代次数")
    p_opt5.add_argument("--swarm", type=int, default=48, help="每台无人机PSO粒子数")
    p_opt5.add_argument("--seed", type=int, default=42, help="随机种子基准")
    p_opt5.add_argument("--dt_coarse", type=float, default=0.02, help="粗评估步长，PSO用")
    p_opt5.add_argument("--refine_dt", type=float, default=0.001, help="细评估步长，用于最终统计")
    p_opt5.add_argument("--restarts", type=int, default=6, help="每个(无人机,导弹)的随机重启次数")

    args = parser.parse_args()

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
        print(f"云团中心以 3 m/s 匀速下沉，有效期 20 s，判据：距离[M1(t), T]线段 ≤ 10 m")
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

        print("优化完成（粒子群算法 PSO）")
        print(f"参数范围：speed∈[{args.speed_min},{args.speed_max}], heading∈[{args.heading_min},{args.heading_max}], t_release∈(0,{args.t_release_max}], t_fuse∈(0,{args.t_fuse_max}]")
        print(f"粗精度 dt = {args.dt_coarse}，PSO：iters = {args.iters}，swarm = {args.swarm}，seed = {args.seed}")
        print("最优参数（粗评价下）：")
        print(f"  speed = {s:.6f} m/s, heading = {h:.6f} °, t_release = {tr:.6f} s, t_fuse = {tf:.6f} s")
        print(f"粗评价下的遮蔽时长 ~= {best_val_coarse:.6f} s")
        print("细评估（refine）结果：")
        print(f"  起爆时刻 = {t_explosion:.6f} s，起爆点 Pe = ({Pe[0]:.3f}, {Pe[1]:.3f}, {Pe[2]:.3f})")
        print(f"  最终有效遮蔽总时长 = {final_time:.6f} s（dt={args.refine_dt}）")

    # ---------------- New subcommand: opt5 (five UAVs, up to 3 bombs each, 3 missiles) ----------------
    elif args.cmd == "opt5":
        # Inputs and defaults
        iters = getattr(args, 'iters', 160)
        swarm = getattr(args, 'swarm', 48)
        seed = getattr(args, 'seed', 42)
        dt_coarse = getattr(args, 'dt_coarse', 0.02)
        refine_dt = getattr(args, 'refine_dt', 0.001)
        restarts = getattr(args, 'restarts', 6)

        FYs = (
            (17800.0, 0.0, 1800.0),
            (12000.0, 1400.0, 1400.0),
            (6000.0, -3000.0, 700.0),
            (11000.0, 2000.0, 1800.0),
            (13000.0, -2000.0, 1300.0),
        )
        print("五架无人机初始位置：FY1(17800,0,1800) FY2(12000,1400,1400) FY3(6000,-3000,700) FY4(11000,2000,1800) FY5(13000,-2000,1300)")
        print("导弹初始：M1(20000,0,2000) M2(19000,600,2100) M3(18000,-600,1900)；目标 T=(0,200,0)")
        print(f"采用最优指派（枚举 UAV->(M1,M2,M3) 分布），每个配对多随机重启 {restarts} 次，PSO iters={iters}, swarm={swarm}, seed_base={seed}")
        random.seed(seed)

        # 预优化：对每个 (UAV, Missile) 做多重启 PSO，记录最优方案与区间（含按弹拆分）
        pair_best = {}  # key:(i,m) -> dict(params, bombs, ivs_all, ivs_by_bomb, len)
        for i, FY in enumerate(FYs, start=1):
            for m_idx in (1,2,3):
                params, bombs, ivs_all, ivs_by_bomb, length = best_for_uav_missile(FY, m_idx, iters, swarm, seed + i*13, dt_coarse, refine_dt, restarts)
                pair_best[(i,m_idx)] = {
                    'params': params,
                    'bombs': bombs,
                    'ivs_all': ivs_all,
                    'ivs_by_bomb': ivs_by_bomb,
                    'len': length,
                    'FY': FY,
                }

        # 枚举每台无人机把3发弹如何分配到(M1,M2,M3)：所有非负整数组合 a1+a2+a3=3，共C(5,2)=10种
        def all_distributions_of_three():
            res = []
            for a1 in range(0,4):
                for a2 in range(0,4-a1):
                    a3 = 3 - a1 - a2
                    res.append((a1,a2,a3))
            return res
        dist_list = all_distributions_of_three()
        print(f"每台无人机分配方案数: {len(dist_list)}，总枚举约 {len(dist_list)**5} 种")

        from itertools import product
        best_total = -1.0
        best_combo = None
        best_per_m_total = None
        # 同时记录用于输出的具体弹参数选择
        best_choice_detail = None

        for combo in product(dist_list, repeat=5):
            # combo[i-1] = (a1,a2,a3) for UAV i
            per_m_intervals = {1: [], 2: [], 3: []}
            choice_detail = {i: {1:0,2:0,3:0} for i in range(1,6)}  # 记录每台无人机给每枚导弹的发弹数
            for i in range(1,6):
                a1,a2,a3 = combo[i-1]
                # 取该无人机对各导弹最优方案的前 a_m 发
                if a1>0:
                    ivs_by_bomb = pair_best[(i,1)]['ivs_by_bomb']
                    for k in range(min(a1, len(ivs_by_bomb))):
                        per_m_intervals[1].extend(ivs_by_bomb[k])
                    choice_detail[i][1] = a1
                if a2>0:
                    ivs_by_bomb = pair_best[(i,2)]['ivs_by_bomb']
                    for k in range(min(a2, len(ivs_by_bomb))):
                        per_m_intervals[2].extend(ivs_by_bomb[k])
                    choice_detail[i][2] = a2
                if a3>0:
                    ivs_by_bomb = pair_best[(i,3)]['ivs_by_bomb']
                    for k in range(min(a3, len(ivs_by_bomb))):
                        per_m_intervals[3].extend(ivs_by_bomb[k])
                    choice_detail[i][3] = a3
            total = 0.0
            per_m_total = {}
            for m in (1,2,3):
                merged = merge_intervals(per_m_intervals[m])
                per_m_total[m] = intervals_total_length(merged)
                total += per_m_total[m]
            if total > best_total:
                best_total = total
                best_combo = combo
                best_per_m_total = per_m_total
                best_choice_detail = choice_detail

        # 输出最优分配方案
        print("最优分配（每台无人机 -> (M1,M2,M3) 发弹数）：")
        for i in range(1,6):
            a1,a2,a3 = best_combo[i-1]
            print(f"  FY{i}: (M1:{a1}, M2:{a2}, M3:{a3})")

        # 汇总输出每台无人机针对其各导弹的参数（采用各导弹的单独最优参数），以及对应三发弹的R/Pe（按需要的发弹数截取）
        for i, FY in enumerate(FYs, start=1):
            a1,a2,a3 = best_combo[i-1]
            for m_idx, ai in ((1,a1),(2,a2),(3,a3)):
                if ai<=0:
                    continue
                item = pair_best[(i, m_idx)]
                s, h, tr, tf = item['params']
                bombs = item['bombs']
                print(f"FY{i} -> M{m_idx}: speed={s:.3f} m/s, heading={h:.3f}°，选用前 {ai} 发：")
                for j in range(1, ai+1):
                    trk, tfk, Rk, Pek, tek = bombs[j-1]
                    print(f"  弹{j}: t_release={trk:.3f}s, t_fuse={tfk:.3f}s, te={tek:.3f}s")
                    print(f"       R=({Rk[0]:.3f}, {Rk[1]:.3f}, {Rk[2]:.3f}), Pe=({Pek[0]:.3f}, {Pek[1]:.3f}, {Pek[2]:.3f})")

        print("—— 每枚导弹的并集有效遮蔽时长（refine）——")
        print(f"M1: {best_per_m_total[1]:.6f} s")
        print(f"M2: {best_per_m_total[2]:.6f} s")
        print(f"M3: {best_per_m_total[3]:.6f} s")
        print(f"总和: {best_total:.6f} s")


if __name__ == "__main__":
    main()