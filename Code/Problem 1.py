#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math

class SmokeInterferenceAnalyzer:
    def __init__(self):
        # 初始化所有物体的参数
        self.missile_init_pos = np.array([20000, 0, 2000])  # 导弹M1初始位置 (m)
        self.drone_init_pos = np.array([17800, 0, 1800])    # 无人机FY1初始位置 (m)
        self.target_center = np.array([0, 200, 0])          # 真目标下底面圆心 (m)
        
        # 运动参数
        self.missile_speed = 300  # 导弹速度 (m/s)
        self.drone_speed = 120    # 无人机速度 (m/s)
        
        # 目标参数
        self.target_radius = 7    # 目标半径 (m)
        self.target_height = 10   # 目标高度 (m)
        
        # 烟幕弹参数
        self.drop_delay = 1.5     # 投放延迟 (s)
        self.explode_delay = 3.6  # 起爆延迟 (s)
        self.cloud_sink_speed = 3 # 云团下沉速度 (m/s)
        self.effective_radius = 10 # 有效遮蔽半径 (m)
        self.effective_duration = 20 # 有效遮蔽时长 (s)
        
        # 重力加速度
        self.gravity = 9.8        # (m/s²)
        
    def calculate_motion_vectors(self):
        """计算各物体的运动方向向量"""
        # 导弹朝向原点的单位向量
        missile_direction = -self.missile_init_pos / np.linalg.norm(self.missile_init_pos)
        
        # 无人机朝向原点的水平方向单位向量
        drone_horizontal_direction = np.array([-self.drone_init_pos[0], -self.drone_init_pos[1], 0])
        drone_horizontal_direction = drone_horizontal_direction / np.linalg.norm(drone_horizontal_direction)
        
        return missile_direction, drone_horizontal_direction
    
    def missile_position(self, t):
        """计算导弹在时间t的位置"""
        missile_direction, _ = self.calculate_motion_vectors()
        return self.missile_init_pos + missile_direction * self.missile_speed * t
    
    def drone_position(self, t):
        """计算无人机在时间t的位置"""
        _, drone_direction = self.calculate_motion_vectors()
        return self.drone_init_pos + drone_direction * self.drone_speed * t
    
    def smoke_bomb_trajectory(self, t, drop_time):
        """计算烟幕弹的运动轨迹"""
        if t < drop_time:
            # 烟幕弹还未投放，跟随无人机
            return self.drone_position(t)
        
        # 烟幕弹投放时的位置和速度
        drop_pos = self.drone_position(drop_time)
        _, drone_direction = self.calculate_motion_vectors()
        initial_velocity = drone_direction * self.drone_speed
        
        # 投放后的时间
        dt = t - drop_time
        
        # 在重力作用下的抛物运动
        position = drop_pos + initial_velocity * dt + 0.5 * np.array([0, 0, -self.gravity]) * dt**2
        return position
    
    def smoke_cloud_center(self, t, drop_time, explode_time):
        """计算烟幕云团中心的位置"""
        if t < explode_time:
            # 还未起爆，返回烟幕弹位置
            return self.smoke_bomb_trajectory(t, drop_time)
        
        # 起爆时的位置
        explode_pos = self.smoke_bomb_trajectory(explode_time, drop_time)
        
        # 起爆后的时间
        dt = t - explode_time
        
        # 云团以固定速度下沉
        cloud_pos = explode_pos + np.array([0, 0, -self.cloud_sink_speed]) * dt
        return cloud_pos

    # ========== 新增：几何求交与采样工具 ==========
    def ray_sphere_first_u(self, O, D, C, R):
        """射线 O + u D 与球(C,R) 最近正交点的u；若无交，返回None。D需单位向量"""
        OC = O - C
        b = 2 * np.dot(D, OC)
        c = np.dot(OC, OC) - R * R
        a = 1.0
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        u1 = (-b - sqrt_disc) / (2 * a)
        u2 = (-b + sqrt_disc) / (2 * a)
        u_candidates = [u for u in (u1, u2) if u >= 0]
        if not u_candidates:
            return None
        return min(u_candidates)

    def ray_first_hit_cylinder_with_caps(self, O, D, R, z0, z1, y_center=200.0, eps=1e-6):
        """
        计算射线与有限圆柱(含端盖)的最近正交点距离u。
        圆柱轴为z轴，半径R，z范围[z0, z1]，圆柱中心线位于(x=0, y=y_center)。
        若无交返回None。
        """
        # 将坐标系在y方向平移，使圆柱轴线位于y'=0
        Ox, Oy, Oz = O[0], O[1] - y_center, O[2]
        Dx, Dy, Dz = D[0], D[1], D[2]
        u_min = None

        # 侧面：解 (Ox+u*Dx)^2 + (Oy+u*Dy)^2 = R^2，且 z 在 [z0,z1]
        a = Dx * Dx + Dy * Dy
        b = 2 * (Ox * Dx + Oy * Dy)
        c = Ox * Ox + Oy * Oy - R * R
        if a > eps:
            disc = b * b - 4 * a * c
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                for u in [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]:
                    if u >= 0:
                        z_hit = Oz + u * Dz
                        if z0 - eps <= z_hit <= z1 + eps:
                            if u_min is None or u < u_min:
                                u_min = u
        # 端盖 z=z0 与 z=z1
        if abs(Dz) > eps:
            for z_plane in (z0, z1):
                u = (z_plane - Oz) * 1.0 / Dz
                if u >= 0:
                    xh = Ox + u * Dx
                    yh = Oy + u * Dy
                    if xh * xh + yh * yh <= R * R + 1e-9:
                        if u_min is None or u < u_min:
                            u_min = u
        return u_min

    def sample_cylinder_points(self, n_theta=72, n_z=10, n_r_caps=8):
        """采样圆柱侧面与上下端盖上的点"""
        R = self.target_radius
        H = self.target_height
        yc = self.target_center[1]
        # 侧面
        side_points = []
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        zs = np.linspace(0, H, n_z)
        for z in zs:
            for th in thetas:
                x = R * math.cos(th)
                y = yc + R * math.sin(th)
                side_points.append(np.array([x, y, z]))
        # 端盖
        def disk_points(z_plane):
            pts = []
            rs = np.linspace(0, R, n_r_caps)
            for r in rs:
                for th in thetas:
                    x = r * math.cos(th)
                    y = yc + r * math.sin(th)
                    pts.append(np.array([x, y, z_plane]))
            return pts
        top_points = disk_points(H)
        bottom_points = disk_points(0.0)
        return np.array(side_points), np.array(top_points), np.array(bottom_points)

    def compute_visible_points(self, M, side_pts, top_pts, bot_pts, eps=1e-4):
        """判定从M出发，哪些采样点是该圆柱表面的首交点(可见点)"""
        R = self.target_radius
        z0, z1 = 0.0, self.target_height
        yc = self.target_center[1]
        visible_pts = []
        for P in np.vstack((side_pts, top_pts, bot_pts)):
            D_vec = P - M
            dist = np.linalg.norm(D_vec)
            if dist < 1e-9:
                continue
            D = D_vec / dist
            u_first = self.ray_first_hit_cylinder_with_caps(M, D, R, z0, z1, y_center=yc)
            if u_first is None:
                continue
            if abs(u_first - dist) <= 1e-3:  # P为首交点
                visible_pts.append((P, dist))
        return visible_pts

    def compute_occlusion_ratio(self, t, drop_time, explode_time, n_theta=72, n_z=10, n_r_caps=8):
        """计算时刻t可见采样点中被云团遮挡的比例(0~1)，以及是否全遮挡"""
        if not (explode_time <= t <= explode_time + self.effective_duration):
            return 0.0, False, 0, 0
        M = self.missile_position(t)
        C = self.smoke_cloud_center(t, drop_time, explode_time)
        R_cloud = self.effective_radius
        side_pts, top_pts, bot_pts = self.sample_cylinder_points(n_theta, n_z, n_r_caps)
        visible = self.compute_visible_points(M, side_pts, top_pts, bot_pts)
        if len(visible) == 0:
            return 0.0, False, 0, 0
        blocked = 0
        for P, dist_MP in visible:
            D = (P - M) / dist_MP
            u_cloud = self.ray_sphere_first_u(M, D, C, R_cloud)
            if u_cloud is not None and u_cloud < dist_MP - 1e-6:
                blocked += 1
        ratio = blocked / len(visible)
        full = (blocked == len(visible))
        return ratio, full, blocked, len(visible)

    def is_target_fully_occluded(self, t, drop_time, explode_time):
        """是否在t时刻实现对真目标(圆柱)的全遮挡"""
        _, full, _, _ = self.compute_occlusion_ratio(t, drop_time, explode_time)
        return full

    # ========== 修改：使用“全圆柱遮挡”判定 ==========
    def calculate_blocking_periods(self):
        """计算‘全遮挡’的时间段，并返回时间序列与遮挡比例序列"""
        drop_time = self.drop_delay
        explode_time = drop_time + self.explode_delay
        missile_distance = np.linalg.norm(self.missile_init_pos)
        total_flight_time = missile_distance / self.missile_speed
        print(f"导弹总飞行时间: {total_flight_time:.2f} 秒")
        print(f"烟幕弹投放时间: {drop_time:.2f} 秒")
        print(f"烟幕弹起爆时间: {explode_time:.2f} 秒")
        print(f"烟幕有效结束时间: {explode_time + self.effective_duration:.2f} 秒")

        # 只需考虑起爆有效窗附近
        start_time = max(0, explode_time - 1)
        end_time = min(total_flight_time, explode_time + self.effective_duration + 1)
        dt = 0.01  # 时间步长改为 0.01s
        times = np.arange(start_time, end_time + 1e-9, dt)
        ratios = []
        blocked_periods = []
        current_start = None
        for t in times:
            ratio, full, _, _ = self.compute_occlusion_ratio(t, drop_time, explode_time)
            ratios.append(ratio)
            if full and current_start is None:
                current_start = t
            elif (not full) and current_start is not None:
                blocked_periods.append((current_start, t))
                current_start = None
        if current_start is not None:
            blocked_periods.append((current_start, times[-1]))
        return blocked_periods, drop_time, explode_time, times, np.array(ratios)

    def analyze(self):
        """执行完整分析"""
        print("=" * 60)
        print("烟幕干扰弹对导弹M1有效遮蔽时长分析 (圆柱体 + 全遮挡判定)")
        print("=" * 60)
        
        drop_time = self.drop_delay
        explode_time = drop_time + self.explode_delay
        
        print(f"\n初始条件:")
        print(f"导弹M1初始位置: {self.missile_init_pos}")
        print(f"无人机FY1初始位置: {self.drone_init_pos}")
        print(f"真目标(圆柱)下底圆心: {self.target_center}, R={self.target_radius}m, H={self.target_height}m")
        print(f"导弹速度: {self.missile_speed} m/s, 无人机速度: {self.drone_speed} m/s")
        
        print(f"\n烟幕弹投放位置 (t={drop_time}s): {self.drone_position(drop_time)}")
        print(f"烟幕弹起爆位置 (t={explode_time}s): {self.smoke_bomb_trajectory(explode_time, drop_time)}")
        
        blocked_periods, _, _, times, ratios = self.calculate_blocking_periods()
        
        print(f"\n‘全遮挡’分析结果:")
        total_full_time = 0.0
        if blocked_periods:
            print(f"发现 {len(blocked_periods)} 个全遮挡时间段:")
            for i, (s, e) in enumerate(blocked_periods, 1):
                dur = e - s
                total_full_time += dur
                print(f"  第{i}段: {s:.3f}s - {e:.3f}s (持续 {dur:.3f}s)")
        else:
            print("未发现全遮挡时间段")
        print(f"\n总有效遮蔽时长(全遮挡): {total_full_time:.3f} 秒")

        return total_full_time, blocked_periods

def main():
    """主函数"""
    analyzer = SmokeInterferenceAnalyzer()
    total_time, periods = analyzer.analyze()
    
    print(f"\n最终结果:")
    print(f"烟幕干扰弹对导弹M1的有效遮蔽时长为: {total_time:.3f} 秒")

if __name__ == "__main__":

    main()
