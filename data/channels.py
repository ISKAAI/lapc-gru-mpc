"""MF4 通道名常量。集中放一处，改一次全生效。
都取 group=1（20Hz 数据组），名字不带 'xcp.' 前缀。
"""

# 友好名 -> MF4 通道全名
CHANNELS = {
    # --- 参考中心线 (三次多项式, 自车坐标系, 左正右负) ---
    "c0": "lapc_req_lat_ctrl_lane_model.center_line.poly_line.c0",  # -> e1=c0 横向误差(m, 左正右负)
    "c1": "lapc_req_lat_ctrl_lane_model.center_line.poly_line.c1",  # -> e2=c1 航向误差(rad)
    "c2": "lapc_req_lat_ctrl_lane_model.center_line.poly_line.c2",  # -> 道路曲率 κ=2*c2 (1/m)

    # --- 自车运动 ---
    "vx":       "sm_req_ego_motion.velocity.x.value",       # 纵向车速
    "yaw_rate": "sm_req_ego_motion.angle_rate.yaw.value",   # 横摆角速度
    "steer":    "sm_req_ego_motion.steer_wheel_angle.value",# 方向盘转角 (rad), /转向比 -> 前轮

    # --- 状态机激活标志 ---
    "lcc_active": "lapc_req_state_machine_status.lac_status.lcc_activate",  # 0/1
    "alc_active": "lapc_req_state_machine_status.lac_status.alc_activate"
    # "alc_active": 待确认  # 处理 alc 文件夹时再补, 用于 lcc==1 且 alc==0 的筛选
}

DT = 0.05  # 采样步长 50ms (20Hz) —— 注意和仿真的 0.02 不同!
