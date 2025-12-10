# -*- coding: utf-8 -*-
# 目的：
# 1) 用 RealSense 同步取得 Color + Depth，並對齊到 Color
# 2) ArUco 偵測 → 用 IPPE_SQUARE 由角點重算 cam->aruco 位姿
# 3) (新) 以最小 ID 為基準，計算 ArUco -> Robot 的 4x4 校準矩陣
# 4) (新) 顯示所有點在「機械手臂座標系」下的 (RX, RY, RZ)

from landmark import aruco
from landmark import util
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs

context = rs.context()
devices = context.query_devices()

if len(devices) == 0:
    print("錯誤：找不到任何 RealSense 設備！")
    exit() # 如果找不到攝影機，就結束腳本

print(f"找到 {len(devices)} 個設備：")
for dev in devices:
    print(f"  {dev.get_info(rs.camera_info.name)}")

# ========= 小工具 =========
def inv_T(T):
    R, t = T[:3, :3], T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti

def pnp_ippe_square_from_corners(corners, K, D, marker_length):
    """
    corners: (4,2) 角點（OpenCV 順序：TL, TR, BR, BL）
    回傳：T_cam2aruco (4x4)
    ArUco 座標定義：面對標籤時 +X 右、+Y 上、+Z 指向相機（右手）
    """
    s = marker_length / 2.0
    objp = np.array([[-s,  s, 0],   # TL
                     [ s,  s, 0],   # TR
                     [ s, -s, 0],   # BR
                     [-s, -s, 0]],  # BL
                    dtype=np.float64)
    imgp = corners.reshape(4, 2).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.flatten()  # aruco 在相機座標
    return T  # T_cam2aruco

def pixel_depth_to_camXYZ(u, v, Z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.array([X, Y, Z], dtype=np.float64)

# ----- (關鍵) 新的 3D-3D 校準函數 -----
def calculate_robot_to_aruco_transform(aruco_poses_m, robot_coords_m):
    """
    aruco_poses_m: dict {id: (ax, ay, az)} in meters (來源點)
    robot_coords_m: dict {id: (rx, ry, rz)} in meters (目標點)
    
    計算 4x4 轉換矩陣 T_robot_to_baseAruco
    使得 P_robot = T_robot_to_baseAruco @ P_aruco
    """
    src_pts = []
    dst_pts = []
    
    # 找出兩邊都有的 ID
    ids = sorted(set(aruco_poses_m.keys()).intersection(robot_coords_m.keys()))
    
    # 3D 仿射變換至少需要 4 個點（或 3D-3D 剛性變換至少 3 個點）
    # 這裡用 estimateAffine3D，它需要至少 4 個點
    if len(ids) < 4:
        return None, ids
        
    for _id in ids:
        src_pts.append(aruco_poses_m[_id])
        dst_pts.append(robot_coords_m[_id])
        
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 3)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 3)
    
    # 找到 3x4 仿射變換 (src -> dst)
    # P_dst = A @ P_src + t  (A 是 3x3, t 是 3x1)
    ok, T_affine, inliers = cv2.estimateAffine3D(src_pts, dst_pts, ransacThreshold=0.05)
    
    if not ok or T_affine is None:
        return None, ids
        
    # T_affine 是 (3, 4)。轉換為 4x4
    T_robot_to_baseAruco = np.eye(4)
    T_robot_to_baseAruco[:3, :] = T_affine
    
    return T_robot_to_baseAruco, ids

# ----- (舊的 estimate_isotropic_scale 已被刪除) -----

def draw_image_axes_hint(frame, corner='top-right', margin=18, axis_len_px=80,
                         color_x=(0,170,255), color_y=(255,170,0)):
    h, w = frame.shape[:2]
    if corner == 'top-right':
        ox, oy = w - margin, margin
        x_dir = (-axis_len_px, 0); y_dir = (0, axis_len_px)
        lx_off, ly_off = (-axis_len_px - 22, -6), (-6, axis_len_px + 18)
    # (其他 corner... 略)
    else: # 預設 top-right
        ox, oy = w - margin, margin
        x_dir = (-axis_len_px, 0); y_dir = (0, axis_len_px)
        lx_off, ly_off = (-axis_len_px - 22, -6), (-6, axis_len_px + 18)

    cv2.arrowedLine(frame, (ox, oy), (ox + x_dir[0], oy + x_dir[1]), color_x, 2, tipLength=0.12)
    cv2.arrowedLine(frame, (ox, oy), (ox + y_dir[0], oy + y_dir[1]), color_y, 2, tipLength=0.12)
    cv2.putText(frame, "+x", (ox + lx_off[0], oy + lx_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_x, 2)
    cv2.putText(frame, "+y", (ox + ly_off[0], oy + ly_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_y, 2)

# ========= 參數 =========
aruco_dict = aruco.ARUCO_DICT().DICT_5X5_100
aruco_detector = aruco.Aruco(aruco_dict, 1, 300)
aruco_length = 0.08  # 黑色方塊邊長（m，不含白邊）

# ========= RealSense 初始化與對齊 =========
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# ========= (關鍵) 機械手臂實測座標 (m) =========
# 這是你提供的地面真實 (Ground Truth)，單位已轉為公尺 (m)
# ID 1: (360, -185, 405) mm
# ID 2: (358, -111, 405) mm
# ID 3: (603, -185, 405) mm
# ID 4: (600, -111, 405) mm
ROBOT_XYZ_M = {
    1: np.array([0.360, -0.185, 0.405]),
    2: np.array([0.358, -0.111, 0.405]),
    3: np.array([0.603, -0.185, 0.405]),
    4: np.array([0.600, -0.111, 0.405]),
}

# 從 RealSense 讀取 color 內參
stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = stream.get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]], dtype=np.float64)
D = np.array(intr.coeffs[:5], dtype=np.float64) 

# ========= Matplotlib 3D 視圖 =========
plt.ion()
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')
Workspace = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]

# ========= 主迴圈 =========
T_robot_to_baseAruco = None # 校準矩陣 (橋樑)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 1. 取得原始影像 (用於計算)
        frame = np.asanyarray(color_frame.get_data())
        depth_map_m = np.asanyarray(depth_frame.get_data()) * depth_scale

        # ----- ArUco 偵測 (在原始影像上) -----
        ret, _Tc2a_list, _Ta2c_list, id_list, corners_list = aruco.Detect_Aruco(
            frame, K, D, aruco_length,
            aruco_detector.aruco_dict, aruco_detector.aruco_params,
            False 
        )

        # ----- 3D 視圖重置 (顯示 ArUco 基準座標) -----
        ax3d.cla()
        util.Draw_Aruco(ax3d, aruco_length) # 畫出 (0,0,0) 的 ArUco
        ax3d.set_xlim3d(Workspace[0], Workspace[1])
        ax3d.set_ylim3d(Workspace[2], Workspace[3])
        ax3d.set_zlim3d(Workspace[4], Workspace[5])
        ax3d.set_xlabel('ArUco X (m)'); ax3d.set_ylabel('ArUco Y (m)'); ax3d.set_zlabel('ArUco Z (m)')
        ax3d.set_title('Markers in Base-Tag (min ID) Frame')

        # ----- 建立用於「顯示」的旋轉影像 -----
        display_frame = cv2.rotate(frame, cv2.ROTATE_180)
        H, W = display_frame.shape[:2]

        def remap_pt(u, v):
            return (W - 1 - int(round(u)), H - 1 - int(round(v)))

        # ----- 在 "display_frame" 上繪製 UI 介面 -----
        draw_image_axes_hint(display_frame, corner='top-right', axis_len_px=80)

        if not ret or id_list is None or len(id_list) == 0:
            cv2.imshow("Color + ArUco + Depth-XYZ", display_frame)
            plt.show(block=False); plt.pause(0.001)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ----- ArUco 位姿計算 (使用原始資料) -----
        T_cam2aruco_ippe = []
        for corners in corners_list:
            T = pnp_ippe_square_from_corners(corners.reshape(4, 2), K, D, aruco_length)
            T_cam2aruco_ippe.append(T)

        packed = [(int(i), T, c) for i, T, c in zip(id_list, T_cam2aruco_ippe, corners_list) if T is not None]
        
        if len(packed) == 0:
            cv2.imshow("Color + ArUco + Depth-XYZ", display_frame)
            plt.show(block=False); plt.pause(0.001)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        packed.sort(key=lambda x: x[0])
        base_id, T_cam2aruco_base, _ = packed[0]
        corners_by_id = {i: c for i, _, c in packed} 

        # ----- 在 "display_frame" 上繪製 UI 介面 (Base ID) -----
        cv2.putText(display_frame, f"Base ID: {base_id}", (20, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ----- 計算各標籤在「基準標籤座標」下的 ArUco 座標 -----
        aruco_poses_m = {} # {id: (ax, ay, az)}
        for _id, T_cam2aruco_i, _ in packed:
            T_i_in_base = inv_T(T_cam2aruco_base) @ T_cam2aruco_i
            pos = T_i_in_base[:3, 3].copy()   # (ax, ay, az) in meters
            aruco_poses_m[_id] = pos

        # ----- (關鍵) 計算 ArUco基準 -> 手臂基準 的 4x4 轉換矩陣 -----
        # T_robot_to_baseAruco @ P_aruco = P_robot
        T_robot_to_baseAruco, calib_ids = calculate_robot_to_aruco_transform(aruco_poses_m, ROBOT_XYZ_M)
        
        row = 0
        if T_robot_to_baseAruco is None:
            # 校準失敗，顯示警告
            found_ids = sorted(aruco_poses_m.keys())
            cv2.putText(display_frame, f"CALIB FAILED (Need {len(ROBOT_XYZ_M)} tags, found {found_ids})",
                        (20, 52 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            row += 1
        else:
            # 校準成功，顯示 OK
            cv2.putText(display_frame, f"CALIB OK (using IDs: {calib_ids})",
                        (20, 52 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            row += 1

        # ----- 繪製（在 display_frame 上） -----
        # 只有校準成功才繪製座標
        if T_robot_to_baseAruco is not None: 
            for _id, P_aruco in aruco_poses_m.items():
                
                # ArUco 基準座標 (m)
                ax, ay, az = P_aruco[0], P_aruco[1], P_aruco[2]
                
                # (轉換) 轉換到手臂座標 (m)
                P_aruco_h = np.array([ax, ay, az, 1.0])[:, None] # (4,1)
                P_robot_h = T_robot_to_baseAruco @ P_aruco_h
                P_robot_m = P_robot_h[:3, 0]
                
                # 轉換為 mm
                rx, ry, rz = P_robot_m * 1000.0
                
                # 3D 繪圖 (matplotlib) - 畫 ArUco 座標 (ax, ay, az)
                ax3d.scatter(ax, ay, az, s=25)
                ax3d.text(ax, ay, az, f'ID:{_id}', fontsize=9)

                # ----- 繪製影像上的圓圈 -----
                pts = corners_by_id[_id].reshape(-1, 2)
                uc, vc = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                uc_r, vc_r = remap_pt(uc, vc)
                cv2.circle(display_frame, (uc_r, vc_r), 6, (0, 255, 0), -1)

                # ----- 繪製 UI 文字 (顯示轉換後的手臂座標) -----
                cv2.putText(display_frame, f"ID:{_id}  RX:{rx:.1f} RY:{ry:.1f} RZ:{rz:.1f} (mm)",
                            (20, 52 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
                row += 1

        # ----- （選配）像素+深度 → 基準世界座標 → 機械手臂座標 -----
        # (以基準標籤 ID 1 的中心點為示範)
        base_corners = corners_by_id.get(base_id, None)
        if base_corners is not None and T_robot_to_baseAruco is not None: # 必須校準成功
            # 1. 取得「原始」像素中心點 (u, v)
            pts = base_corners.reshape(-1, 2)
            u, v = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            
            Z = float(depth_map_m[int(round(v)), int(round(u))])
            if np.isfinite(Z) and Z > 0:
                # 1. 像素 -> 相機座標 (m)
                p_cam = pixel_depth_to_camXYZ(u, v, Z, K) 
                p_cam_h = np.hstack([p_cam, 1.0])[:, None] # (4,1)
                
                # 2. 相機座標 -> ArUco 基準座標 (m)
                T_cam2base = inv_T(T_cam2aruco_base) 
                p_base_h = T_cam2base @ p_cam_h
                # p_base = p_base_h[:3, 0] # (這是 ArUco 座標)
                
                # 3. ArUco 基準座標 -> 機械手臂座標 (m -> mm)
                p_robot_h = T_robot_to_baseAruco @ p_base_h
                p_robot_mm = p_robot_h[:3, 0] * 1000.0 # 轉為 mm
                rx, ry, rz = p_robot_mm[0], p_robot_mm[1], p_robot_mm[2]

                # 4. 將 (u, v) 轉換為旋轉後的 (uc_r, vc_r)
                uc_r, vc_r = remap_pt(u, v)
                
                # 5. 在 "display_frame" 上的「新座標」繪製圓圈
                cv2.circle(display_frame, (uc_r, vc_r), 5, (0, 255, 255), -1)
                
                # 6. 繪製 UI 文字 (顯示最終的手臂座標)
                cv2.putText(display_frame, f"Base(depth): RX:{rx:.1f} RY:{ry:.1f} RZ:{rz:.1f} (mm)",
                            (20, 52 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)


        # ----- 顯示 -----
        plt.show(block=False); plt.pause(0.001)
        cv2.imshow("Color + ArUco + Depth-XYZ", display_frame) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.ioff()