from landmark import aruco
from landmark import util
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from drv_modbus import send, request
from pymodbus.client import ModbusTcpClient
import time

def inv_T(T):
    R, t = T[:3, :3], T[:3, 3:4]; Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T; Ti[:3, 3:4] = -R.T @ t
    return Ti
def pnp_ippe_square_from_corners(corners, K, D, marker_length):
    s = marker_length / 2.0
    objp = np.array([[-s,  s, 0], [ s,  s, 0], [ s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    imgp = corners.reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok: ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_EPNP)
    if not ok: return None
    R, _ = cv2.Rodrigues(rvec); T = np.eye(4)
    T[:3, :3] = R; T[:3, 3]  = tvec.flatten()
    return T
def pixel_depth_to_camXYZ(u, v, Z, K):
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) / fx * Z; Y = (v - cy) / fy * Z
    return np.array([X, Y, Z], dtype=np.float64)
def calculate_robot_to_aruco_transform(aruco_poses_m, robot_coords_m):
    src_pts, dst_pts = [], []
    ids = sorted(set(aruco_poses_m.keys()).intersection(robot_coords_m.keys()))
    if len(ids) < 4: return None, ids
    for _id in ids:
        src_pts.append(aruco_poses_m[_id]); dst_pts.append(robot_coords_m[_id])
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 3)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 3)
    ok, T_affine, inliers = cv2.estimateAffine3D(src_pts, dst_pts, ransacThreshold=0.05)
    if not ok or T_affine is None: return None, ids
    T_robot_to_baseAruco = np.eye(4)
    T_robot_to_baseAruco[:3, :] = T_affine
    return T_robot_to_baseAruco, ids
def get_median_depth(depth_map, u, v, roi_size=9):
    H, W = depth_map.shape; half = roi_size // 2
    u_min = max(0, u - half); u_max = min(W, u + half + 1)
    v_min = max(0, v - half); v_max = min(H, v + half + 1)
    roi = depth_map[v_min:v_max, u_min:u_max]
    roi = roi[roi > 0]
    if roi.size == 0: return None
    return float(np.median(roi))
def draw_image_axes_hint(frame, corner='top-right', margin=18, axis_len_px=80,
                         color_x=(0,170,255), color_y=(255,170,0)):
    h, w = frame.shape[:2]; ox, oy = w - margin, margin
    x_dir, y_dir = (-axis_len_px, 0), (0, axis_len_px)
    lx_off, ly_off = (-axis_len_px - 22, -6), (-6, axis_len_px + 18)
    cv2.arrowedLine(frame, (ox, oy), (ox + x_dir[0], oy + x_dir[1]), color_x, 2, tipLength=0.12)
    cv2.arrowedLine(frame, (ox, oy), (ox + y_dir[0], oy + y_dir[1]), color_y, 2, tipLength=0.12)
    cv2.putText(frame, "+x", (ox + lx_off[0], oy + lx_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_x, 2)
    cv2.putText(frame, "+y", (ox + ly_off[0], oy + ly_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_y, 2)
def average_pose_dictionaries(buffer):
    avg_poses = {}; counts = {}
    for pose_dict in buffer:
        for _id, pos in pose_dict.items():
            if _id not in avg_poses:
                avg_poses[_id] = np.zeros(3); counts[_id] = 0
            avg_poses[_id] += pos; counts[_id] += 1
    for _id in avg_poses:
        avg_poses[_id] /= counts[_id]
    return avg_poses
def refine_T_with_depth(T_pnp, corners, depth_map, K):
    """
    使用深度圖的數據來修正 PnP 算出來的 Translation (tvec)。
    保留 PnP 的旋轉 (R)，但強制將 Z 軸距離對齊到 Depth Sensor。
    """
    # --- 修正點開始 ---
    # 強制將角落點轉為 (N, 2) 的形狀，解決 (4,2) 與 (1,4,2) 的維度衝突問題
    pts = corners.reshape(-1, 2)
    
    # 計算中心點 (平均值)
    center_u = np.mean(pts[:, 0])
    center_v = np.mean(pts[:, 1])
    # --- 修正點結束 ---
    
    # 2. 從深度圖取得該點的深度 (使用中位數濾波避免雜訊)
    u_int, v_int = int(center_u), int(center_v)
    
    # 確保座標在圖片範圍內
    h, w = depth_map.shape
    if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
        return T_pnp

    Z_depth = get_median_depth(depth_map, u_int, v_int, roi_size=5)
    
    if Z_depth is None or Z_depth <= 0:
        return T_pnp # 如果抓不到深度，只好回退用原本的 PnP
        
    # 3. 反投影回相機座標系 (Camera Coordinate)
    P_cam = pixel_depth_to_camXYZ(center_u, center_v, Z_depth, K)
    
    # 4. 構建新的 T 矩陣
    T_new = T_pnp.copy()
    T_new[:3, 3] = P_cam # 替換掉原本的 tvec
    
    return T_new
# (手臂移位...)
HOME_POSITION = {'x': 140, 'y': 345, 'z': 600, 'rx': -180, 'ry': 0, 'rz': -17 ,'speed':150}
print("正在連線到機械手臂..."); c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
if not c.connect(): print(f"錯誤：無法連線到 Modbus TCP {c.host}:{c.port}"); exit()
print(f"連線成功！正在移動手臂至拍照位置..."); send.Go_Position(c, HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'], HOME_POSITION['rx'], HOME_POSITION['ry'], HOME_POSITION['rz'], HOME_POSITION['speed'])
move_wait_time = 1
print(f"正在等待手臂移動 {move_wait_time} 秒..."); time.sleep(move_wait_time); print("手臂已到達拍照位置。正在啟動 RealSense 攝影機...")
context = rs.context(); devices = context.query_devices()
if len(devices) == 0: print("錯誤：找不到任何 RealSense 設備！"); c.close(); exit() 
print(f"找到 {len(devices)} 個設備：")
for dev in devices: print(f"  {dev.get_info(rs.camera_info.name)}")

# (狀態變數...)
program_state = "CALIBRATING"; CALIBRATION_DURATION = 5.0; calibration_start_time = time.time()
cam_pose_buffer = []; aruco_poses_buffer = []
mouse_coords = [0, 0]; is_averaging = False; averaging_start_time = 0; sample_buffer = []
target_pixel_uv = (0, 0); target_pixel_uv_r = (0, 0); AVERAGING_DURATION = 5
locked_state = {"T_cam2base": None, "T_robot_to_baseAruco": None, "calib_ok": False}
def mouse_callback(event, x, y, flags, param): global mouse_coords; mouse_coords[0] = x; mouse_coords[1] = y

# (參數...)
aruco_dict = aruco.ARUCO_DICT().DICT_5X5_100; aruco_detector = aruco.Aruco(aruco_dict, 1, 300)
aruco_length = 0.0515; pipeline = rs.pipeline(); config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config); align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
ROBOT_XYZ_M = {1: np.array([-0.105 , 0.280, 0.205]), 
               2: np.array([-0.096 , 0.591, 0.205]), 
               3: np.array([0.323 , 0.290, 0.205]), 
               4: np.array([0.330 , 0.593, 0.205])}
stream = profile.get_stream(rs.stream.color).as_video_stream_profile(); intr = stream.get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
D = np.array(intr.coeffs[:5], dtype=np.float64) 
WIN_NAME = "Color + ArUco + Depth-XYZ (Mouse Probe)"; cv2.namedWindow(WIN_NAME); cv2.setMouseCallback(WIN_NAME, mouse_callback)
print(f"視窗 '{WIN_NAME}' 已建立。"); print("---"); print(f">>> 手臂已就位。正在開始 {CALIBRATION_DURATION} 秒自動校準..."); print(">>> 請確保所有 ArUco 標籤都在視野內且清晰。"); print("---")

# (主迴圈...)
try:
    while True:
        frames = pipeline.wait_for_frames(); aligned = align.process(frames)
        color_frame = aligned.get_color_frame(); depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame: continue
        frame = np.asanyarray(color_frame.get_data()); depth_map_m = np.asanyarray(depth_frame.get_data()) * depth_scale
        display_frame = cv2.rotate(frame, cv2.ROTATE_180); H, W = display_frame.shape[:2]
        def unmap_pt(u_r, v_r): u_r_clamped = max(0, min(W - 1, u_r)); v_r_clamped = max(0, min(H - 1, v_r)); return (W - 1 - u_r_clamped, H - 1 - v_r_clamped)
        key = cv2.waitKey(1); 
        if key & 0xFF == ord('q'): break
        if key & 0xFF == ord('a'):
            if program_state != "READY": print("錯誤：必須等待 10 秒自動校準完成，才能開始平均。")
            else:
                if not is_averaging:
                    is_averaging = True; sample_buffer = []; averaging_start_time = time.time()
                    target_pixel_uv_r = (mouse_coords[0], mouse_coords[1]); target_pixel_uv = unmap_pt(target_pixel_uv_r[0], target_pixel_uv_r[1])
                    print(f"開始在像素 {target_pixel_uv} 進行 {AVERAGING_DURATION} 秒平均...")
                else:
                    is_averaging = False
                    if len(sample_buffer) > 0:
                        final_avg = np.mean(sample_buffer, axis=0); print(f"--- 平均已手動停止 (共 {len(sample_buffer)} 筆) ---"); print(f"RX: {final_avg[0]:.2f}, RY: {final_avg[1]:.2f}, RZ: {final_avg[2]:.2f} (mm)"); print("------------------------------")
                    sample_buffer = []

        if program_state == "CALIBRATING":
            ret, _Tc2a_list, _Ta2c_list, id_list, corners_list = aruco.Detect_Aruco(frame, K, D, aruco_length, aruco_detector.aruco_dict, aruco_detector.aruco_params, False)
            aruco_found_this_frame = False
            if ret and id_list is not None and len(id_list) > 0:
                T_cam2aruco_ippe = [pnp_ippe_square_from_corners(c.reshape(4, 2), K, D, aruco_length) for c in corners_list]
                T_cam2aruco_refined = []
                for i, T in enumerate(T_cam2aruco_ippe):
                    if T is not None:
                # 傳入 depth_map_m (公尺單位的深度圖)
                        T_fixed = refine_T_with_depth(T, corners_list[i], depth_map_m, K)
                        T_cam2aruco_refined.append(T_fixed)
                    else:
                        T_cam2aruco_refined.append(None)
                    T_cam2aruco_ippe = T_cam2aruco_refined
                packed = [(int(i), T, c) for i, T, c in zip(id_list, T_cam2aruco_ippe, corners_list) if T is not None]
                if len(packed) > 0:
                    packed.sort(key=lambda x: x[0]); base_id, T_cam2aruco_base, _ = packed[0]; T_cam2base = inv_T(T_cam2aruco_base); aruco_poses_m = {} 
                    for _id, T_cam2aruco_i, _ in packed: T_i_in_base = inv_T(T_cam2aruco_base) @ T_cam2aruco_i; aruco_poses_m[_id] = T_i_in_base[:3, 3].copy()
                    T_robot, calib_ids = calculate_robot_to_aruco_transform(aruco_poses_m, ROBOT_XYZ_M)
                    if T_robot is not None:
                        cam_pose_buffer.append(T_cam2base); aruco_poses_buffer.append(aruco_poses_m); aruco_found_this_frame = True
            elapsed = time.time() - calibration_start_time
            cv2.putText(display_frame, f"CALIBRATING... {elapsed:.1f}s / {CALIBRATION_DURATION:.1f}s (Samples: {len(cam_pose_buffer)})", (W // 2 - 200, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            if not aruco_found_this_frame: cv2.putText(display_frame, "Looking for all 4 ArUco tags...", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if elapsed >= CALIBRATION_DURATION:
                if not cam_pose_buffer or not aruco_poses_buffer:
                    print("校準失敗：10 秒內未能穩定偵測到所有 4 個標籤。正在重試..."); calibration_start_time = time.time()
                else:
                    print(f"校準 10 秒完成，收集到 {len(cam_pose_buffer)} 筆有效樣本。"); print("正在計算平均轉換矩陣...")
                    LOCKED_T_cam2base = np.mean(np.array(cam_pose_buffer), axis=0)
                    U, S, Vt = np.linalg.svd(LOCKED_T_cam2base[:3, :3]); LOCKED_T_cam2base[:3, :3] = U @ Vt
                    avg_poses = average_pose_dictionaries(aruco_poses_buffer)
                    LOCKED_T_robot_to_baseAruco, _ = calculate_robot_to_aruco_transform(avg_poses, ROBOT_XYZ_M)
                    
                    if LOCKED_T_robot_to_baseAruco is None:
                         print("校準失敗：無法從平均值計算轉換。正在重試..."); calibration_start_time = time.time()
                    else:
                        locked_state["T_cam2base"] = LOCKED_T_cam2base
                        locked_state["T_robot_to_baseAruco"] = LOCKED_T_robot_to_baseAruco
                        locked_state["calib_ok"] = True
                        program_state = "READY" 
                        print("--- CALIBRATION LOCKED (READY) ---")
                        print("現在可以按 'a' 鍵開始探測。")
                        
                        # (*** 新增 ***) ------------------------------------
                        # 格式化輸出，以便複製
                        np.set_printoptions(precision=15, suppress=True)
                        print("\n--- (請複製以下矩陣) ---")
                        print("LOCKED_T_cam2base = np.array(")
                        print(repr(LOCKED_T_cam2base))
                        print(")")
                        print("\nLOCKED_T_robot_to_baseAruco = np.array(")
                        print(repr(LOCKED_T_robot_to_baseAruco))
                        print(")")
                        print("--- (複製完畢) ---\n")
                        # --------------------------------------------------

        if program_state == "READY":
            draw_image_axes_hint(display_frame, corner='top-right', axis_len_px=80)
            cv2.putText(display_frame, "CALIBRATION LOCKED (READY)", (W // 2 - 150, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            probe_text = ""; probe_color = (0, 0, 255); probe_u_r, probe_v_r = 0, 0
            if is_averaging:
                elapsed = time.time() - averaging_start_time; probe_u_r, probe_v_r = target_pixel_uv_r[0], target_pixel_uv_r[1]; u, v = target_pixel_uv[0], target_pixel_uv[1]
                Z = get_median_depth(depth_map_m, u, v, roi_size=9)
                if Z is not None:
                    p_cam = pixel_depth_to_camXYZ(u, v, Z, K); p_cam_h = np.hstack([p_cam, 1.0])[:, None]
                    p_base_h = locked_state["T_cam2base"] @ p_cam_h; p_robot_h = locked_state["T_robot_to_baseAruco"] @ p_base_h
                    p_robot_mm = p_robot_h[:3, 0] * 1000.0; sample_buffer.append(p_robot_mm) 
                    probe_color = (0, 255, 255); probe_text = f"AVERAGING... {elapsed:.1f}s / {AVERAGING_DURATION:.1f}s (Samples: {len(sample_buffer)})"
                else: probe_text = "AVERAGING... No Depth!"
                if elapsed >= AVERAGING_DURATION:
                    is_averaging = False
                    if len(sample_buffer) > 0:
                        final_avg = np.mean(sample_buffer, axis=0); print(f"--- 平均已完成 (共 {len(sample_buffer)} 筆) ---"); print(f"RX: {final_avg[0]:.2f}, RY: {final_avg[1]:.2f}, RZ: {final_avg[2]:.2f} (mm)"); print("------------------------------")
                    sample_buffer = []
            else:
                u_r, v_r = mouse_coords[0], mouse_coords[1]; probe_u_r, probe_v_r = u_r, v_r; u, v = unmap_pt(u_r, v_r)
                Z = get_median_depth(depth_map_m, u, v, roi_size=9)
                if Z is not None:
                    p_cam = pixel_depth_to_camXYZ(u, v, Z, K); p_cam_h = np.hstack([p_cam, 1.0])[:, None]
                    p_base_h = locked_state["T_cam2base"] @ p_cam_h; p_robot_h = locked_state["T_robot_to_baseAruco"] @ p_base_h
                    p_robot_mm = p_robot_h[:3, 0] * 1000.0; rx, ry, rz = p_robot_mm[0], p_robot_mm[1], p_robot_mm[2]
                    probe_color = (0, 255, 255); probe_text = f"Probe: RX:{rx:.1f} RY:{ry:.1f} RZ:{rz:.1f} (mm) | Depth:{Z:.3f}m"
                else: probe_text = "Probe: No Depth Data"
            cv2.circle(display_frame, (probe_u_r, probe_v_r), 5, probe_color, 2); cv2.putText(display_frame, probe_text, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, probe_color, 2)
        cv2.imshow(WIN_NAME, display_frame) 
finally:
    print("正在關閉 Modbus 連線..."); c.close(); print("正在關閉 RealSense pipeline..."); pipeline.stop(); cv2.destroyAllWindows(); plt.ioff(); print("程式已結束。")