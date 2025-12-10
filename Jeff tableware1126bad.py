import numpy as np
import cv2
import pyrealsense2 as rs
from drv_modbus import send, request
from pymodbus.client import ModbusTcpClient
import time
from ultralytics import YOLO
import json 
from datetime import datetime 
import math

# ========= 1. 參數設定 (請確認高度！) =========

# 手臂 Home 點(鏡頭在手臂最前方+X方向)
HOME_POSITION = {'x': 380, 'y': -20, 'z': 715, 'rx': -180, 'ry': 0, 'rz': -105 ,'speed':150}

# 夾取動作高度參數 (單位: mm) -> ★請務必拿尺量測並修改這裡★
SAFE_Z = 715        # 移動時的安全高度 (不會撞到東西的高度)
PICK_Z = 406       # 夾取高度 (夾爪指尖剛好接觸桌面的高度，建議 +2~5mm)
PLACE_Z = 500       # 放置物品的高度

# ========= 1. 參數設定 (請確認高度！) =========
# ... (現有參數) ...

# 放置區位置 (範例)
PLACE_X = 400
PLACE_Y = 315

# ★★★ 手臂工作範圍限制 (請務必根據您的手臂型號設定這些限制！) ★★★
WORK_RANGE = {
    'X_MIN': 345,  # X 軸最小工作範圍
    'X_MAX': 615,   # X 軸最大工作範圍
    'Y_MIN': -160,  # Y 軸最小工作範圍
    'Y_MAX': 160, # Y 軸最大工作範圍
    'Z_MIN': 406,  # Z 軸最小工作範圍
    'Z_MAX': 715   # Z 軸最大工作範圍

}
# --- 手動座標補償量 (MANUAL FIX) ---
# 這是為了校正視覺系統和手臂系統之間的殘餘誤差。
MANUAL_FIX_X = 134.0 
MANUAL_FIX_Y = 43.0
PREFERRED_RZ = -105.0  # 偏好手臂 Rz 角度 (用於優化選擇)
# RealSense 設定
RS_WIDTH = 1280
RS_HEIGHT = 720
RS_FPS = 30

# 模型路徑
YOUR_MODEL_PATH = r"C:\Users\User\Desktop\拯救\NCU\114\三上\Robot arm\DRV-Modbus-main\weights\best.pt"
TARGET_MIN_CONF = 0.7 
SCAN_DURATION_SECONDS = 5.0 

# --- 夾具偏移量 (您提供的數值) ---
# Rz=0 時，夾具中心相對於法蘭中心的偏移
GRIPPER_OFFSET_X = -46.54 
GRIPPER_OFFSET_Y = -45.38

# --- 校準矩陣 (保持不變) ---
LOCKED_T_cam2base = np.array([
    [ 0.029623779563952, -0.999531530619142, -0.007690967592015, 0.124223933238081],
    [-0.999438081866001, -0.029498533975165, -0.01591719225635 , 0.173890014646511],
    [ 0.015682863270334,  0.008158173292537, -0.999843733794523, 0.419878879569209],
    [ 0.               ,  0.               ,  0.               , 1.               ]]
)

LOCKED_T_robot_to_baseAruco = np.array([
    [ 0.975155010676217,  0.020126172979157,  0.817038938619755, 0.245000004768372],
    [ 0.021477800970591,  0.987943618147872, -0.685277627416083,-0.186000004410744],
    [ 0.000000000000002, -0.000000000000003, -0.00000000000006 , 0.405000001192093],
    [ 0.               ,  0.               ,  0.               , 1.               ]]
)

# ========= 2. 工具函數 =========

def pixel_depth_to_camXYZ(u, v, Z, K):
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) / fx * Z; Y = (v - cy) / fy * Z
    return np.array([X, Y, Z], dtype=np.float64)

def get_median_depth(depth_map, u, v, roi_size=9):
    H, W = depth_map.shape; half = roi_size // 2
    u_min = max(0, u - half); u_max = min(W, u + half + 1)
    v_min = max(0, v - half); v_max = min(H, v + half + 1)
    roi = depth_map[v_min:v_max, u_min:u_max]
    roi = roi[roi > 0] 
    if roi.size == 0: return None
    return float(np.median(roi))

def get_3d_robot_coords(u_r, v_r, depth_map_m, K):
    H, W = RS_HEIGHT, RS_WIDTH
    u_r_clamped = max(0, min(W - 1, u_r))
    v_r_clamped = max(0, min(H - 1, v_r))
    u, v = (W - 1 - u_r_clamped, H - 1 - v_r_clamped) # 旋轉回來

    Z = get_median_depth(depth_map_m, u, v, roi_size=9)
    
    if Z is not None:
        p_cam = pixel_depth_to_camXYZ(u, v, Z, K); p_cam_h = np.hstack([p_cam, 1.0])[:, None]
        p_base_h = LOCKED_T_cam2base @ p_cam_h
        p_robot_h = LOCKED_T_robot_to_baseAruco @ p_base_h
        robot_mm = p_robot_h[:3, 0] * 1000.0 
        return robot_mm, Z
    return None, None

def calculate_flange_target(spoon_x, spoon_y, target_angle_deg):
    """
    根據偏移量計算中心軸座標
    """
    # 將角度轉為弧度
    theta = math.radians(target_angle_deg)
    
    # 計算旋轉後的偏移向量 (2D 旋轉矩陣)
    # x' = x*cos - y*sin
    # y' = x*sin + y*cos
    rotated_offset_x = GRIPPER_OFFSET_X * math.cos(theta) - GRIPPER_OFFSET_Y * math.sin(theta)
    rotated_offset_y = GRIPPER_OFFSET_X * math.sin(theta) + GRIPPER_OFFSET_Y * math.cos(theta)
    
    # 中心軸位置 = 餐具位置 - 旋轉後的偏移量
    cmd_x = spoon_x - rotated_offset_x
    cmd_y = spoon_y - rotated_offset_y
    
    return cmd_x, cmd_y

def is_in_range(x, y, z_safe):
    """檢查給定的 (X, Y, Z) 座標是否在機械手臂的工作範圍內。"""
    global WORK_RANGE
    
    x_in = WORK_RANGE['X_MIN'] <= x <= WORK_RANGE['X_MAX']
    y_in = WORK_RANGE['Y_MIN'] <= y <= WORK_RANGE['Y_MAX']
    # 檢查 Z 軸是否在安全高度範圍內
    z_in = WORK_RANGE['Z_MIN'] <= z_safe <= WORK_RANGE['Z_MAX'] 
    
    return x_in and y_in and z_in

def calculate_angle_diff(angle1, angle2):
    """計算兩個角度之間最短的角距離 (考慮 +/- 180 度界線)。"""
    diff = abs(angle1 - angle2)
    # 處理跨越 +/-180 邊界的距離計算
    if diff > 180:
        return 360 - diff
    return diff

# --- (HSV + 寬度分析) 方向判斷 ---
def get_orientation_and_direction(roi_img):
    try:
        h, w = roi_img.shape[:2]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # (A) 紅色
        lower_red1 = np.array([0, 80, 80]); upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 80, 80]); upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # (B) 藍綠色 (您的自訂數值)
        lower_blue_green = np.array([66, 18, 137]) 
        upper_blue_green = np.array([109, 68, 199])
        mask_bg = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        
        target_mask = cv2.bitwise_or(mask_red, mask_bg)
        
        kernel = np.ones((5,5), np.uint8)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)

        if cv2.countNonZero(target_mask) < 100:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, target_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        (center_x, center_y), (rect_w, rect_h), angle = rect
        
        if rect_w < rect_h:
            base_angle = angle + 90
        else:
            base_angle = angle
            
        rad = math.radians(base_angle)
        cos_a = math.cos(rad); sin_a = math.sin(rad)
        pts = largest_contour.reshape(-1, 2)
        projections = pts[:, 0] * cos_a + pts[:, 1] * sin_a
        
        min_proj = np.min(projections); max_proj = np.max(projections)
        length = max_proj - min_proj
        if length == 0: return base_angle
        
        margin = length * 0.25
        cross_dists = -pts[:, 0] * sin_a + pts[:, 1] * cos_a
        
        idx_pos = np.where(projections > (max_proj - margin))
        idx_neg = np.where(projections < (min_proj + margin))
        
        width_pos = 0; width_neg = 0
        if len(idx_pos[0]) > 0: width_pos = np.max(cross_dists[idx_pos]) - np.min(cross_dists[idx_pos])
        if len(idx_neg[0]) > 0: width_neg = np.max(cross_dists[idx_neg]) - np.min(cross_dists[idx_neg])
            
        final_angle = base_angle
        if width_neg > width_pos: final_angle += 180
            
        return final_angle % 360

    except Exception as e:
        return 0.0

def save_scan_to_json(item_list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tableware_scan_{timestamp}.json"
    output_data = {"scan_timestamp": datetime.now().isoformat(), "items": item_list}
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"儲存成功: {filename}")
    except Exception as e:
        print(f"儲存錯誤: {e}")

# ========= 4. 核心流程 =========

def scan_workspace_for_all_items(pipeline, align, model, K, depth_scale):
    print(f"--- 開始 3D 掃描 ---")
    object_samples = {}; object_angles = {}; object_info = {}    
    
    start_time = time.time()
    WIN_NAME = "Scan Window"
    cv2.namedWindow(WIN_NAME)

    try:
        while time.time() - start_time < SCAN_DURATION_SECONDS:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame: continue
            
            depth_map_m = np.asanyarray(depth_frame.get_data()) * depth_scale
            display_frame = np.asanyarray(color_frame.get_data())
            display_frame = cv2.rotate(display_frame, cv2.ROTATE_180)
            
            results = model.track(display_frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=TARGET_MIN_CONF)
            result = results[0]
            
            if result.boxes.id is None:
                cv2.imshow(WIN_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            for box in result.boxes:
                track_id = int(box.id[0]) 
                class_name = model.names[int(box.cls[0])] 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                u_r, v_r = (x1 + x2) // 2, (y1 + y2) // 2

                robot_mm, depth_m = get_3d_robot_coords(u_r, v_r, depth_map_m, K)
                
                h_img, w_img = display_frame.shape[:2]
                safe_x1, safe_x2 = max(0, x1), min(w_img, x2)
                safe_y1, safe_y2 = max(0, y1), min(h_img, y2)
                
                detected_angle = 0.0
                if safe_x2 > safe_x1 and safe_y2 > safe_y1:
                    roi = display_frame[safe_y1:safe_y2, safe_x1:safe_x2]
                    detected_angle = get_orientation_and_direction(roi)

                # 視覺化
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                angle_rad = math.radians(detected_angle)
                end_x = int(u_r + 40 * math.cos(angle_rad))
                end_y = int(v_r + 40 * math.sin(angle_rad))
                cv2.arrowedLine(display_frame, (u_r, v_r), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)

                if robot_mm is not None:
                    if track_id not in object_samples:
                        object_samples[track_id] = []
                        object_angles[track_id] = []
                        object_info[track_id] = class_name 
                    object_samples[track_id].append(robot_mm)
                    object_angles[track_id].append(detected_angle)

            cv2.imshow(WIN_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cv2.destroyWindow(WIN_NAME)

    final_item_list = []
    if not object_samples: return []

    for track_id, samples in object_samples.items():
        avg_coords = np.mean(samples, axis=0)
        angles_rad = np.deg2rad(object_angles[track_id])
        sin_sum = np.sum(np.sin(angles_rad))
        cos_sum = np.sum(np.cos(angles_rad))
        avg_angle_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        
        item_data = {
            "track_id": track_id,
            "class_name": object_info.get(track_id, "unknown"),
            "detected_coords_mm": {
                "rx": round(avg_coords[0], 2),
                "ry": round(avg_coords[1], 2),
                "rz": round(avg_coords[2], 2)
            },
            "angle_head_direction": round(avg_angle_deg, 2)
        }
        final_item_list.append(item_data)
        print(f" [ID:{track_id}] {item_data['class_name']} @ ({item_data['detected_coords_mm']['rx']:.0f}, {item_data['detected_coords_mm']['ry']:.0f}) Angle: {avg_angle_deg:.0f}")

    return final_item_list

# ========= 5. 夾取動作執行函數 =========

def execute_pick_and_place(client, item_list):
    # ... (前面的定義) ...

    for item in item_list:
        track_id = item['track_id']
        coords = item['detected_coords_mm']
        theta1 = item['angle_head_direction'] # 這是 Sys1 (相機角度)
        
        # --- 1. 取得目標物座標並加入補償 ---
        # 【★請確認這兩行存在！★】
        raw_x = coords['rx']
        raw_y = coords['ry']
        
        obj_x = raw_x + MANUAL_FIX_X
        obj_y = raw_y + MANUAL_FIX_Y
        
        # --- 2. 角度轉換與正規化 ---
        # ... (以下繼續，保持不變) ...
        
        # --- 2. 角度轉換與正規化 ---
        base_rz = 135.0 - theta1
        base_rz = base_rz % 360
        if base_rz > 180: base_rz -= 360
        
        # --- 3. 智慧決策 (新增範圍檢查與優化選擇) ---
        
        # 方案 A: 原始轉換角度
        cand_a = base_rz
        cmd_x_a, cmd_y_a = calculate_flange_target(obj_x, obj_y, cand_a)
        in_range_a = is_in_range(cmd_x_a, cmd_y_a, SAFE_Z) 

        # 方案 B: 對稱角度 (+/- 180 度)
        cand_b = base_rz + 180.0
        if cand_b > 180: cand_b -= 360
        elif cand_b < -180: cand_b += 360
        cmd_x_b, cmd_y_b = calculate_flange_target(obj_x, obj_y, cand_b)
        in_range_b = is_in_range(cmd_x_b, cmd_y_b, SAFE_Z)
        
        target_rz = None
        cmd_x, cmd_y = None, None
        note = ""

        # --- 決策流程：優先考慮範圍 ---
        
        if in_range_a and in_range_b:
            # 情況 1: 兩個方案都在範圍內 -> 選擇離 PREFERRED_RZ 最近的
            diff_a = calculate_angle_diff(cand_a, PREFERRED_RZ)
            diff_b = calculate_angle_diff(cand_b, PREFERRED_RZ)
            
            if diff_a <= diff_b: # 優先選擇 A (如果距離相等)
                target_rz, cmd_x, cmd_y = cand_a, cmd_x_a, cmd_y_a
                note = "方案A (優化選擇)"
            else:
                target_rz, cmd_x, cmd_y = cand_b, cmd_x_b, cmd_y_b
                note = "方案B (反向, 優化選擇)"
                
        elif in_range_a:
            # 情況 2: 只有方案 A 在範圍內
            target_rz, cmd_x, cmd_y = cand_a, cmd_x_a, cmd_y_a
            note = "方案A (唯一可用)"
            
        elif in_range_b:
            # 情況 3: 只有方案 B 在範圍內
            target_rz, cmd_x, cmd_y = cand_b, cmd_x_b, cmd_y_b
            note = "方案B (反向, 唯一可用)"
            
        else:
            # 情況 4: 兩個方案都不在範圍內
            print(f"\n[跳過] 抓取 ID:{track_id}：兩個角度方案計算出的 XY 座標都**超出工作範圍**！")
            print(f"  原始視覺: ({raw_x:.1f}, {raw_y:.1f})")
            continue # 跳到下一個物件

        # --- 4. 輸出資訊與執行動作 ---
        
        print(f"\n[動作] 抓取 ID:{track_id}")
        print(f"  原始視覺: ({raw_x:.1f}, {raw_y:.1f}) | 相機角度(Sys1): {theta1:.1f}")
        print(f"  智慧決策: 選 {target_rz:.1f} ({note}, 離{PREFERRED_RZ}最近且在範圍內)")
        print(f"  手臂前往: ({cmd_x:.1f}, {cmd_y:.1f})")

        # --- 動作流程 (保持不變) ---
        
        # A. 移動到上方
        send.Go_Position(client, cmd_x, cmd_y, SAFE_Z, -180, 0, target_rz, 100)
        time.sleep(1) 
        
        # B. 鬆開
        send.Grasp_OFF(client)
        time.sleep(0.5)

        # C. 下降
        # ... (以下保持不變) ...
    
# ========= 6. 主程式入口 =========
if __name__ == "__main__":
    try:
        print("連線機械手臂...")
        c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
        if not c.connect():
            print("無法連線到手臂 Modbus TCP"); exit()
        
        print("啟動 RealSense...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
        config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
        profile = pipeline.start(config)
        
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        print("載入 YOLO 模型...")
        model = YOLO(YOUR_MODEL_PATH)
        
        # 1. 回 Home 拍照
        print("移動至拍照點...")
        send.Go_Position(c, HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'],
                         HOME_POSITION['rx'], HOME_POSITION['ry'], HOME_POSITION['rz'], HOME_POSITION['speed'])
        time.sleep(1) 

        # 2. 掃描
        final_items = scan_workspace_for_all_items(pipeline, align, model, K, depth_scale)
        
        if final_items: 
            save_scan_to_json(final_items)
            
            # 3. 執行夾取
            execute_pick_and_place(c, final_items)
        else:
            print("未偵測到物件，略過夾取。")
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        
    finally:
        try:
            pipeline.stop()
            c.close()
            cv2.destroyAllWindows()
            print("程式已關閉。")
        except:
            pass