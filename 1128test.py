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

# 手臂 Home 點

HOME_POSITION = {'x': 380, 'y': -45, 'z': 715, 'rx': -180, 'ry': 0, 'rz': -106.1 ,'speed':150}



# 手臂的工作範圍參數 (單位: mm)

WORK_RANGE_X_MIN = 345

WORK_RANGE_X_MAX = 620

WORK_RANGE_Y_MIN = -160

WORK_RANGE_Y_MAX = 160



# 夾取動作高度參數 (單位: mm) -> ★請務必拿尺量測並修改這裡★

SAFE_Z = 715        # 移動時的安全高度

PICK_Z = 406        # 夾取高度

PLACE_Z = 500       # 放置物品的高度



# 放置區位置

PLACE_X = 300

PLACE_Y = -20



# RealSense 設定

RS_WIDTH = 1280

RS_HEIGHT = 720

RS_FPS = 30



# 模型路徑

YOUR_MODEL_PATH = r"C:\Users\User\Desktop\拯救\NCU\114\三上\Robot arm\DRV-Modbus-main\experiment2\weights\best.pt"

TARGET_MIN_CONF = 0.7

SCAN_DURATION_SECONDS = 5.0



# --- [已加回] 夾具偏移量 (物理性質) ---

# Rz=0 時，夾具中心相對於法蘭中心的偏移

# 即使校正矩陣是對的，這段是用來補償「夾爪與法蘭盤」的相對位置

GRIPPER_OFFSET_X = -46.35

GRIPPER_OFFSET_Y = -45.25



# [保留] 餐具夾取點的偏移量 (從中心點沿著方向線往尾部移動的距離)

SPOON_HANDLE_OFFSET_MM = 0.0  



# 新的校正矩陣 (直接對應法蘭盤中心)

LOCKED_T_cam2base = np.array(

([[-0.999658092723382,  0.010269331579717,  0.024046589811345,

          0.123259893134841],

        [ 0.010878810520659,  0.999619349934939,  0.025353633217064,

         -0.134419096681162],

        [-0.023777071609112,  0.025606562919604, -0.999389290917778,

          0.425382962501847],

        [ 0.               ,  0.               ,  0.               ,

          1.               ]])

)



LOCKED_T_robot_to_baseAruco = np.array(

([[-0.011882540632085, -1.010768361829119, -0.264008308855973,

          0.360000014305115],

        [ 1.004955192019391, -0.015534595501931, -0.093490183851549,

         -0.140000000596046],

        [ 0.000000000000002, -0.000000000000001, -0.000000000000156,

          0.405000001192093],

        [ 0.               ,  0.               ,  0.               ,

          1.               ]])

)



# ========= 2. 工具函數 =========

def is_in_working_range(x, y):

    """

    檢查手臂法蘭盤目標 (cmd_x, cmd_y) 是否在工作範圍內。

    """

    if WORK_RANGE_X_MIN <= x <= WORK_RANGE_X_MAX and WORK_RANGE_Y_MIN <= y <= WORK_RANGE_Y_MAX :
        return True

    return False



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

    u, v = (W - 1 - u_r_clamped, H - 1 - v_r_clamped)



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

    [已加回偏移運算]

    根據目標物的座標 (spoon_x, spoon_y) 和預計旋轉角度 (target_angle_deg)，

    計算「法蘭盤中心」應該在哪裡，才能讓「有偏移的夾爪」剛好抓到物體。

    """

    # 將角度轉為弧度

    theta = math.radians(target_angle_deg)

   

    # 計算旋轉後的偏移向量 (2D 旋轉矩陣)

    # x' = x*cos - y*sin

    # y' = x*sin + y*cos

    rotated_offset_x = GRIPPER_OFFSET_X * math.cos(theta) - GRIPPER_OFFSET_Y * math.sin(theta)

    rotated_offset_y = GRIPPER_OFFSET_X * math.sin(theta) + GRIPPER_OFFSET_Y * math.cos(theta)

   

    # 法蘭盤位置 = 物體目標位置 - 旋轉後的夾具偏移量

    cmd_x = spoon_x - rotated_offset_x

    cmd_y = spoon_y - rotated_offset_y

   

    return cmd_x, cmd_y



# --- (HSV + 寬度分析) 方向判斷 ---

def get_orientation_and_direction(roi_img):

    try:

        h, w = roi_img.shape[:2]

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

       

        # (A) 紅色

        lower_red1 = np.array([0, 80, 80]); upper_red1 = np.array([15, 255, 255])

        lower_red2 = np.array([165, 80, 80]); upper_red2 = np.array([180, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

       

        # (B) 藍綠色

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

   

    last_annotated_frame = None



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

           

            last_annotated_frame = display_frame.copy()



            if result.boxes.id is None:

                cv2.imshow(WIN_NAME, display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): break

                continue



            for box in result.boxes:

                track_id = int(box.id[0])

                class_name = model.names[int(box.cls[0])]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                u_center, v_center = (x1 + x2) // 2, (y1 + y2) // 2



                robot_mm, depth_m = get_3d_robot_coords(u_center, v_center, depth_map_m, K)

               

                h_img, w_img = display_frame.shape[:2]

                safe_x1, safe_x2 = max(0, x1), min(w_img, x2)

                safe_y1, safe_y2 = max(0, y1), min(h_img, y2)

               

                detected_angle = 0.0

                if safe_x2 > safe_x1 and safe_y2 > safe_y1:

                    roi = display_frame[safe_y1:safe_y2, safe_x1:safe_x2]

                    detected_angle = get_orientation_and_direction(roi)



                # --- 視覺化標註 ---

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.circle(display_frame, (u_center, v_center), 5, (255, 255, 0), -1)

                cv2.putText(display_frame, f"ID:{track_id} {class_name}", (x1, y1 - 10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



                angle_rad = math.radians(detected_angle)

                line_length = 50

                end_x_dir = int(u_center + line_length * math.cos(angle_rad))

                end_y_dir = int(v_center + line_length * math.sin(angle_rad))

                cv2.arrowedLine(display_frame, (u_center, v_center), (end_x_dir, end_y_dir), (0, 0, 255), 3, tipLength=0.3)



                visual_offset_pixels = -80

                visual_pick_x_pixel = int(u_center + visual_offset_pixels * math.cos(angle_rad))

                visual_pick_y_pixel = int(v_center + visual_offset_pixels * math.sin(angle_rad))

                cv2.circle(display_frame, (visual_pick_x_pixel, visual_pick_y_pixel), 7, (255, 0, 0), -1)

                cv2.putText(display_frame, "Grab Point", (visual_pick_x_pixel + 10, visual_pick_y_pixel + 10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)



                if robot_mm is not None:

                    if track_id not in object_samples:

                        object_samples[track_id] = []

                        object_angles[track_id] = []

                        object_info[track_id] = class_name

                    object_samples[track_id].append(robot_mm)

                    object_angles[track_id].append(detected_angle)

           

            last_annotated_frame = display_frame.copy()



            cv2.imshow(WIN_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

           

    finally:

        cv2.destroyWindow(WIN_NAME)

       

        if last_annotated_frame is not None:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"scan_result_annotated_{timestamp}.png"

            try:

                cv2.imwrite(filename, last_annotated_frame)

                print(f"視覺化結果已保存: {filename}")

            except Exception as e:

                print(f"保存視覺化圖片錯誤: {e}")



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





# ========= 5. 執行夾取函式 =========



# ========= 5. 執行夾取函式 (修正尾部計算錯誤) =========



def execute_pick_and_place(client, item_list):

    """

    讀取掃描結果，執行夾取。

    """

    if not item_list:

        print("無物件可抓取。")

        return



    print("\n" + "="*30)

    print(f"--- 開始執行夾取 (共 {len(item_list)} 個) ---")



    for item in item_list:

        track_id = item['track_id']

        coords = item['detected_coords_mm']

        # 這是原始視覺偵測到的角度 (例如 151 度)

        theta1 = item['angle_head_direction']

       

        # --- 1. 取得目標物中心座標 ---

        obj_x = coords['rx']

        obj_y = coords['ry']

       

        # ==========================================

        # [修正重點在此] 預先計算固定的尾部目標點

        # ==========================================

        # 我們使用原始的視覺角度 (theta1) 來決定尾部在空間中的物理位置。

        # 這個位置是固定的，不應該隨著手臂嘗試不同角度而改變。

       

        # 提醒：視覺角度通常指向物體的「頭部」或「主要方向」。

        # 如果您發現它往「頭部」偏移而不是「尾部」，請把下面這行改成:

        # tail_angle_rad = math.radians(theta1 + 180)

        tail_angle_rad = math.radians(theta1)

       

        # 計算出固定的夾取點 (尾部位置)

        fixed_pick_x = obj_x + SPOON_HANDLE_OFFSET_MM * math.cos(tail_angle_rad)

        fixed_pick_y = obj_y + SPOON_HANDLE_OFFSET_MM * math.sin(tail_angle_rad)



        print(f"[除錯] ID:{track_id} 中心:({obj_x:.1f}, {obj_y:.1f}) 原角:{theta1:.1f}° -> 固定尾部點:({fixed_pick_x:.1f}, {fixed_pick_y:.1f})")

        # ==========================================





        # --- 2. 角度轉換與正規化 (計算手臂該轉幾度) ---

        base_rz = 135.0 - theta1

        base_rz = base_rz % 360

        if base_rz > 180: base_rz -= 360

        elif base_rz < -180: base_rz += 360

           

        # 方案 A: 正向角度

        cand_a = base_rz

        # 方案 B: 反向角度 (+180度)

        cand_b = base_rz + 180.0

        if cand_b > 180: cand_b -= 360

        elif cand_b < -180: cand_b += 360



        # --- 3. 智慧決策 (檢查兩種角度下，法蘭盤是否都會超出範圍) ---

        # 這裡只是初步檢查，我們需要用上面的 fixed_pick_x/y 來代入檢查才準確

        # 為了簡化流程，我們直接進入迴圈嘗試，在迴圈內做最終檢查。

       

        priority_list = [cand_a, cand_b] # 優先嘗試 A，再來 B



        best_rz = None

        best_cmd_x = None

        best_cmd_y = None

       

        print(f"\n[動作] 抓取 ID:{track_id} | 嘗試角度: {cand_a:.1f} (A) 與 {cand_b:.1f} (B)")



        for i, current_rz in enumerate(priority_list):

           

            # --- 4a. 設定目標夾取點 (使用修正後的固定座標) ---

            pick_x = fixed_pick_x

            pick_y = fixed_pick_y



            # --- 4b. 計算手臂法蘭盤中心點 (含 GRIPPER_OFFSET 修正) ---

            # 這裡的重點是：我們要去這一個固定的尾部點 (pick_x, pick_y)，

            # 如果我們用 current_rz 這個角度去抓，法蘭盤中心 (cmd_x, cmd_y) 應該在哪裡？

            cmd_x, cmd_y = calculate_flange_target(pick_x, pick_y, current_rz)

           

            # --- 4c. 最終範圍檢查 ---

            if is_in_working_range(cmd_x, cmd_y):

                best_rz = current_rz

                best_cmd_x = cmd_x

                best_cmd_y = cmd_y

               

                note = "方案 A 成功" if i == 0 else "方案 B (反向) 成功"

                print(f"  -> {note}。使用角度 {best_rz:.1f}°。法蘭盤前往: ({best_cmd_x:.1f}, {best_cmd_y:.1f})")

                break

            else:

                note = "方案 A" if i == 0 else "方案 B"

                print(f"  -> {note} 失敗: 角度 {current_rz:.1f}° 會導致法蘭盤超出範圍 ({cmd_x:.1f}, {cmd_y:.1f})。")





        if best_rz is None:

            print(f"  **ID:{track_id} 夾取失敗**：正反向角度均超出工作範圍。")

            continue



        target_rz = best_rz

        cmd_x = best_cmd_x

        cmd_y = best_cmd_y



        # --- 5. 執行動作 ---

        # A. 移動到上方

        send.Go_Position(client, cmd_x, cmd_y, SAFE_Z, -180, 0, target_rz, 100)

        time.sleep(1)

       

        # B. 鬆開

        send.Grasp_OFF(client)

        time.sleep(0.5)



        # C. 下降

        send.Go_Position(client, cmd_x, cmd_y, PICK_Z, -180, 0, target_rz, 50,mov=1)

        time.sleep(1)

       

        # D. 夾緊

        send.Grasp_ON(client)

        time.sleep(1.0)

       

        # E. 提起

        send.Go_Position(client, cmd_x, cmd_y, SAFE_Z, -180, 0, target_rz, 100)

        time.sleep(1)

       

        # F. 放置 (這裡示範只放一個位置，實際可能需要動態調整放置點)

        send.Go_Position(client, PLACE_X, PLACE_Y, SAFE_Z, -180, 0, 0, 100)

        time.sleep(1)

        send.Grasp_OFF(client)

        print(f"ID:{track_id} 完成。")



    print("任務結束，回 Home。")

    send.Go_Position(client, HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'],

                     HOME_POSITION['rx'], HOME_POSITION['ry'], HOME_POSITION['rz'], 100)

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