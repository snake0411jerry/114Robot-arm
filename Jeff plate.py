import os
import numpy as np
import cv2
import pyrealsense2 as rs
from drv_modbus import send
from pymodbus.client import ModbusTcpClient
import time
from ultralytics import YOLO

# 避免 OpenMP 衝突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ========= 1. 參數設定 =========
HOME_POSITION = {'x': 140, 'y': 345, 'z': 600, 'rx': -180, 'ry': 0, 'rz': -17 ,'speed':150}
RS_WIDTH = 1280
RS_HEIGHT = 720
RS_FPS = 30
YOUR_MODEL_PATH = r"C:\Users\User\Desktop\拯救\NCU\114\三上\Robot arm\DRV-Modbus-main\weights\best.pt"
TARGET_MIN_CONF = 0.7
SAMPLING_DURATION = 3.0 # 取樣時間 (秒)

# ★★★ 你的校正矩陣 ★★★
LOCKED_T_cam2base = np.array(
([[ 0.016620923030916, -0.99906785048456 , -0.039839365529101,
         0.178059279919845],
       [-0.999485346235243, -0.017694843751682,  0.026756964805844,
        -0.215909431687569],
       [-0.027436974662269,  0.039374136597068, -0.998847781090101,
         0.495917283728009],
       [ 0.               ,  0.               ,  0.               ,
         1.               ]])
)

LOCKED_T_robot_to_baseAruco = np.array(
([[ 0.039097425919592, -1.07549324693531 ,  1.605492181284141,
        -0.104999996721745],
       [ 1.041551569469163, -0.130742374983054,  3.82154540354902 ,
         0.280000001192093],
       [ 0.000000000000001, -0.000000000000005,  0.000000000000152,
         0.204999998211861],
       [ 0.               ,  0.               ,  0.               ,
         1.               ]])
)
# ========= 2. 計算函式 =========
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
    u, v = (W - 1 - u_r_clamped, H - 1 - v_r_clamped) # 假設畫面有翻轉180度

    Z = get_median_depth(depth_map_m, u, v, roi_size=9)
    if Z is not None:
        p_cam = pixel_depth_to_camXYZ(u, v, Z, K); p_cam_h = np.hstack([p_cam, 1.0])[:, None]
        p_base_h = LOCKED_T_cam2base @ p_cam_h
        p_robot_h = LOCKED_T_robot_to_baseAruco @ p_base_h
        return p_robot_h[:3, 0] * 1000.0, Z
    return None, None

# ========= 3. 核心功能 (3秒平均) =========

def real_time_plate_averaging(pipeline, align, model, K, depth_scale):
    print(f"--- 開始偵測 (每個 ID 會取 {SAMPLING_DURATION} 秒平均) ---")
    print("按 'r' 重置所有 ID，按 'q' 離開。")
    
    WIN_NAME = "Plate Averaging Tool"
    cv2.namedWindow(WIN_NAME)
    
    # 用字典來儲存每個 ID 的狀態
    # 格式: { track_id: { 'start_time': float, 'samples': [], 'final_coord': np.array, 'status': 'sampling' } }
    object_data = {} 

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue
            
            depth_map_m = np.asanyarray(depth_frame.get_data()) * depth_scale
            display_frame = np.asanyarray(color_frame.get_data())
            display_frame = cv2.rotate(display_frame, cv2.ROTATE_180)

            results = model.track(display_frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=TARGET_MIN_CONF)
            result = results[0]
            
            current_ids = []

            if result.boxes.id is not None:
                for box in result.boxes:
                    track_id = int(box.id[0])
                    current_ids.append(track_id)
                    class_name = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    u_center, v_center = (x1 + x2) // 2, (y1 + y2) // 2 

                    # 初始化新 ID
                    if track_id not in object_data:
                        object_data[track_id] = {
                            'start_time': time.time(),
                            'samples': [],
                            'final_coord': None,
                            'status': 'sampling'
                        }
                    
                    obj = object_data[track_id]
                    
                    # 只有在採樣狀態才計算與儲存
                    if obj['status'] == 'sampling':
                        robot_mm, _ = get_3d_robot_coords(u_center, v_center, depth_map_m, K)
                        if robot_mm is not None:
                            obj['samples'].append(robot_mm)
                        
                        elapsed = time.time() - obj['start_time']
                        
                        # 顯示黃色進度條
                        progress = min(elapsed / SAMPLING_DURATION, 1.0)
                        bar_w = int((x2 - x1) * progress)
                        cv2.rectangle(display_frame, (x1, y1), (x1 + bar_w, y1 + 5), (0, 255, 255), -1)
                        
                        # 時間到，計算平均
                        if elapsed >= SAMPLING_DURATION:
                            if len(obj['samples']) > 0:
                                avg_coord = np.mean(obj['samples'], axis=0)
                                obj['final_coord'] = avg_coord
                                obj['status'] = 'locked'
                                print(f"[完成] ID:{track_id} 平均座標: X:{avg_coord[0]:.2f}, Y:{avg_coord[1]:.2f}, Z:{avg_coord[2]:.2f}")
                            else:
                                # 雖然時間到但沒深度數據，重置時間
                                obj['start_time'] = time.time() 
                    
                    # 繪製畫面
                    color = (0, 255, 255) # 黃色 (採樣中)
                    text_content = ""
                    
                    if obj['status'] == 'sampling':
                        color = (0, 255, 255)
                        text_content = f"Sampling... {len(obj['samples'])}"
                        cv2.circle(display_frame, (u_center, v_center), 5, color, 2)
                    
                    elif obj['status'] == 'locked':
                        color = (0, 255, 0) # 綠色 (鎖定)
                        rx, ry, rz = obj['final_coord']
                        text_content = f"LOCKED: ({rx:.0f}, {ry:.0f}, {rz:.0f})"
                        cv2.circle(display_frame, (u_center, v_center), 5, color, -1)
                        
                        # 額外顯示詳細數值在框框下方
                        cv2.putText(display_frame, f"X:{rx:.1f} Y:{ry:.1f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"ID:{track_id} {text_content}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            else:
                cv2.putText(display_frame, "Looking for plates...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # 顯示操作提示
            cv2.putText(display_frame, "[R] Reset  [Q] Quit", (20, RS_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(WIN_NAME, display_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('r'):
                print("重置所有數據...")
                object_data.clear() # 清空字典，重新開始

    finally:
        cv2.destroyAllWindows()

# ========= 4. 主程式 =========
if __name__ == "__main__":
    try:
        print("連線機械手臂 (移動至拍照點)...")
        c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
        if c.connect():
            send.Go_Position(c, HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'],
                             HOME_POSITION['rx'], HOME_POSITION['ry'], HOME_POSITION['rz'], HOME_POSITION['speed'])
            time.sleep(3) 
            c.close()
        else:
            print("注意：無法連線手臂，僅執行相機測試。")

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
        
        # 執行平均程式
        real_time_plate_averaging(pipeline, align, model, K, depth_scale)

    except Exception as e:
        print(f"發生錯誤: {e}")
    finally:
        try:
            pipeline.stop()
            print("程式結束。")
        except:
            pass