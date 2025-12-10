import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from drv_modbus import send, request
from pymodbus.client import ModbusTcpClient
import time

# ========= 手臂移位 (在啟動 RealSense 之前) =========
HOME_POSITION = {'x': 380, 'y': -20, 'z': 715, 'rx': -180, 'ry': 0, 'rz': -105 ,'speed':150}
print("正在連線到機械手臂...")
c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
if not c.connect():
    print(f"錯誤：無法連線到 Modbus TCP {c.host}:{c.port}")
    exit()

print(f"連線成功！正在移動手臂至拍照位置...")
send.Go_Position(c, HOME_POSITION['x'], HOME_POSITION['y'], HOME_POSITION['z'],
                 HOME_POSITION['rx'], HOME_POSITION['ry'], HOME_POSITION['rz'], HOME_POSITION['speed'])
move_wait_time = 8 
print(f"正在等待手臂移動 {move_wait_time} 秒...")
time.sleep(move_wait_time) 
print("手臂已到達拍照位置。正在啟動 RealSense 攝影機...")


# ========= RealSense 啟動 (在手臂停止後) =========
context = rs.context()
devices = context.query_devices()
if len(devices) == 0:
    print("錯誤：找不到任何 RealSense 設備！")
    c.close(); exit() 
print(f"找到 {len(devices)} 個設備：")
for dev in devices:
    print(f"  {dev.get_info(rs.camera_info.name)}")

# ========= 狀態變數 =========
mouse_coords = [0, 0]
is_averaging = False        
averaging_start_time = 0    
sample_buffer = []          
target_pixel_uv = (0, 0)    
target_pixel_uv_r = (0, 0)  
AVERAGING_DURATION = 10.0   

def mouse_callback(event, x, y, flags, param):
    global mouse_coords
    mouse_coords[0] = x
    mouse_coords[1] = y

# ========= (新) *** 你的永久校準矩陣 *** =========
# (你剛剛從終端機複製的值)
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

# ---------------------------------------------------


# ========= (新) 簡化的工具函數 =========
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

def draw_image_axes_hint(frame, corner='top-right', margin=18, axis_len_px=80,
                         color_x=(0,170,255), color_y=(255,170,0)):
    h, w = frame.shape[:2]; ox, oy = w - margin, margin
    x_dir, y_dir = (-axis_len_px, 0), (0, axis_len_px)
    lx_off, ly_off = (-axis_len_px - 22, -6), (-6, axis_len_px + 18)
    cv2.arrowedLine(frame, (ox, oy), (ox + x_dir[0], oy + x_dir[1]), color_x, 2, tipLength=0.12)
    cv2.arrowedLine(frame, (ox, oy), (ox + y_dir[0], oy + y_dir[1]), color_y, 2, tipLength=0.12)
    cv2.putText(frame, "+x", (ox + lx_off[0], oy + lx_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_x, 2)
    cv2.putText(frame, "+y", (ox + ly_off[0], oy + ly_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_y, 2)

# ========= 參數 (簡化) =========
pipeline = rs.pipeline(); config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config); align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# (我們仍然需要 K 和 D 來做 pixel_depth_to_camXYZ)
stream = profile.get_stream(rs.stream.color).as_video_stream_profile(); intr = stream.get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
# (D不再需要，因為我們不做 ArUco 偵測了)
# D = np.array(intr.coeffs[:5], dtype=np.float64) 

WIN_NAME = "Color + ArUco + Depth-XYZ (Mouse Probe)"; cv2.namedWindow(WIN_NAME); cv2.setMouseCallback(WIN_NAME, mouse_callback)
print(f"視窗 '{WIN_NAME}' 已建立。")
print("---")
print(">>> CALIBRATION LOCKED (READY) ---")
print(">>> 手臂已就位，校準已載入。")
print(">>> 移動滑鼠瞄準，按下 'a' 鍵 [開始 10 秒平均]。")
print("---")

# (不再需要 matplotlib)
# plt.ion(); fig = plt.figure()
# ax3d = fig.add_subplot(projection='3d')
# Workspace = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]

# ========= 主迴圈 (簡化) =========
try:
    while True:
        # ----- 1. 獲取影像 -----
        frames = pipeline.wait_for_frames(); aligned = align.process(frames)
        color_frame = aligned.get_color_frame(); depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame: continue
        
        # (注意: 我們不再需要 'frame' (原始影像)，因為所有偵測都移除了)
        # frame = np.asanyarray(color_frame.get_data()) 
        depth_map_m = np.asanyarray(depth_frame.get_data()) * depth_scale
        
        # (我們仍然需要原始的 'color_frame' 來旋轉)
        display_frame = np.asanyarray(color_frame.get_data())
        display_frame = cv2.rotate(display_frame, cv2.ROTATE_180); 
        H, W = display_frame.shape[:2]
        
        def unmap_pt(u_r, v_r):
            u_r_clamped = max(0, min(W - 1, u_r))
            v_r_clamped = max(0, min(H - 1, v_r))
            return (W - 1 - u_r_clamped, H - 1 - v_r_clamped)

        # ----- 2. 鍵盤控制 -----
        key = cv2.waitKey(1); 
        if key & 0xFF == ord('q'): break
        if key & 0xFF == ord('a'):
            if not is_averaging:
                is_averaging = True; sample_buffer = []; averaging_start_time = time.time()
                target_pixel_uv_r = (mouse_coords[0], mouse_coords[1]); target_pixel_uv = unmap_pt(target_pixel_uv_r[0], target_pixel_uv_r[1])
                print(f"開始在像素 {target_pixel_uv} 進行 {AVERAGING_DURATION} 秒平均...")
            else:
                is_averaging = False
                if len(sample_buffer) > 0:
                    final_avg = np.mean(sample_buffer, axis=0); print(f"--- 平均已手動停止 (共 {len(sample_buffer)} 筆) ---"); print(f"RX: {final_avg[0]:.2f}, RY: {final_avg[1]:.2f}, RZ: {final_avg[2]:.2f} (mm)"); print("------------------------------")
                sample_buffer = []

        # ----- 3. 繪圖 (永遠執行) -----
        draw_image_axes_hint(display_frame, corner='top-right', axis_len_px=80)
        cv2.putText(display_frame, "CALIBRATION LOCKED (READY)", (W // 2 - 150, 24), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ----- 4. 滑鼠探測 (永遠執行) -----
        probe_text = ""; probe_color = (0, 0, 255); probe_u_r, probe_v_r = 0, 0
        
        if is_averaging:
            # --- 正在平均 ---
            elapsed = time.time() - averaging_start_time
            probe_u_r, probe_v_r = target_pixel_uv_r[0], target_pixel_uv_r[1]
            u, v = target_pixel_uv[0], target_pixel_uv[1]
            Z = get_median_depth(depth_map_m, u, v, roi_size=9)
            
            if Z is not None:
                p_cam = pixel_depth_to_camXYZ(u, v, Z, K); p_cam_h = np.hstack([p_cam, 1.0])[:, None]
                p_base_h = LOCKED_T_cam2base @ p_cam_h
                p_robot_h = LOCKED_T_robot_to_baseAruco @ p_base_h
                p_robot_mm = p_robot_h[:3, 0] * 1000.0; sample_buffer.append(p_robot_mm) 
                probe_color = (0, 255, 255); probe_text = f"AVERAGING... {elapsed:.1f}s / {AVERAGING_DURATION:.1f}s (Samples: {len(sample_buffer)})"
            else: probe_text = "AVERAGING... No Depth!"

            if elapsed >= AVERAGING_DURATION:
                is_averaging = False
                if len(sample_buffer) > 0:
                    final_avg = np.mean(sample_buffer, axis=0); print(f"--- 平均已完成 (共 {len(sample_buffer)} 筆) ---"); print(f"RX: {final_avg[0]:.2f}, RY: {final_avg[1]:.2f}, RZ: {final_avg[2]:.2f} (mm)"); print("------------------------------")
                sample_buffer = []

        else:
            # --- 正常滑鼠探測 (Live) ---
            u_r, v_r = mouse_coords[0], mouse_coords[1]; probe_u_r, probe_v_r = u_r, v_r; u, v = unmap_pt(u_r, v_r)
            Z = get_median_depth(depth_map_m, u, v, roi_size=9)
            
            if Z is not None:
                p_cam = pixel_depth_to_camXYZ(u, v, Z, K); p_cam_h = np.hstack([p_cam, 1.0])[:, None]
                p_base_h = LOCKED_T_cam2base @ p_cam_h
                p_robot_h = LOCKED_T_robot_to_baseAruco @ p_base_h
                p_robot_mm = p_robot_h[:3, 0] * 1000.0; rx, ry, rz = p_robot_mm[0], p_robot_mm[1], p_robot_mm[2]
                probe_color = (0, 255, 255); probe_text = f"Probe: RX:{rx:.1f} RY:{ry:.1f} RZ:{rz:.1f} (mm) | Depth:{Z:.3f}m"
            else: probe_text = "Probe: No Depth Data"

        cv2.circle(display_frame, (probe_u_r, probe_v_r), 5, probe_color, 2)
        cv2.putText(display_frame, probe_text, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, probe_color, 2)

        # ----- 5. 顯示 -----
        cv2.imshow(WIN_NAME, display_frame) 

finally:
    print("正在關閉 Modbus 連線..."); c.close()
    print("正在關閉 RealSense pipeline..."); pipeline.stop()
    cv2.destroyAllWindows()
    # (不再需要 plt.ioff())
    print("程式已結束。")