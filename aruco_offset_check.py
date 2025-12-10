# -*- coding: utf-8 -*-
# 多 ArUco 驗證：同時顯示每張標籤相對基準標籤的位移量（影像 + 3D），並在影像上畫出
# 1) 影像座標軸（以影像中心為原點）
# 2) 基準 ArUco 的 3D 世界軸投影（X=紅、Y=綠、Z=藍）

from landmark import aruco, util
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# ============ RealSense 初始化 ============
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# 相機內參
stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = stream.get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]], dtype=np.float64)
D = np.array(intr.coeffs[:5], dtype=np.float64)


# ============ ArUco 初始化 ============
aruco_dict = aruco.ARUCO_DICT().DICT_5X5_100
aruco_detector = aruco.Aruco(aruco_dict, 1, 300)
marker_length = 0.08  # 公尺

# ============ 工具 ============
def inv_T(T):
    R, t = T[:3, :3], T[:3, 3:4]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti

def draw_frame(ax, T, length=0.05, label=None):
    """在 3D 圖上畫一個座標軸（以 T 的位置/方向），長度 length (m)。"""
    o = T[:3, 3]
    R = T[:3, :3]
    x_axis = o + R @ np.array([length, 0, 0])
    y_axis = o + R @ np.array([0, length, 0])
    z_axis = o + R @ np.array([0, 0, length])
    ax.plot([o[0], x_axis[0]], [o[1], x_axis[1]], [o[2], x_axis[2]])
    ax.plot([o[0], y_axis[0]], [o[1], y_axis[1]], [o[2], y_axis[2]])
    ax.plot([o[0], z_axis[0]], [o[1], z_axis[1]], [o[2], z_axis[2]])
    if label:
        ax.text(o[0], o[1], o[2], label)

def draw_image_axes_hint(frame, corner='top-left', margin=18, axis_len_px=90,
                         color_x=(0, 170, 255), color_y=(255, 170, 0)):
    """
    在影像角落畫 2D 影像座標軸提示（OpenCV慣例：+x向右、+y向下）。
    corner: 'top-left' / 'top-right' / 'bottom-left' / 'bottom-right'
    """
    h, w = frame.shape[:2]

    # 決定原點位置
    if corner == 'top-left':
        ox, oy = margin, margin
        x_dir, y_dir = (axis_len_px, 0), (0, axis_len_px)
        lx_off, ly_off = (6, -6), (6, 18)  # 文字偏移
    elif corner == 'top-right':
        ox, oy = w - margin, margin
        x_dir, y_dir = (-axis_len_px, 0), (0, axis_len_px)
        lx_off, ly_off = (-axis_len_px - 22, -6), (-6, axis_len_px + 18)
    elif corner == 'bottom-left':
        ox, oy = margin, h - margin
        x_dir, y_dir = (axis_len_px, 0), (0, -axis_len_px)
        lx_off, ly_off = (6, 18), (6, -axis_len_px - 10)
    else:  # 'bottom-right'
        ox, oy = w - margin, h - margin
        x_dir, y_dir = (-axis_len_px, 0), (0, -axis_len_px)
        lx_off, ly_off = (-axis_len_px - 22, 18), (-6, -axis_len_px - 10)

    # 畫軸
    cv2.arrowedLine(frame, (ox, oy), (ox + x_dir[0], oy + x_dir[1]), color_x, 2, tipLength=0.12)  # +x
    cv2.arrowedLine(frame, (ox, oy), (ox + y_dir[0], oy + y_dir[1]), color_y, 2, tipLength=0.12)  # +y

    # 軸標
    cv2.putText(frame, "+x", (ox + lx_off[0], oy + lx_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_x, 2)
    cv2.putText(frame, "+y", (ox + ly_off[0], oy + ly_off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_y, 2)

def draw_base_tag_axes_on_image(frame, K, T_aruco2cam, axis_len_m=0.06):
    """
    在影像上畫「基準 ArUco 的 3D 軸」(R=+X, G=+Y, B=+Z)。
    T_aruco2cam: 4x4，把標籤座標點轉到相機座標的變換。
    """
    R = T_aruco2cam[:3, :3]
    t = T_aruco2cam[:3, 3:4]  # (3,1)

    # 標籤座標系下的 4 個點（原點 & 三軸端點）
    P_tag = np.array([
        [0,            0,            0,           1],
        [axis_len_m,   0,            0,           1],
        [0,            axis_len_m,   0,           1],
        [0,            0,            axis_len_m,  1],
    ], dtype=np.float64).T  # (4,4)

    # 轉到相機座標
    T = np.eye(4); T[:3,:3] = R; T[:3,3:4] = t
    P_cam = T @ P_tag
    P_cam = P_cam[:3, :]  # (3,4)

    # 投影到像平面
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    X, Y, Z = P_cam[0,:], P_cam[1,:], P_cam[2,:]
    valid = Z > 1e-6
    u = fx * (X[valid]/Z[valid]) + cx
    v = fy * (Y[valid]/Z[valid]) + cy
    pts = np.stack([u, v], axis=1).astype(int)

    if pts.shape[0] == 4:
        o = (pts[0,0], pts[0,1])
        px = (pts[1,0], pts[1,1])
        py = (pts[2,0], pts[2,1])
        pz = (pts[3,0], pts[3,1])

        cv2.arrowedLine(frame, o, px, (0, 0, 255), 2, tipLength=0.1)   # +X 紅
        cv2.arrowedLine(frame, o, py, (0, 255, 0), 2, tipLength=0.1)   # +Y 綠
        cv2.arrowedLine(frame, o, pz, (255, 0, 0), 2, tipLength=0.1)   # +Z 藍

        cv2.putText(frame, "X", (px[0]+5, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(frame, "Y", (py[0]+5, py[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, "Z", (pz[0]+5, pz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

# ============ 3D 視覺化 ============
plt.ion()
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')
workspace = [-0.4, 0.4, -0.4, 0.4, -0.4, 0.4]

def reset_axes():
    ax3d.cla()
    util.Draw_Aruco(ax3d, marker_length)  # 視覺參考軸
    ax3d.set_xlim3d(workspace[0], workspace[1])
    ax3d.set_ylim3d(workspace[2], workspace[3])
    ax3d.set_zlim3d(workspace[4], workspace[5])
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('Markers in Base-Tag Frame')

# ============ 主迴圈 ============
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        # depth = np.asanyarray(depth_frame.get_data()) * depth_scale  # 本腳本未用到深度

        # 偵測多張 ArUco
        ret, T_cam2aruco_list, T_aruco2cam_list, id_list, corners_list = aruco.Detect_Aruco(
            color, K, D, marker_length,
            aruco_detector.aruco_dict, aruco_detector.aruco_params, True
        )

        reset_axes()

        if not ret or id_list is None or len(id_list) == 0:
            # 也畫影像座標軸，方便對齊
            draw_image_axes_hint(color, corner='top-right', axis_len_px=80)

            cv2.imshow("Aruco Offsets", color)
            plt.show(block=False); plt.pause(0.001)
            if cv2.waitKey(1) == ord('q'): break
            continue

        # 以最小 ID 的標籤為基準
        zipped = list(zip(id_list, T_cam2aruco_list, T_aruco2cam_list, corners_list))
        zipped.sort(key=lambda x: int(x[0]))
        base_id, T_cam2aruco_base, T_base_aruco2cam, _ = zipped[0]


        # === 影像上畫軸 ===
        draw_image_axes_hint(color, corner='top-right', axis_len_px=80)# 影像座標軸
        draw_base_tag_axes_on_image(color, K, T_base_aruco2cam, axis_len_m=0.06)  # 基準標籤 3D 軸投影
        cv2.putText(color, f"Base ID: {int(base_id)}", (20, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 在 3D 圖上畫出基準標籤（作為世界原點）
        draw_frame(ax3d, np.eye(4), length=0.06, label=f'BASE {int(base_id)}')
        ax3d.scatter(0, 0, 0, s=30)

        # 把每張標籤轉到「基準標籤座標系」下並顯示
        console_lines = []
        row = 0
        for _id, T_cam2aruco_i, T_aruco2cam_i, corners in zipped:
            # 以 base 為參考：T_i_in_base = inv(T_cam2aruco_base) @ T_cam2aruco_i
            T_i_in_base = inv_T(T_cam2aruco_base) @ T_cam2aruco_i
            pos = T_i_in_base[:3, 3]
            draw_frame(ax3d, T_i_in_base, length=0.05, label=str(int(_id)))
            ax3d.scatter(pos[0], pos[1], pos[2], s=25)

            # 影像上標示中心點與位移
            pts = corners.reshape(-1, 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            cv2.circle(color, (cx, cy), 6, (0, 255, 0), -1)

            text = f"ID:{int(_id)}  dX:{pos[0]:.3f}  dY:{pos[1]:.3f}  dZ:{pos[2]:.3f} m"
            cv2.putText(color, text, (20, 44 + 28 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
            console_lines.append(text)
            row += 1

        # 印到終端機（可對照實測距離）
        if console_lines:
            print("\n".join(console_lines))

        # 顯示
        cv2.imshow("Aruco Offsets", color)
        plt.show(block=False); plt.pause(0.001)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.ioff()
