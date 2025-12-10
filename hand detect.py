from drv_modbus import send
from drv_modbus import request
from pymodbus.client import ModbusTcpClient
import cv2
import mediapipe as mp
import math
import time

# 建立通訊連線
c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
c.connect()
x , y , z , rx, ry, rz = request.Get_TCP_Pose(c)
# 初始化 MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 計算兩向量角度
def vector_2d_angle(v1, v2):
    try:
        angle = math.degrees(
            math.acos(
                (v1[0]*v2[0] + v1[1]*v2[1]) /
                (((v1[0]**2 + v1[1]**2)**0.5) * ((v2[0]**2 + v2[1]**2)**0.5))
            )
        )
    except:
        angle = 180
    return angle

# 計算每根手指的角度
def hand_angle(hand_):
    angle_list = []
    finger_indices = [(0, 2, 3, 4), (0, 6, 7, 8), (0, 10, 11, 12), (0, 14, 15, 16), (0, 18, 19, 20)]
    for idx in finger_indices:
        angle = vector_2d_angle(
            (hand_[idx[0]][0] - hand_[idx[1]][0], hand_[idx[0]][1] - hand_[idx[1]][1]),
            (hand_[idx[2]][0] - hand_[idx[3]][0], hand_[idx[2]][1] - hand_[idx[3]][1])
        )
        angle_list.append(angle)
    return angle_list

# 根據手指角度判斷手勢
def hand_pos(finger_angle):
    f1, f2, f3, f4, f5 = finger_angle
    if f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '1'  #Red
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '2'  # Blue
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 > 50:
        return '3'  # Green
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '4'  #X+
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '5' #X- 
    elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return '6' # Y+
    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return 'good' 
    elif f1>=50 and f2>=50 and f3<50 and f4>=50 and f5>=50:
        return 'no!!!'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5<50:
        return 'ROCK!'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return '0' # Gasp Off
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return 'pink'
    elif f1>=50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1<50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '7' # Y-
    elif f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '8'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5>=50:
        return '9'
    else:
        return 'none'

# gesture debounce 設定
last_gesture = None
gesture_repeat_count = 0
gesture_threshold = 2

# 範圍限制區域
LIMIT_AREA = {
    'x_min': 340,
    'x_max': 510,
    'y_min': -88,
    'y_max': 180
}

# 初始機械手臂位置
current_position = {'x': x, 'y': y}

# 是否鎖定動作（避免重複執行1/2/3）
locked = False

# 包裝 Jog_Function，限制範圍內移動
def Jog_Position_Limited(c, dx, dy, dz, drx, dry, drz):
    # 重新從機械手臂讀取實際位置
    x, y, z, rx, ry, rz = request.Get_TCP_Pose(c)

    next_x = x + dx * 5
    next_y = y + dy * 5

    if LIMIT_AREA['x_min'] <= next_x <= LIMIT_AREA['x_max'] and \
       LIMIT_AREA['y_min'] <= next_y <= LIMIT_AREA['y_max']:
        send.Jog_Position(c, dx, dy, dz, drx, dry, drz)
    else:
        send.Jog_Stop(c)



# 啟動攝影機
cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA

# 初始位置移動一次（可選）
send.Grasp_OFF(c)
send.Grasp_ON(c) 
send.Go_Position(c, 565 , 310 , 497 , 180 , 0 , 225 , 100) 
send.Go_Position(c, 565 , 310 , 300 , 180 , 0 , 225 , 100) #選筆初始位置


# 啟用手部追蹤

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    last_locked_gesture = None
    limit_area_enabled = False
    while True:
        key = cv2.waitKey(5)
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (540, 310))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture = 'none'
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = [(lm.x * 540, lm.y * 310) for lm in hand_landmarks.landmark]
                finger_angle = hand_angle(finger_points)
                gesture = hand_pos(finger_angle)

                # 顯示手勢名稱
                cv2.putText(img, gesture, (30, 120), fontFace, 5, (255, 255, 255), 10, lineType)

                # Debounce 處理
                if gesture == last_gesture:
                    gesture_repeat_count += 1
                else:
                    gesture_repeat_count = 1
                    last_gesture = gesture
        if gesture_repeat_count >= gesture_threshold:
            if not locked:
                if gesture in ['1', '2', '3'] and gesture != last_locked_gesture:
                    locked = True
                    last_locked_gesture = gesture
                    send.Grasp_OFF(c)
                    send.Grasp_ON(c)

                    if gesture == '1':
                        send.Go_Position(c, 332 , 310 , 310.0 , 180, 0, 225, 100, mov=1)
                    elif gesture == '2':
                        send.Go_Position(c, 405., 310, 310.0, 180, 0, 225, 100, mov=1)
                    elif gesture == '3':
                        send.Go_Position(c, 475., 310, 310.0, 180, 0, 225, 100, mov=1)

                    send.Grasp_OFF(c)

            if gesture == 'ROCK!':
                locked = False
                last_locked_gesture = None
                print("解鎖")

            elif gesture == '8':
                x, y, z, rx, ry, rz = request.Get_TCP_Pose(c)
                send.Go_Position(c, x , y, 510, rx, ry, rz, 100)
                send.Go_Position(c, 350, -90, 500, 180, 0, 45, 100)
                limit_area_enabled = True
                print("範圍限制已啟用，可開始畫圖")

        # 👇 這段建議獨立出來
        if locked and limit_area_enabled:
            if gesture == '4':
                print(request.Get_TCP_Pose(c))
                Jog_Position_Limited(c, 1, 0, 0, 0, 0, 0)
            elif gesture == '5':
                print(request.Get_TCP_Pose(c))
                Jog_Position_Limited(c, -1, 0, 0, 0, 0, 0)
            elif gesture == '6':
                print(request.Get_TCP_Pose(c))
                Jog_Position_Limited(c, 0, 1, 0, 0, 0, 0)
            elif gesture == '7':
                print(request.Get_TCP_Pose(c))
                Jog_Position_Limited(c, 0, -1, 0, 0, 0, 0)
            else:
                send.Jog_Stop(c)



        cv2.imshow('Hand Gesture Control', img)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
send.Jog_Stop(c)