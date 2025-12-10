import cv2
import numpy as np

def detect_tableware_centroid(image_source=0):
    """
    偵測桌面餐具並計算質心
    image_source: 0 代表預設攝影機，或是傳入圖片路徑字串
    """
    
    # 1. 讀取影像
    if isinstance(image_source, int):
        cap = cv2.VideoCapture(image_source)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("無法讀取攝影機")
            return
    else:
        frame = cv2.imread(image_source)
        if frame is None:
            print("找不到圖片路徑")
            return

    # 複製一份影像用於顯示結果
    output_img = frame.copy()

    # 2. 影像前處理
    # 轉為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊 (去除高頻噪訊，這對於金屬反光很有幫助)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 邊緣檢測 (Canny)
    # 參數 30 和 150 是閾值，需根據現場光線調整
    # 針對灰色桌面，如果餐具反光強，Canny 效果通常不錯
    edges = cv2.Canny(blurred, 30, 150)

    # 進行膨脹 (Dilation) 與 侵蝕 (Erosion) 以連接斷開的邊緣 (Morphological Close)
    kernel = np.ones((5,5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 4. 尋找輪廓
    # RETR_EXTERNAL 只找最外層輪廓，忽略餐具內部的紋路
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"偵測到 {len(contours)} 個潛在物體")

    detected_objects = []

    for i, cnt in enumerate(contours):
        # 5. 過濾雜訊
        area = cv2.contourArea(cnt)
        if area < 1000:  # 過濾掉太小的面積 (例如灰塵或光斑)
            continue

        # 6. 計算影像矩與質心
        M = cv2.moments(cnt)
        
        # 避免除以零 (雖然前面有面積過濾，但在數學上要嚴謹)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # 儲存數據
        obj_info = {
            "id": i,
            "centroid": (cX, cY),
            "area": area
        }
        detected_objects.append(obj_info)

        # 7. 視覺化繪圖
        # 畫出輪廓 (綠色)
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
        # 畫出質心 (紅色圓點)
        cv2.circle(output_img, (cX, cY), 7, (0, 0, 255), -1)
        # 顯示座標文字
        cv2.putText(output_img, f"({cX}, {cY})", (cX - 20, cY - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        print(f"物體 ID: {i} | 質心座標: ({cX}, {cY}) | 面積: {area}")

    # 顯示結果
    cv2.imshow("Original", frame)
    cv2.imshow("Edges", edges_closed) # 觀察邊緣處理是否乾淨
    cv2.imshow("Result", output_img)
    
    print("按任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 執行程式 (若有圖片可換成 'table.jpg')
# detect_tableware_centroid('path_to_your_image.jpg') 
detect_tableware_centroid(r"C:\Users\User\Pictures\Screenshots\Screenshot 2025-11-19 155251.png") # 使用 Webcam