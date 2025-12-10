import cv2
import math
import numpy as np
import os

# ============ 全域變數 ============
points = []        # 存每次兩個點
ratios = []        # 儲存多次量測結果
image_display = None


# ============ 滑鼠事件函式 ============
def mouse_click(event, x, y, flags, param):
    global points, ratios, image_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image_display)

        # 當點滿兩個點後，畫線 + 計算
        if len(points) == 2:
            cv2.line(image_display, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Image", image_display)

            px_dist = math.dist(points[0], points[1])
            print(f"\n🔹 選取的兩點像素距離：{px_dist:.2f} px")

            try:
                real_mm = float(input("請輸入這兩點的實際距離 (mm)："))
                ratio = px_dist / real_mm
                ratios.append(ratio)

                print(f"👉 每毫米像素數(px/mm) = {ratio:.4f}")
                print(f"👉 每像素實際長度(mm/px) = {1/ratio:.4f}")

                if len(ratios) > 1:
                    avg_ratio = sum(ratios) / len(ratios)
                    print(f"📊 目前 {len(ratios)} 次量測平均：{avg_ratio:.4f} px/mm")

            except ValueError:
                print("⚠️ 輸入的距離不是數字，請重新點選。")

            points.clear()  # 清空以便下一次量測


# ============ 影像讀取函式（支援中文路徑） ============
def read_image_safely():
    while True:
        img_path = input("請輸入影像路徑（或直接拖曳圖片到此視窗後按 Enter）：").strip('"')

        if not img_path:
            print("⚠️ 沒輸入路徑，請重新輸入。")
            continue

        if not os.path.exists(img_path):
            print("❌ 找不到檔案，請檢查路徑。")
            continue

        # 用 np.fromfile + imdecode 支援中文檔案路徑
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            print("❌ 無法讀取影像內容，請確認影像未損壞。")
            continue

        print(f"✅ 成功讀取影像：{os.path.basename(img_path)}")
        return image


# ============ 主程式 ============
if __name__ == "__main__":
    image_display = read_image_safely()
    cv2.imshow("Image", image_display)
    cv2.setMouseCallback("Image", mouse_click)

    print("\n📏 使用說明：")
    print("1️⃣ 左鍵點兩下端點以量測距離。")
    print("2️⃣ 終端會要求輸入實際長度（mm）。")
    print("3️⃣ 可多次量測，會自動顯示平均值。")
    print("4️⃣ 按 ESC 結束程式。\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n✅ 平均像素比例：{avg_ratio:.4f} px/mm")
        print(f"✅ 平均實際比例：{1/avg_ratio:.4f} mm/px")
    else:
        print("\n⚠️ 未進行任何量測。")

    cv2.destroyAllWindows()
