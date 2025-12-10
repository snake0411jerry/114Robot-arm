import cv2
import numpy as np
import pyrealsense2 as rs

# RealSense Configuration
RS_WIDTH = 1280 # Use your actual RS_WIDTH
RS_HEIGHT = 720 # Use your actual RS_HEIGHT
RS_FPS = 30     # Use your actual RS_FPS

def nothing(x):
    pass

if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
    
    try:
        pipeline.start(config)
        print("RealSense camera started. Adjust HSV values to find the right range.")

        cv2.namedWindow("Image")
        cv2.namedWindow("Mask")

        # Create trackbars for HSV adjustment
        cv2.createTrackbar("LH", "Mask", 90, 179, nothing) # Lower Hue
        cv2.createTrackbar("LS", "Mask", 60, 255, nothing) # Lower Saturation
        cv2.createTrackbar("LV", "Mask", 60, 255, nothing) # Lower Value
        cv2.createTrackbar("UH", "Mask", 140, 179, nothing) # Upper Hue
        cv2.createTrackbar("US", "Mask", 255, 255, nothing) # Upper Saturation
        cv2.createTrackbar("UV", "Mask", 255, 255, nothing) # Upper Value

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            # Rotate for consistency if your main program does
            frame = cv2.rotate(frame, cv2.ROTATE_180) 
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Get current positions of trackbars
            lh = cv2.getTrackbarPos("LH", "Mask")
            ls = cv2.getTrackbarPos("LS", "Mask")
            lv = cv2.getTrackbarPos("LV", "Mask")
            uh = cv2.getTrackbarPos("UH", "Mask")
            us = cv2.getTrackbarPos("US", "Mask")
            uv = cv2.getTrackbarPos("UV", "Mask")

            lower_bound = np.array([lh, ls, lv])
            upper_bound = np.array([uh, us, uv])

            # Create a mask using the HSV range
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Display images
            cv2.imshow("Image", frame)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'): # Print current HSV values
                print(f"Lower H:{lh}, S:{ls}, V:{lv}")
                print(f"Upper H:{uh}, S:{us}, V:{uv}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped and windows closed.")