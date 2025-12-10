from landmark import aruco
from landmark import util
import numpy as np
import cv2
import matplotlib.pyplot as plt

# The ID here is set to 1, not 24. I've renamed the variable for clarity.
aruco_5x5_100_id_1 = aruco.Aruco(aruco.ARUCO_DICT().DICT_5X5_100, 1, 300)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
aruco_length = 0.08 # Marker side length in meters

# Camera intrinsics and distortion coefficients
K = np.array([[739.0986938476562, 0.0, 660.466777227186], 
              [0.0, 737.6295166015625, 371.63861588193686], 
              [0.0, 0.0, 1.0]])
D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# 3D plot setup
Workspace = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

while True:
    plt.cla()
    ret, frame = cap.read()
    if not ret:
        break
        
    # --- Detect all ArUco markers in the frame ---
    ret, T_cam_to_aruco_result, T_aruco_to_cam_result, id_result, corner_result = aruco.Detect_Aruco(
        frame, K, D, aruco_length, 
        aruco_5x5_100_id_1.aruco_dict, 
        aruco_5x5_100_id_1.aruco_params, 
        True # This draws the default green box and axes on markers
    )
    
    # If any markers were detected, process each one
    if ret and id_result is not None:
        # Loop through every detected marker
        for marker_id, T, corners in zip(id_result, T_aruco_to_cam_result, corner_result):
            # T is the transformation from the marker's coordinate system to the camera's
            # t_aruco_to_cam is the position of the marker in the camera's frame
            R_aruco_to_cam, t_aruco_to_cam = util.T_to_R_and_t(T)
            
            # --- Get the coordinates (in meters) ---
            x_m = t_aruco_to_cam[0][0]
            y_m = t_aruco_to_cam[1][0]
            z_m = t_aruco_to_cam[2][0]

            # --- Prepare the text to display on the video frame ---
            # Format the coordinates string
            coord_text = f"X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f} m"
            id_text = f"ID: {marker_id[0]}"

            # Get the top-left corner of the marker to position the text
            top_left_corner = (int(corners[0][0][0]), int(corners[0][0][1]))
            
            # --- Draw the text on the frame ---
            cv2.putText(frame, id_text, (top_left_corner[0], top_left_corner[1] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, coord_text, (top_left_corner[0], top_left_corner[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # This part still only draws the 3D camera pose for the specific ID=1 marker
            if marker_id[0] == aruco_5x5_100_id_1.id:
                util.Draw_Camera(K, R_aruco_to_cam, t_aruco_to_cam, coord_text, ax, f=0.08)
    
    # Update the 3D plot
    util.Draw_Aruco(ax, aruco_length)
    ax.set_xlim3d(Workspace[0], Workspace[1])
    ax.set_ylim3d(Workspace[2], Workspace[3])
    ax.set_zlim3d(Workspace[4], Workspace[5])
    ax.set_xlabel('x')
    ax.set_ylabel('y') 
    ax.set_zlabel('z')
    plt.show(block=False)
    plt.pause(0.001)
    
    # Show the final video frame with annotations
    cv2.imshow("Aruco Coordinate Detection", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.close(fig)