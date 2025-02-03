import os
import cv2
import mediapipe as mp
import time

video_path = r'E:\Udacity_DL\Pytorch_Learning\Learning\Optimization\Statstics\video.mp4'
if not os.path.exists(video_path):
    print(f"Error: The video file '{video_path}' does not exist.")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1, smooth_landmarks=True)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'.")
    print("Possible reasons: Unsupported codec, corrupted file, or invalid path.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Use absolute path for output
output_path = os.path.abspath('output.mp4')  # Changed to AVI for testing
fourcc = cv2.VideoWriter_fourcc(*'X264')  # Changed codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# Check if VideoWriter initialized successfully
if not out.isOpened():
    print(f"Error: Could not initialize VideoWriter at '{output_path}'.")
    print("Check codec, file extension, and directory permissions.")
    cap.release()
    exit()
else:
    print(
        f"VideoWriter initialized successfully at '{output_path}' with codec '{fourcc}' and resolution {(frame_width, frame_height)}.")
prev_time = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        if frame is None:
            print("Warning: Empty frame detected.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time -
                                              prev_time) != 0 else 0
        prev_time = curr_time

        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(annotated_frame)

        cv2.imshow('Pose Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Ensure resources are released even if an error occurs
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"Processing completed. Output saved to: {output_path}")
