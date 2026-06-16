# visualize_skeleton.py
import os
import cv2
import mediapipe as mp

def main():
    video_dir = "./badminton_storke_video"
    video_path = None
    
    if not os.path.exists(video_dir):
        print("❌ 错误：找不到视频目录！")
        return
        
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                break
        if video_path: break

    if not video_path:
        print("❌ 错误：未找到视频文件！")
        return

    print(f"🎬 成功捕获示例视频: {video_path}")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        print("❌ 错误：无法读取视频帧。")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        print("🎉 成功捕捉到骨架！正在使用 [加粗高亮样式] 重新绘制...")
        
        # 1. 自定义核心关键点样式：亮蓝色，半径加大到 6 像素
        landmark_style = mp_drawing.DrawingSpec(
            color=(255, 255, 0),       # 亮蓝色 (BGR 格式)
            thickness=6, 
            circle_radius=5
        )
        
        # 2. 自定义骨骼连接线样式：高亮纯黄色，线宽强行加粗到 5 像素（防撞色）
        connection_style = mp_drawing.DrawingSpec(
            color=(0, 255, 255),       # 纯黄色 (BGR 格式)，在绿色球场上超级显眼！
            thickness=5, 
            circle_radius=1
        )
        
        # 3. 绘制到画面上
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_style=connection_style
        )
        
        output_image_path = "./mediapipe_skeleton_result.png"
        cv2.imwrite(output_image_path, frame)
        print(f"📸 亮黄色加粗骨架图已重新保存至: {output_image_path}")
    else:
        print("❌ 该帧未识别到有效骨架。")

if __name__ == "__main__":
    main()