# preprocess.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import mediapipe as mp

print("正在载入环境，激活完整版 MediaPipe 0.10.14 官方标准原生接口...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
print("🚀 MediaPipe Pose 骨架核心已全功能就绪！")

# 严格匹配你左侧文件树里的文件夹名称
LABEL_MAP = {
    "forehand_drive": 0, "forehand_lift": 1, "forehand_net_shot": 2,
    "forehand_clear": 3, "backhand_drive": 4, "backhand_net_shot": 5
}

def normalize_skeleton(landmarks):
    """ 严格落实任务书：骨架空间归一化 """
    coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]) 
    left_hip = coords[23, :3]
    right_hip = coords[24, :3]
    hip_center = (left_hip + right_hip) / 2.0
    coords[:, :3] = coords[:, :3] - hip_center
    
    left_shoulder = coords[11, :3]
    right_shoulder = coords[12, :3]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width > 1e-5:
        coords[:, :3] = coords[:, :3] / shoulder_width
    return coords.flatten()

def extract_skeleton_from_video(video_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            frames_features.append(normalize_skeleton(results.pose_landmarks.landmark))
        else:
            frames_features.append([0.0] * 132)
    cap.release()
    
    if len(frames_features) == 0: 
        return None
    frames_features = np.array(frames_features)
    idx = np.linspace(0, len(frames_features) - 1, target_frames).astype(int)
    return frames_features[idx]

def main():
    # 指向你的羽毛球视频独立主文件夹
    DATA_DIR = "./badminton_storke_video" 
    X, y = [], []
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 找不到视频目录 '{DATA_DIR}'，请检查目录层级！")
        return

    print("开始利用官方 MediaPipe 逐帧提取真实的动作骨架序列...")
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path) and folder_name in LABEL_MAP:
            label = LABEL_MAP[folder_name]
            print(f"👉 正在处理类别: {folder_name} (标签: {label})")
            for video_name in os.listdir(folder_path):
                if video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(folder_path, video_name)
                    features = extract_skeleton_from_video(video_path, target_frames=30)
                    if features is not None:
                        X.append(features)
                        y.append(label)

    if len(X) == 0:
        print("❌ 未提取到任何有效特征，请检查视频文件。")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"\n✅ 真实骨架特征提取成功！总样本数: {len(X)}，矩阵形状: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs("./data", exist_ok=True)
    np.save("./data/X_train.npy", X_train)
    np.save("./data/y_train.npy", y_train)
    np.save("./data/X_test.npy", X_test)
    np.save("./data/y_test.npy", y_test)
    print("✨ .npy 特征数据已完美固化到 ./data/ 目录下！")

if __name__ == "__main__":
    main()