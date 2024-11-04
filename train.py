import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import model

# Mediapipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# 데이터셋 전처리 클래스 정의
class HandGestureDataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        total_files = sum([len(files) for _, _, files in os.walk(self.dataset_folder)])  # 전체 파일 개수 계산
        processed_files = 0
        
        for label in os.listdir(self.dataset_folder):
            label_path = os.path.join(self.dataset_folder, label)
            if os.path.isdir(label_path) and label in {'0', '1', '2', '3'}:  # 0, 1, 2, 3으로 조정
                for file_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, file_name)
                    landmarks = self.extract_landmarks(image_path)
                    if landmarks is not None:
                        self.data.append(landmarks)
                        self.labels.append(int(label))

                    # 진행 상황 출력
                    processed_files += 1
                    print(f"데이터 변환 진행률: {processed_files/total_files*100:.2f}% 완료", end="\r")

        print(f"\n데이터 로드 완료: 총 {len(self.data)}개의 샘플")

    def extract_landmarks(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            joint = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
            base_x, base_y = joint[0]
            relative_joint = joint - [base_x, base_y]
            return relative_joint.reshape(21, 2)  # 21개의 랜드마크를 (21, 2) 형태로 반환
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = data.astype(np.float32)
        data = torch.tensor(data).unsqueeze(0)  # CNN 입력을 위한 1채널 추가
        label = self.labels[idx]
        return data, label

# CNN 모델 정의


# 데이터셋 및 데이터로더 생성
dataset = HandGestureDataset('data_augmented_2')
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델 초기화
model = model.CNNHandGestureModel()

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 실시간 손실 그래프 설정
plt.ion()
fig, ax = plt.subplots()
losses = []
line, = ax.plot([], [], label="Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss over Epochs")
ax.legend()

# 모델 학습
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for data, labels in train_loader:
        # 순전파
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # 역전파 및 옵티마이저 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 에포크당 평균 손실 계산 및 실시간 업데이트
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    
    # 실시간 손실 그래프 업데이트
    line.set_xdata(range(len(losses)))
    line.set_ydata(losses)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print("모델 학습 완료")
plt.ioff()

# 모델 저장 함수
def save_model_with_incremental_name(base_name):
    i = 1
    file_name = f"{base_name}.pth"
    while os.path.exists(file_name):
        file_name = f"{base_name}_{i}.pth"
        i += 1
    torch.save(model.state_dict(), file_name)
    print(f"모델 저장 완료: {file_name}")

# 모델 저장
save_model_with_incremental_name('cnn_hand_gesture_model_gray_deeper')
