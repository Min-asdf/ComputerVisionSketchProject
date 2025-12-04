import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tkinter import filedialog, Tk

# ==========================================
# [설정] 모델 경로 및 이미지 크기
# ==========================================
# 현재 파일 위치 기준 경로 설정
try:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_PATH = os.getcwd()

# 학습된 모델 파일 경로 (파일 이름 확인 필수!)
MODEL_PATH = os.path.join(BASE_PATH, 'models', 'sketch_unet_best.h5')
IMG_SIZE = 256

# ==========================================
# [1] 모델 로드 함수
# ==========================================
def load_trained_model():
    print("⏳ 모델을 불러오는 중입니다...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 모델 파일이 없습니다! ({MODEL_PATH})")
        return None
    
    # compile=False: 학습을 더 안 할 것이므로 최적화 정보 로드 생략 (속도 향상 & 오류 방지)
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ 모델 로드 성공!")
        return model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

# ==========================================
# [2] 이미지 전처리 및 예측 함수
# ==========================================
def predict_image(model, img_path):
    # 1. 이미지 읽기 (흑백)
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print("❌ 이미지를 읽을 수 없습니다.")
        return

    # 2. 전처리 (학습 때와 똑같이 맞춰줘야 함)
    # 크기 조절
    img_resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    
    # 정규화 (0~1)
    input_data = img_resized.astype(np.float32) / 255.0
    
    # 차원 확장: (H, W) -> (H, W, 1) -> (H, W, 3) -> (1, H, W, 3)
    input_data = np.expand_dims(input_data, axis=-1)       # 채널 차원 추가
    input_data = np.repeat(input_data, 3, axis=-1)         # 3채널로 복사 (모델 입력 형태)
    input_data = np.expand_dims(input_data, axis=0)        # 배치 차원 추가

    # 3. 예측 (Inference)
    print("🖌️ AI가 그림을 깨끗하게 만드는 중...")
    pred = model.predict(input_data)
    
    # 4. 후처리 (0~1 -> 0~255)
    result_img = (pred[0] * 255).astype(np.uint8)

    # ==========================================
    # [3] 결과 시각화 (팝업창)
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # 원본 (Resize됨)
    plt.subplot(1, 2, 1)
    plt.title("Input (Noisy Sketch)")
    plt.imshow(img_resized, cmap='gray')
    plt.axis('off')

    # 결과물
    plt.subplot(1, 2, 2)
    plt.title("AI Output (Cleaned)")
    plt.imshow(result_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("✨ 변환 완료!")

# ==========================================
# [4] 메인 실행 (파일 선택 창)
# ==========================================
if __name__ == '__main__':
    # 1. 모델 로드
    model = load_trained_model()

    if model:
        # 2. 윈도우 파일 선택 창 띄우기
        print("📂 변환할 스케치 이미지를 선택하세요...")
        root = Tk()
        root.withdraw() # 빈 윈도우 숨김
        root.attributes('-topmost', True) # 창을 맨 앞으로
        
        file_path = filedialog.askopenfilename(
            title="변환할 스케치 이미지 선택",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if file_path:
            print(f"선택된 파일: {file_path}")
            predict_image(model, file_path)
        else:
            print("취소되었습니다.")
