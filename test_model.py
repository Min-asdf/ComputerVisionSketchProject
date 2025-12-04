import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tkinter import filedialog
import tkinter as tk

# ==========================================
# [설정] 경로 및 이미지 크기
# ==========================================
# 학습 때 저장한 모델 경로
MODEL_PATH = './models/sketch_unet_best.h5' 
IMG_SIZE = 256

# ==========================================
# [함수] 이미지 전처리 (학습 때와 똑같이 맞춰야 함)
# ==========================================
def preprocess_image(img_path):
    # 1. 이미지 읽기 (흑백)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    # 2. 리사이즈 (256x256)
    original = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 3. 정규화 (0~1)
    input_tensor = original.astype(np.float32) / 255.0
    
    # 4. 차원 확장 (학습 모델 입력이 3채널이었으므로 흑백 -> RGB로 복제)
    # (H, W) -> (H, W, 1) -> (H, W, 3) -> (1, H, W, 3)
    input_tensor = np.expand_dims(input_tensor, axis=-1)
    input_tensor = np.repeat(input_tensor, 3, axis=-1) 
    input_tensor = np.expand_dims(input_tensor, axis=0) # 배치 차원 추가
    
    return original, input_tensor

# ==========================================
# [메인] 실행
# ==========================================
if __name__ == '__main__':
    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        print("학습 코드를 먼저 실행해 주세요.")
        exit()
        
    print(f"📂 모델 로드 중... ({MODEL_PATH})")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ 모델 로드 완료!")

    # 2. 테스트할 이미지 선택 (파일 탐색기 열기)
    print("🖼️ 테스트할 이미지를 선택하세요...")
    root = tk.Tk()
    root.withdraw() # 빈 창 숨기기
    file_path = filedialog.askopenfilename(
        title="테스트할 이미지 선택",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        print("취소되었습니다.")
    else:
        # 3. 전처리 및 예측
        original_img, input_tensor = preprocess_image(file_path)
        
        if original_img is not None:
            print("🔮 변환 중...")
            pred = model.predict(input_tensor)
            
            # 4. 결과 후처리 (0~1 -> 0~255)
            result_img = (pred[0] * 255).astype(np.uint8)
            
            # 5. 시각화 (원본 vs 결과)
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.title("Input (Original/Noisy)")
            plt.imshow(original_img, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Output (AI Cleaned)")
            plt.imshow(result_img, cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # 결과 저장 (선택 사항)
            cv2.imwrite('result_output.png', result_img)
            print("✅ 변환 완료! 'result_output.png'로 저장되었습니다.")
            
        else:
            print("❌ 이미지를 읽을 수 없습니다.")