import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# ====================================================
# [0] GPU 설정
# ====================================================
# 윈도우 TF 2.10 환경에서 GPU 메모리 증가 허용
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print(f"GPU 감지됨: {len(gpus)}개 사용 가능")
        print(f"장치명: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("GPU가 감지되지 않았습니다. CPU 모드로 실행합니다.")

# ====================================================
# [1] 설정: 경로 및 하이퍼파라미터
# ====================================================
# 현재 파일 위치를 기준으로 경로를 설정
try:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_PATH = os.getcwd()

CLEAN_IMG_PATH = os.path.join(BASE_PATH, 'clean_images') # 원본 이미지 폴더
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')        # 생성된 데이터셋 저장소
MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'models')      # 모델 저장소

IMG_SIZE = 256  # 이미지 크기 (GPU 메모리 부족 시 128로 줄이세요)
BATCH_SIZE = 4  # 배치 크기 (OOM 발생 시 2로 줄이세요)
EPOCHS = 50     # 학습 횟수

# 폴더 자동 생성
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ====================================================
# [2] 노이즈 생성 함수들 (OpenCV)
# ====================================================
def add_noise_line(img):
    img_copy = img.copy()
    h, w = img_copy.shape
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 0), random.randint(1, 3))
    return img_copy

def add_noise_cut(img):
    img_copy = img.copy()
    h, w = img_copy.shape
    for _ in range(random.randint(10, 30)):
        x, y = random.randint(0, w-20), random.randint(0, h-20)
        cv2.rectangle(img_copy, (x, y), (x+random.randint(5, 20), y+random.randint(5, 20)), (255, 255, 255), -1)
    return img_copy

def add_noise_thickness(img):
    kernel = np.ones((3,3), np.uint8)
    inverted_img = cv2.bitwise_not(img)
    dilated_img = cv2.dilate(inverted_img, kernel, iterations=1)
    return cv2.bitwise_not(dilated_img)

def add_noise_skew(img):
    rows, cols = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    px, py = random.randint(-20, 20), random.randint(-20, 20)
    pts2 = np.float32([[50+px, 50+py], [200+px, 50-py], [50-px, 200+py]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

# ====================================================
# [3] 데이터셋 생성 및 로드
# ====================================================
def generate_dataset():
    print("\n[1/4] 데이터셋 생성 시작...")
    if not os.path.exists(CLEAN_IMG_PATH):
        print(f"오류: '{CLEAN_IMG_PATH}' 폴더가 없습니다.")
        return False

    clean_files = [f for f in os.listdir(CLEAN_IMG_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not clean_files:
        print(f"오류: '{CLEAN_IMG_PATH}' 폴더에 이미지가 없습니다. 스케치 이미지를 넣어주세요.")
        return False

    noise_functions = [add_noise_line, add_noise_cut, add_noise_thickness, add_noise_skew]
    count = 0
    
    for img_name in clean_files:
        img_path = os.path.join(CLEAN_IMG_PATH, img_name)
        clean_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if clean_img is None: continue
        
        # 리사이즈
        clean_img = cv2.resize(clean_img, (IMG_SIZE, IMG_SIZE))
        
        # 저장 폴더 (sketch_01, sketch_02...)
        base_name = os.path.splitext(img_name)[0]
        save_dir = os.path.join(DATASET_PATH, base_name)
        os.makedirs(save_dir, exist_ok=True)

        # 정답 저장
        cv2.imwrite(os.path.join(save_dir, 'clean.png'), clean_img)

        # 노이즈 저장
        for func in noise_functions:
            noisy_img = func(clean_img)
            fname = func.__name__.replace('add_', '') + '.png'
            cv2.imwrite(os.path.join(save_dir, fname), noisy_img)
        count += 1
        
    print(f"총 {count}세트({count*4}장) 데이터 생성 완료.")
    return True

def load_dataset_paths():
    X_paths = []
    Y_paths = []
    noise_types = ['noise_line.png', 'noise_cut.png', 'noise_thickness.png', 'noise_skew.png']

    for folder in os.listdir(DATASET_PATH):
        folder_full = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_full):
            clean_p = os.path.join(folder_full, 'clean.png')
            if os.path.exists(clean_p):
                for n_file in noise_types:
                    noisy_p = os.path.join(folder_full, n_file)
                    if os.path.exists(noisy_p):
                        X_paths.append(noisy_p)
                        Y_paths.append(clean_p)
    return X_paths, Y_paths

def load_image_tf(noisy_path, clean_path):
    # 파일 읽기 및 디코딩
    noisy = tf.io.read_file(noisy_path)
    noisy = tf.image.decode_png(noisy, channels=1)
    clean = tf.io.read_file(clean_path)
    clean = tf.image.decode_png(clean, channels=1)
    
    # 정규화 (0~1)
    noisy = tf.cast(noisy, tf.float32) / 255.0
    clean = tf.cast(clean, tf.float32) / 255.0
    
    # Diffusion Style 입력을 위해 3채널로 복제 (흑백 -> RGB 형태)
    noisy = tf.image.grayscale_to_rgb(noisy)
    return noisy, clean

# ====================================================
# [4] Diffusion Style U-Net 모델 (ResBlock + Attention)
# ====================================================
def ResBlock(x, filters):
    shortcut = x
    # 채널 수가 다르면 1x1 Conv로 맞춰줌
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(x)
    
    # 첫 번째 컨볼루션
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)  # <--- GroupNormalization 대신 이거 사용!
    x = layers.Activation("swish")(x)
    
    # 두 번째 컨볼루션
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)  # <--- 여기도 변경!

    # 입력값과 결과값을 더함 (Residual Connection)
    return layers.Add()([x, shortcut])

def AttentionBlock(x):
    channels = x.shape[-1]
    q = layers.Conv2D(channels // 2, 1)(x)
    k = layers.Conv2D(channels // 2, 1)(x)
    v = layers.Conv2D(channels // 2, 1)(x)
    
    attn = layers.Multiply()([q, k])
    attn = layers.Activation("softmax")(attn)
    out = layers.Multiply()([attn, v])
    out = layers.Conv2D(channels, 1)(out)
    return layers.Add()([x, out])

def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    x1 = layers.Conv2D(32, 3, padding='same')(inputs)
    x1 = ResBlock(x1, 32)
    p1 = layers.MaxPooling2D(2)(x1)

    x2 = ResBlock(p1, 64)
    x2 = AttentionBlock(x2)
    p2 = layers.MaxPooling2D(2)(x2)

    x3 = ResBlock(p2, 128)
    x3 = AttentionBlock(x3)
    p3 = layers.MaxPooling2D(2)(x3)

    # Bottleneck
    b = ResBlock(p3, 256)
    b = AttentionBlock(b)
    b = ResBlock(b, 256)

    # Decoder
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Concatenate()([u1, x3])
    u1 = ResBlock(u1, 128)
    u1 = AttentionBlock(u1)

    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Concatenate()([u2, x2])
    u2 = ResBlock(u2, 64)
    u2 = AttentionBlock(u2)

    u3 = layers.UpSampling2D(2)(u2)
    u3 = layers.Concatenate()([u3, x1])
    u3 = ResBlock(u3, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u3)
    return models.Model(inputs, outputs)

# ====================================================
# [5] 메인 실행
# ====================================================
if __name__ == '__main__':
    # 1. 데이터 생성
    if generate_dataset():
        X_paths, Y_paths = load_dataset_paths()
        
        if len(X_paths) > 0:
            # 2. 데이터 파이프라인 구축
            print("\n[2/4] 데이터 파이프라인 구축 중...")
            dataset = tf.data.Dataset.from_tensor_slices((X_paths, Y_paths))
            dataset = dataset.map(load_image_tf)
            dataset = dataset.shuffle(400).batch(BATCH_SIZE)
            
            # 3. 모델 구축 및 학습
            print("\n[3/4] 모델 구축 및 학습 시작...")
            model = build_model()
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # (1) Best Model: 성능(loss)이 가장 좋을 때만 덮어쓰기
            best_save_path = os.path.join(MODEL_SAVE_PATH, 'sketch_unet_best.h5')
            checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
                best_save_path, 
                save_best_only=True, 
                monitor='loss', 
                verbose=1
            )

            # (2) Final/Latest Model: 매 Epoch마다 무조건 저장 (가장 최신 상태 유지)
            #     'save_best_only=False'로 설정하면 매번 덮어씁니다.
            final_save_path = os.path.join(MODEL_SAVE_PATH, 'sketch_unet_final.h5')
            checkpoint_final = tf.keras.callbacks.ModelCheckpoint(
                final_save_path, 
                save_best_only=False,  # <--- 핵심! 무조건 저장
                verbose=0              # 매번 출력하면 시끄러우니 0으로 설정
            )

            # callbacks 리스트에 둘 다 추가
            history = model.fit(
                dataset, 
                epochs=EPOCHS, 
                callbacks=[checkpoint_best, checkpoint_final]
            )
            
            print(f"\n학습 완료!")
            print(f"   - 최고 성능 모델: {best_save_path}")
            print(f"   - 최종 학습 모델: {final_save_path}")

            # 4. 결과 확인 (첫 번째 데이터로 테스트)
            print("\n[4/4] 학습 결과 테스트...")
            
            # [수정] save_path 대신 best_save_path 사용
            if os.path.exists(best_save_path):
                best_model = tf.keras.models.load_model(best_save_path)
                print(f"최적 모델 로드 성공: {best_save_path}")
            else:
                print("최적 모델을 찾을 수 없어 학습된 마지막 상태(model)를 사용합니다.")
                best_model = model
            
            test_noisy_path = X_paths[0]
            test_img_raw = cv2.imread(test_noisy_path, cv2.IMREAD_GRAYSCALE)
            test_img = cv2.resize(test_img_raw, (IMG_SIZE, IMG_SIZE))
            
            # 입력 전처리
            input_tensor = test_img.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=-1)
            input_tensor = np.repeat(input_tensor, 3, axis=-1) # 3채널 확장
            input_tensor = np.expand_dims(input_tensor, axis=0) # 배치 차원 추가

            # 예측
            pred = best_model.predict(input_tensor)
            result = (pred[0] * 255).astype(np.uint8)

            # 결과 시각화
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Input (Noisy)")
            plt.imshow(test_img, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Output (AI Cleaned)")
            plt.imshow(result, cmap='gray')
            plt.axis('off')
            plt.show()
            
        else:

            print("데이터 경로를 찾지 못했습니다.")
