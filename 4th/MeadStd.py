import glob
import numpy as np
from PIL import Image

def get_image_stats(folder_path):
    # 하위 폴더에 들어있는 모든 이미지를 들고옴 
    # 확장자 jpg와 jpeg 조건을 줘야 함 
    image_paths = glob.glob(f"{folder_path}/**/*", recursive=True)
    valid_extensions = (".png", ".jpg", ".jpeg")
    all_images = [img for img in image_paths if img.lower().endswith(valid_extensions)]

    # 문제 없는 이미지 담고 
    images = []

    # 이미지를 돌면서 
    for img_path in all_images:
        # 문제 없으면 불러와서 이미지에 담고 
        ## RGB만 활용, numpy 배열로 치환 & 저장 
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB') 
                np_img = np.array(img) 
                images.append(np_img) 
        
        ## 손상 상태라면 이미지 path로 알려줌 
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    # 병합 먼저 하고 
    images = np.stack(images)

    # 채널별 평균과 표준편차 계산 
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    # 결과 출력 (255 기준)
    print(f"Mean (R, G, B): {mean}")  # Mean (R, G, B): [105.64623082 103.83895667  86.9514535 ]
    print(f"Std (R, G, B): {std}")  # Std (R, G, B): [77.87899635 74.44912627 75.87830578]
    
    # 결과 출력 (1 기준)
    print(f"Mean (R, G, B): {mean/255}")  # Mean (R, G, B): [0.41429894 0.40721159 0.34098609]
    print(f"Std (R, G, B): {std/255}")  # Std (R, G, B): [0.30540783 0.29195736 0.29756198]

# 폴더 경로 설정
folder_path = '../animal_data'
get_image_stats(folder_path)