import os 
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

class AnimalDataset(Dataset): 
    def __init__(self, folder_path, transform=None): 
        self.folder_path = folder_path
        self.transform = transform

        # 일단 클래스 이름 가져오고 
        self.class_names = os.listdir(folder_path)
        self.class2index = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.index2class = {i: class_name for i, class_name in enumerate(self.class_names)}

        self.images = [] 
        self.lables = [] 

        # 그 클래스를 돌면서 
        for cls in self.class_names:
            target_idx = 1 
            
            cls_path = os.path.join(folder_path, cls)
            cls_data = os.listdir(cls_path)
            
            # 기준 이미지를 인덱스 기반으로 찾아보기 
            while f'{cls}_{target_idx}.jpg' in cls_data or f'{cls}_{target_idx}.jpeg' in cls_data: 
                # 만약 해당 인덱스의 기준 이미지가 있다면 
                # jpg 혹은 jpeg 확장자 고려해서 
                current_base_image_name = f'{cls}_{target_idx}.jpg' if os.path.exists(os.path.join(cls_path, f'{cls}_{target_idx}.jpg')) \
                                                                else f'{cls}_{target_idx}.jpeg'
                current_base_image_path = os.path.join(cls_path, current_base_image_name)
                
                # 파일을 읽고 
                base_image = cv2.imread(current_base_image_path)
                
                # 이미지와 레이블 정보를 저장 
                self.images.append(current_base_image_path)
                self.lables.append(self.class2index[cls])
                
                # 비교를 위한 평균 밝기를 계산해두고 
                base_brightness = self.calc_brightness(base_image)
                
                # 기준 이미지 이름 뒤에 _숫자 형태의 이미지를 찾아서 
                sub_index = 1 
                while os.path.exists(os.path.join(cls_path, f'{cls}_{target_idx}_{sub_index}.jpg')):
                    current_sub_image_name = f'{cls}_{target_idx}_{sub_index}.jpg'
                    current_sub_image_path = os.path.join(cls_path, current_sub_image_name)

                    # 값을 읽고 
                    sub_image = cv2.imread(current_sub_image_path)
                    # 역시 비교할 밝기를 계산하고 
                    sub_brightness = self.calc_brightness(sub_image)

                    # 만약 밝기가 20% 이내로 차이가 있다면 
                    if sub_brightness > base_brightness * 0.8: 
                        # 활용할 데이터로 사용하고 
                        self.images.append(current_sub_image_path)
                        self.lables.append(self.class2index[cls])
                    # 아니면 무시
                    
                    sub_index += 1
                target_idx += 1
        
        pass
    
    # 평균 밝기 계산 
    def calc_brightness(self, image):
        # 흑백으로 바꾸고 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 평균값 계산 
        brightness = np.mean(gray)
        return brightness

    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, idx): 
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = self.lables[idx]

        if self.transform : 
            image = self.transform(image)

        return image, label 