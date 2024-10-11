# 패키지 임포트 

# Hparam 설정

## 데이터 처리하는 과정 (CIFAR10 활용)
# dataset 필요 
# dataloader 필요 

## 모델 만들기 ##
# 모델 설계도 만들기 
    # 초기화 
        # CIFAR10 활용으로 크기 변화 주의
        # Layer1~5 & Out 
    # forward 
        # Layer1~5 & Out
# 모델 생성에 사용이 될 Hparam 설정 
# 모델 객체를 생성 (설계도 + Hparam)

# Loss
# Optimizer 

# for loop를 돌면서 데이터를 불러오기 
    # 불러온 데이터를 모델에 넣기 
    
    # 나온 출력물로 loss를 계산 
    # Loss로 back prop 진행 
    # optimizer를 이용해 최적화를 진행 

    # 학습 중간에 평가를 진행해서 
    # 성능이 좋으면 저장을 진행 