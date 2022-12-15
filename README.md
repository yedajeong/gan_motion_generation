# GAN을 이용한 사람 행동 생성

### 1. Summary
생성 모델인 GAN을 개선시킨 여러 모델들 중 이미지 생성에 높은 성능을 보이는 DCGAN(Deep Convolutional Generative Adversarial Network)을 사용해 사람의 행동에 해당하는 이미지를 생성한다.  사람의 움직임을 모션 인식 카메라로 촬영 후 frame별로 촬영된 3차원 x, y, z 좌표 상의 관절 위치값이 기록된 파일을 이미지(이를 모션 패치 라고 함)화 시켜 학습에 사용한다. 이미지의 가로 축은 25개의 joint에 해당되고 세로 축은 기록된 프레임에 해당되며 x, y, z값을 각각 R, G, B 값으로 저장되어 하나의 모션 당 한 장의 모션 패치(이미지)로 변환하여 이를 학습하고 유사한 이미지를 생성한다. 생성된 모션 패치는 다시 3d상의 좌표 값으로 mapping되어 움직이는 모션으로 시각화 해 최종적으로 생성된 모션을 확인한다.

### 2. Preprocessing
- 학습 데이터 셋: NTU-RGB+D action recognition dataset
  - 60개의 액션 class, 1 class 당 948개의 sample
  - 동작 간의 차이가 큰 총 9개의 액션을 선정해 개별 class, 전체 class를 input으로 하여 학습시킴 (총 8532개의 sample 중 missing skeleton 파일 혹은 noise가 포함된 파일을 제외한 8154개의 sample 사용)
  - data_class9.npy: A009(standing up)
  - data_class27.npy: A027(jump up)
  - data_class33.npy: A033(check time from watch)
  - data.npy: A005(drop), A008(sitting down), A009(standing up), A027(jump up), A031(pointing to something with finger), A033(check time from watch), A038(salute), A039(put the palms together), A040(cross hands in front to say stop)
  
- 파일 내에 기록된 joint 정보(3D 위치값, depth 정보, 적외선 센서 정보) 중 3D 위치값을 RGB 값으로 저장해 한 동작 당 한 장의 이미지 파일로 mapping
- 한 동작 class 내의 sample당 촬영된 frame 수가 불일치 -> 최대 frame 수(=128개)에 맞게 늘려 128 frame으로 통일 (이미지 파일의 크기는 가로는 25개의 joint, 세로는 128개의 frame으로 128 by 25 pixels)
- 1번 Joint(엉덩이 중앙 관절)를 좌표계의 원점 (0, 0, 0)으로 기준 삼아 전체 위치값 평행이동
- GAN의 input 형태로 만들기 위해 128 by 128 크기의 정방형 이미지로 resizing (opencv의 resize 함수, interpolation 시 INTER_CUBIC 적용)
