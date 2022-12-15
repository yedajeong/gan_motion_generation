# GAN을 이용한 사람 행동 생성

## Summary
생성 모델인 GAN을 개선시킨 여러 모델들 중 이미지 생성에 높은 성능을 보이는 DCGAN(Deep Convolutional Generative Adversarial Network)을 사용해 사람의 행동에 해당하는 이미지를 생성한다.  사람의 움직임을 모션 인식 카메라로 촬영 후 frame별로 촬영된 3차원 x, y, z 좌표 상의 관절 위치값이 기록된 파일을 이미지(이를 모션 패치 라고 함)화 시켜 학습에 사용한다. 이미지의 가로 축은 25개의 joint에 해당되고 세로 축은 기록된 프레임에 해당되며 x, y, z값을 각각 R, G, B 값으로 저장되어 하나의 모션 당 한 장의 모션 패치(이미지)로 변환하여 이를 학습하고 유사한 이미지를 생성한다. 생성된 모션 패치는 다시 3d상의 좌표 값으로 mapping되어 움직이는 모션으로 시각화 해 최종적으로 생성된 모션을 확인한다.


### 1. Preprocessing
학습 데이터 셋: NTU-RGB+D action recognition dataset (https://github.com/shahroudy/NTURGB-D)

  - 60개의 액션 class, 1 class 당 948개의 sample
  - 동작 간의 차이가 큰 총 9개의 액션을 선정해 개별 class, 전체 class를 input으로 하여 학습시킴 (총 8532개의 sample 중 missing skeleton 파일 혹은 noise가 포함된 파일을 제외한 8154개의 sample 사용)
  - data_class9.npy: A009(standing up)
  - data_class27.npy: A027(jump up)
  - data_class33.npy: A033(check time from watch)
  - data.npy: A005(drop), A008(sitting down), A009(standing up), A027(jump up), A031(pointing to something with finger), A033(check time from watch), A038(salute), A039(put the palms together), A040(cross hands in front to say stop)
  
1) 파일 내에 기록된 joint 정보(3D 위치값, depth 정보, 적외선 센서 정보) 중 3D 위치값을 RGB 값으로 저장해 한 동작 당 한 장의 이미지(=모션 패치) 파일로 mapping
2) 한 동작 class 내의 sample당 촬영된 frame 수가 불일치 -> 최대 frame 수(=128개)에 맞게 늘려 128 frame으로 통일 (이미지 파일의 크기는 가로는 25개의 joint, 세로는 128개의 frame으로 128 by 25 pixels) -> 이를 **모션패치**라 한다.
3) 1번 Joint(엉덩이 중앙 관절)를 좌표계의 원점 (0, 0, 0)으로 기준 삼아 전체 위치값 평행이동
4) GAN의 input 형태로 만들기 위해 128 by 128 크기의 정방형 이미지로 resizing (interpolation 시 INTER_CUBIC 적용)


### 2. Train Model(DCGAN)
GAN의 discriminator, generator의 각 층에 convolutional layer을 적용시킨 DCGAN 모델을 학습
  - 1가지 class, 4가지 class, 9가지 class에 대한 데이터를 input으로 넣어 학습 진행
  - generator의 출력층에서 활성 함수로 사용된 tanh 함수의 출력값의 범위에 맞춰 입력 데이터(이미지)의 범위를 [-1, 1] 사이로 정규화
  - batch size는 64, 128, epoch 수는 50, 100, 150 내에서 조정해가며 학습시켰을 때 가장 생성 이미지의 성능이 좋았던 batch size는 128, epoch은 150
  - discriminator는 5개, generator는 8개 층 사용
  - discriminator의 input layer와 generator의 output layer를 제외한 모든 층에 Batch Normalization을 사용
  - generator의 활성 함수는 ReLU, discriminator의 활성 함수는 Leaky ReLU를 사용
  - hyper parameter을 조정한 Adam optimizer 사용 (learning rate 0.0002, momentum(=beta1) 0.5로 트레이닝시 가장 안정적)
  - discriminator에서 작은 random noise값을 label에 더하는 label smoothing을 사용
  - generator의 deconvolution시 UpSampling2d+Conv2d 와 Conv2dTranspose 함께 사용 (생성 이미지 내의 grid artifact 제거를 위함)


### 3. Visualization
  - 학습 및 생성된 정방형 이미지 128 by 128 -> 원래 크기(128 by 25)의 모션 패치로 resizing (interpolation 시 INTER_AREA 적용)
  - matplotlib, Axes3D로 25개의 관절 위치값을 3차원 상의 좌표로 mapping, 연결된 관절 사이는 선으로 연결하여 plotting
  - 128개의 frame을 연속적인 모션으로 시각화
  
  
### 4. Output
(1가지 class_A009: standing up 에 대한 결과 예시)

- model loss
![image](https://user-images.githubusercontent.com/49023751/207840634-e3bb2cc1-aaad-481c-a595-d05e29cb44a9.png)


- (순서대로)(a)원점 이동 전 학습 data (b)grid artifact 제거 전 생성 이미지 (c)grid artifact 제거 후 생성 이미지 (a)로 학습한 이미지 (d)원점 이동 후 학습 data (e)(d)로 학습한 이미지
![image](https://user-images.githubusercontent.com/49023751/207840709-09f27520-e844-445e-9576-c5f31f5122af.png)

![image](https://user-images.githubusercontent.com/49023751/207840744-a121c0ea-32f8-452e-882f-5c6c3e314d9f.png)

![image](https://user-images.githubusercontent.com/49023751/207840777-59be1b16-bec5-4e9a-acb2-268aa200b903.png)

![image](https://user-images.githubusercontent.com/49023751/207840808-fa3b5c3b-e21e-4415-8e07-19bff9f2d56a.png)

![image](https://user-images.githubusercontent.com/49023751/207840836-e24398aa-1802-4cd4-8454-89156fa23da0.png)


- (a)생성된 fake image (b) (a)를 모션패치화 (c)A009 class의 실제 동작 (d) 생성된 (b)를 모션으로 시각화한 동작
![image](https://user-images.githubusercontent.com/49023751/207840836-e24398aa-1802-4cd4-8454-89156fa23da0.png)

![image](https://user-images.githubusercontent.com/49023751/207840872-63799ea3-787e-4a6d-ab76-34079ddd2f05.png)

![output_A009](https://user-images.githubusercontent.com/49023751/207840943-469f8f2c-bb0f-4aea-b180-f864fa88e9f9.gif)

![output_fake_A009(7)](https://user-images.githubusercontent.com/49023751/207840983-ba4d9206-f27b-4373-9290-fd6b76c34d10.gif)
