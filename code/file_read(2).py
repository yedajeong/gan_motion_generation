import numpy as np
import re
import os
import cv2
from PIL import Image
import imageio

# 개별 이미지를 128, 25로 늘리기 (interpolation)
# +) 원점 이동 (엉덩이 중앙 관절_1번째 joint 을 좌표계의 (0, 0, 0)으로 평행이동
# filename(~.skeleton) read
missing_sk = []
reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'
f = open(reposDir+'/NTU_RGBD_samples_with_missing_skeletons.txt')

for line in f.readlines():
    # filename = r'S\d{3}C\d{3}P\d{3}R\d{3}A0(05|08|09|27|31|33|38|39|40)\n'
    # filename = r'S\d{3}C\d{3}P\d{3}R\d{3}A009\n'
    # filename = r'S\d{3}C\d{3}P\d{3}R\d{3}A0(08|09|33|40)\n'
    # filename = r'S\d{3}C\d{3}P\d{3}R\d{3}A033\n'
    filename = r'S\d{3}C\d{3}P\d{3}R\d{3}A027\n'

    if re.match(filename, line):
        missing_sk.append(re.match(filename, line).group().strip('\n') + '.skeleton')

# except missing file
fileDir = os.listdir(reposDir+'/nturgb+d_skeletons')
# filename = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A0(05|08|09|27|31|33|38|39|40).skeleton', file) for file in fileDir if file not in missing_sk]
# filename = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A009.skeleton', file) for file in fileDir if file not in missing_sk]
# filename = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A0(08|09|33|40).skeleton', file) for file in fileDir if file not in missing_sk]
# filename = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A033.skeleton', file) for file in fileDir if file not in missing_sk]
filename = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A027.skeleton', file) for file in fileDir if file not in missing_sk]

# drop None
filename = list(filter(None, filename))
filename = [file.group() for file in filename]

# max frame count
frame = []

# filename export
# with open('filename.txt', 'w') as f:
#     for file in filename:
#         f.write(file)
#         f.write('\n')


for file in filename:
    f = open(reposDir+'/nturgb+d_skeletons/'+file, 'r')
    frameCnt = int(f.readline().strip('\n'))
    frame.append(frameCnt)

frame.sort()
# max_frame = frame[-1]  # 128
max_frame = 128

# max frame count for all files
'''
filename(all files) export
with open(reposDir+'/filename.txt', 'w', encoding='UTF-8') as f:
    for file in filename:
        f.write(file+'\n')

frame_dict = {}  # surfix(action class): min_frame, max_frame
# for surfix in ['06', '07', '08', '09', '23', '24', '26', '31', '43', '52']:
for surfix in [str(i) for i in range(1, 10)]:
    frame = []

    tmp = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A00'+surfix+'.skeleton', file) for file in filename]
    tmp = list(filter(None, tmp))
    tmp = [file.group() for file in tmp]

    for file in tmp:
        f = open(reposDir+'/nturgb+d_skeletons/'+file, 'r')
        frameCnt = int(f.readline().strip('\n'))
        frame.append(frameCnt)

    frame.sort()
    frame_dict[surfix] = (frame[0], frame[-1])

for surfix in [str(i) for i in range(10, 61)]:
    frame = []

    tmp = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A0' + surfix + '.skeleton', file) for file in filename]
    tmp = list(filter(None, tmp))
    tmp = [file.group() for file in tmp]

    for file in tmp:
        f = open(reposDir + '/nturgb+d_skeletons/' + file, 'r')
        frameCnt = int(f.readline().strip('\n'))
        frame.append(frameCnt)

    frame.sort()
    frame_dict[surfix] = (frame[0], frame[-1])
'''

# data read from filename
data = np.empty((0, max_frame, 25, 3))
passCnt = 0
# t = open('./overFrameCnt.txt', 'w')

for file in filename:
    f = open(reposDir+'/nturgb+d_skeletons/'+file, 'r')
    readline = f.readlines()
    frameCnt = int(readline[0].strip('\n'))

    ###### frameCnt보다 실제 기록된 frame 더 많은 파일 제외 ######
    startJoint = 0
    for i in range(len(readline)):
        if readline[i] == "25\n":  # frame 시작부분
            startJoint += 1
            i += 25

    if startJoint > frameCnt:
        # t.write(file+'\n')
        passCnt += 1
        continue  # 해당 파일 pass
    ######################################################

    x = np.zeros(shape=(frameCnt, 25))
    y = np.zeros(shape=(frameCnt, 25))
    z = np.zeros(shape=(frameCnt, 25))

    frameNum = 0
    first = True

    for i in range(len(readline)):

        if readline[i] == "25\n":  # frame 시작부분

            if first:
                line = readline[i + 0 + 1].split(' ')
                move_x = float(line[0])
                move_y = float(line[1])
                move_z = float(line[2])
                first = False

            for joint in range(25):
                line = readline[i + joint + 1].split(' ')

                # 원점 이동 (1번째 관절=index 0을 원점으로)
                x[frameNum, joint] = float(line[0]) - move_x
                y[frameNum, joint] = float(line[1]) - move_y
                z[frameNum, joint] = float(line[2]) - move_z

            # 다음 frame 읽기
            i += 25
            frameNum += 1

    # max_frame에 frame 개수 맞추기 (idx: (0, maxframe))
    sample = np.stack([x, y, z], axis=-1)
    sample_resize = cv2.resize(sample, dsize=(25, 128), interpolation=cv2.INTER_CUBIC)
    sample_resize = np.array(sample_resize)

    # 매 파일마다 data에 저장
    # np.expand_dims: 3d sample -> 4d 확장
    data = np.append(data, np.expand_dims(sample_resize, axis=0), axis=0)

    f.close()

# t.close()
print(data.shape)  # (8154, 128, 25, 3) _ class 9개
print(passCnt)

# np.save(reposDir+'/data(2)', data)
# np.save(reposDir+'/data_class9(3)', data)
# np.save(reposDir+'/data(3)', data)
# np.save(reposDir+'/data_class33(2)', data)
np.save(reposDir+'/data_class27(2)', data)