import numpy as np
import re
import os

# 모자란 frame만큼 마지막 위치 값으로 채우기
# filename(~.skeleton) read
missing_sk = []
reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'
f = open(reposDir+'/NTU_RGBD_samples_with_missing_skeletons.txt')

for line in f.readlines():
    filename = r'S\d{3}C\d{3}P\d{3}R\d{3}A0(05|08|09|27|31|33|38|39|40)\n'
    if re.match(filename, line):
        missing_sk.append(re.match(filename, line).group().strip('\n') + '.skeleton')

# missing file except
fileDir = os.listdir(reposDir+'/nturgb+d_skeletons')
filename = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A0(05|08|09|27|31|33|38|39|40).skeleton', file) for file in fileDir if file not in missing_sk]

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
max_frame = frame[-1]  # 128

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

    x = np.zeros(shape=(max_frame, 25))
    y = np.zeros(shape=(max_frame, 25))
    z = np.zeros(shape=(max_frame, 25))

    frameNum = 0

    for i in range(len(readline)):

        if readline[i] == "25\n":  # frame 시작부분

            for joint in range(25):
                line = readline[i + joint + 1].split(' ')

                x[frameNum, joint] = line[0]
                y[frameNum, joint] = line[1]
                z[frameNum, joint] = line[2]

            # 다음 frame 읽기
            i += 25
            frameNum += 1

    # max_frame에 frame 개수 맞추기 (idx: (0, maxframe))
    while frameNum < max_frame:

        for joint in range(25):
            # 이전 x, y, z 좌표 값으로 채워넣기 (마지막 프레임에서 정지상태)
            x[frameNum, joint] = x[frameNum - 1, joint]
            y[frameNum, joint] = y[frameNum - 1, joint]
            z[frameNum, joint] = z[frameNum - 1, joint]

        frameNum += 1

    # 매 파일마다 data에 저장
    # np.expand_dims: 3d sample -> 4d 확장
    sample = np.stack([x, y, z], axis=-1)
    data = np.append(data, np.expand_dims(sample, axis=0), axis=0)

    f.close()

# t.close()
print(data.shape)  # (8154, 128, 25, 3)
print(passCnt)

np.save(reposDir+'/data', data)