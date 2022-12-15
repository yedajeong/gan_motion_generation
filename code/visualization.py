import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
import cv2
from PIL import Image

reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'

# labels of the body joints
# 1-base of the spine 2-middle of the spine 3-neck
# 4-head 5-left shoulder 6-left elbow 7-left wrist
# 8- left hand 9-right shoulder 10-right elbow
# 11-right wrist 12- right hand 13-left hip 14-left knee
# 15-left ankle 16-left foot 17- right hip 18-right knee
# 19-right ankle 20-right foot 21-spine 22- tip of the left hand
# 23-left thumb 24-tip of the right hand 25- right thumb
bone_list = [[1, 2], [2, 21], [21, 5], [21, 9], [21, 3], [3, 4],
             [9, 10], [10, 11], [11, 12], [12, 24], [11, 25],
             [5, 6], [6, 7], [7, 8], [7, 23], [8, 22],
             [1, 17], [17, 18], [18, 19], [19, 20],
             [1, 13],  [13, 14], [14, 15], [15, 16]]

# convert the list to a numpy-array to easily decrement all the values by 1
bone_list = np.array(bone_list) - 1
frame = 128  # frame 개수 고정

# marker color _ 0_based indexing으로 바꿔주기
red = [4, 3, 21, 2, 1]  # 머리~척추
green = [9, 10, 11, 12, 24, 25]  # 오른팔
blue = [5, 6, 7, 8, 22, 23]  # 왼팔
yellow = [17, 18, 19, 20]  # 왼다리
magenta = [13, 14, 15, 16]  # 오른다리

'''
######### real image ###########
# data = np.load(reposDir + '/data_class9.npy')
# data = np.load(reposDir + '/data_class27.npy')
# data = np.load(reposDir + '/data_class9(2).npy')
# data = np.load(reposDir + '/data_class9(3).npy')
# data = np.load(reposDir + '/data_class33(2).npy')
data = np.load(reposDir + '/data_class27(2).npy')

movement = data[1]  # data.npy: A33 sample 하나 (128*25*3)

    # normalization [0, 1]
# movement = cv2.resize(movement, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
movement = (movement - movement.min()) / (movement.max() - movement.min())

    # img show
img2 = Image.fromarray((movement*255).astype(np.uint8))
img2.show()

img = cv2.resize((movement*255).astype(np.uint8), dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
img = Image.fromarray(img)
img.show()
'''

######### fake image ###########
# data = np.load(reposDir + '/gan_tensorflow/history/model3/4트/PREDICTIONS_4t.npy')
# movement = data[8]  # generated image 하나 (128*128*3)

    # image 읽어서 vis
# img = Image.open(reposDir + '/gan_tensorflow/generated_image/model3/4트_batch64/generated_img_009_2.png')
# img = Image.open(reposDir + '/gan_tensorflow/generated_image/model3-A009/1트_batch64/generated_img_195_0.png')
# img = Image.open(reposDir + '/gan_tensorflow/generated_image/model3-A009/1트_batch64/generated_img_186_6.png')
# img = Image.open((reposDir + '/gan_tensorflow/generated_image/model3-A027/2트_batch128/generated_img_185_1.png'))
# img = Image.open((reposDir + '/gan_tensorflow/generated_image/model4/1트_batch128_epoch100(중단)/generated_img_051_3.png'))
# img = Image.open((reposDir + '/gan_tensorflow/generated_image/model5/1트_batch128_A009/generated_img_034_0.png'))
# img = Image.open((reposDir + '/gan_tensorflow/generated_image/model5/1트_batch128_A009/generated_img_107_0.png'))
# img = Image.open((reposDir + '/gan_tensorflow/generated_image/model5/2트_batch128_A033/generated_img_092_6.png'))
# img = Image.open((reposDir + '/gan_tensorflow/generated_image/model6/2트_전처리_A009/generated_img_132_0.png'))

img = Image.open((reposDir + '/gan_tensorflow/vis/fake_pp_reshape.png'))
# img.show()
movement = np.array(img)

    # resizing 128*128 -> 128*25
    # 축소할 때 보간법: 주로 cv2.INTER_AREA
movement = cv2.resize(movement, dsize=(25, 128), interpolation=cv2.INTER_AREA)
img2 = Image.fromarray(movement)
img2.show()

    # normalization [0, 1]
movement = np.array(movement, dtype=np.float64)
# movement /= 255.0
movement = (movement - movement.min()) / (movement.max() - movement.min())
# '''

# 개별 frame plot
def make_plot(skeleton):
    # global min_lim, max_lim

    ax = plt.axes(projection='3d')

    # 축 범위 설정
    # ax.set_xlim(left=0, right=0.4)
    # ax.set_ylim(bottom=0.6, top=1)
    # ax.set_zlim(bottom=0, top=0.4)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_zlim(bottom=0, top=1)


    plt.title('Skeleton')

    x1, z1, y1 = zip(*[(x, y, z) for x, y, z in [skeleton[idx-1] for idx in red]])
    x2, z2, y2 = zip(*[(x, y, z) for x, y, z in [skeleton[idx-1] for idx in green]])
    x3, z3, y3 = zip(*[(x, y, z) for x, y, z in [skeleton[idx-1] for idx in blue]])
    x4, z4, y4 = zip(*[(x, y, z) for x, y, z in [skeleton[idx-1] for idx in yellow]])
    x5, z5, y5 = zip(*[(x, y, z) for x, y, z in [skeleton[idx-1] for idx in magenta]])

    ax.scatter(x1, y1, z1, c='r', s=20)
    ax.scatter(x2, y2, z2, c='g', s=20)
    ax.scatter(x3, y3, z3, c='b', s=20)
    ax.scatter(x4, y4, z4, c='y', s=20)
    ax.scatter(x5, y5, z5, c='m', s=20)

    x = skeleton[:, 0]  # x
    y = skeleton[:, 2]  # z
    z = skeleton[:, 1]  # y
    #
    # sc = ax.scatter(x, y, z, s=20) # s: 마커의 크기


    for bone in bone_list:
        ax.plot3D([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], [z[bone[0]], z[bone[1]]], 'gray')

    # plt.show()


# plot 이미지 저장
duration_rate = 0.05
filenames = []
for i, skeleton in enumerate(movement):
    # skeleton.shape: (25, 3)
    # movement.shape: (128, 25, 3)
    make_plot(skeleton)
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

frames = []
for filename in filenames:
    if filename.endswith(".png"):
        print(filename)
        # frames.append(imageio.imread(filename))
        frames.append(imageio.v2.imread(filename))

exportname = "output.gif"

imageio.mimsave(exportname, frames, format='GIF', duration=duration_rate)

for filename in set(filenames):
    os.remove(filename)