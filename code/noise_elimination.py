import numpy as np
import cv2
import tensorflow
from tensorflow import keras

data = np.load('/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos/data.npy')

resize = [cv2.resize(data[i], dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for i in range(len(data))]
resize = np.array(resize)

resize = (resize - resize.min()) / (resize.max() - resize.min())
resize *= 255

for i in range(len(resize)):
    img = keras.preprocessing.image.array_to_img(resize[i])
    img.save("/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos/gan_pytorch_1.12/nturgb_gan/%04d.png" % i)
import numpy as np
import cv2
from PIL import Image
import imageio

reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'
# data = np.load(reposDir + '/data_class9.npy')
# data = np.load(reposDir + '/data_class33.npy')
# data = np.load(reposDir + '/data_class9(2).npy')
# data = np.load(reposDir + '/data_class9(3).npy')
data = np.load(reposDir + '/data_class27(2).npy')
# data = np.load(reposDir + '/data_class33(2).npy')

real = data[1]  # 원래 위치값 범위

# fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model5/1트_batch128_A009/generated_img_107_0.png')  # 이미지->배열화 [0, 255] 범위
# fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model5/2트_batch128_A033/generated_img_092_6.png')
# fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model6/1트_batch128_A009/generated_img_137_4.png')
# fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model6/1트_batch128_A009/generated_img_121_9.png')
# fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model6/2트_전처리_A009/generated_img_132_0.png')
# fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model6/2트_전처리_A009/generated_img_135_9.png')
# fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model6/3트_A033/generated_img_147_0.png')
fake = Image.open(reposDir + '/gan_tensorflow/generated_image/model6/4트_A027/generated_img_144_9.png')

fake = np.array(fake)
fake = cv2.resize(fake, dsize=(25, 128), interpolation=cv2.INTER_AREA)
# fake = data[1]

def img_show(movement, filename):
    # normalization [0, 1]
    movement = (movement - movement.min()) / (movement.max() - movement.min())

    img = cv2.resize((movement*255).astype(np.uint8), dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(img)
    img.save(reposDir + '/gan_tensorflow/vis/' + filename + '_reshape.png')

    img2 = cv2.resize((movement*255).astype(np.uint8), dsize=(25, 128), interpolation=cv2.INTER_AREA)
    img2 = Image.fromarray(img2)
    img2.save(reposDir + '/gan_tensorflow/vis/' + filename + '.png')

# with open(reposDir + '/gan_tensorflow/vis/real.txt', 'w') as f:
#     for frame in real:
#         for joint in frame:
#             f.write(str(joint))
#             f.write('\n')
#         f.write('\n')
#
# with open(reposDir + '/gan_tensorflow/vis/fake.txt', 'w') as f:
#     for frame in fake:
#         for joint in frame:
#             f.write(str(joint))
#             f.write('\n')
#         f.write('\n')
#
# img_show(real, 'real')
img_show(fake, 'fake')


# '''
# [0, 1] initialization
real = (real - real.min()) / (real.max() - real.min())

fake = (fake - fake.min()) / (fake.max() - fake.min())

# joint별 각 축에서의 최대 이동량
threshold_x = []
threshold_y = []
threshold_z = []

for joint in range(25):

    max_x = abs(real[0, joint, 0] - real[1, joint, 0])
    max_y = abs(real[0, joint, 1] - real[1, joint, 1])
    max_z = abs(real[0, joint, 2] - real[1, joint, 2])

    for i in range(1, 127):
        new_x = abs(real[i, joint, 0] - real[i+1, joint, 0])
        new_y = abs(real[i, joint, 1] - real[i+1, joint, 1])
        new_z = abs(real[i, joint, 2] - real[i+1, joint, 2])

        max_x = new_x if max_x < new_x else max_x
        max_y = new_y if max_y < new_y else max_y
        max_z = new_z if max_z < new_z else max_z

    threshold_x.append(max_x)
    threshold_y.append(max_y)
    threshold_z.append(max_z)


fake_x = []  # (25, 128)
fake_y = []
fake_z = []

alpha = 1.3

for joint in range(25):
    x = []
    y = []
    z = []
    new_x = []
    new_y = []
    new_z = []

    for i in range(128):
        x.append(fake[i, joint, 0])
        y.append(fake[i, joint, 1])
        z.append(fake[i, joint, 2])

    new_x.append(x[0])
    new_y.append(y[0])
    new_z.append(z[0])
    for frame in range(127):
        dif_x = x[frame] - x[frame + 1]
        dif_y = y[frame] - y[frame + 1]
        dif_z = z[frame] - z[frame + 1]

        '''
        if abs(dif_x) > threshold_x[joint] and dif_x > 0:
            new_x.append(x[frame] - threshold_x[joint])
            # print(joint)
        elif abs(dif_x) > threshold_x[joint] and dif_x < 0:
            new_x.append(x[frame] + threshold_x[joint])
            # print(joint)
        else:
            new_x.append(x[frame  + 1])

        if abs(dif_y) > threshold_y[joint] and dif_y > 0:
            new_y.append(y[frame] - threshold_y[joint])
            # print(joint)
        elif abs(dif_y) > threshold_y[joint] and dif_y < 0:
            new_y.append(y[frame] + threshold_y[joint])
            # print(joint)
        else:
            new_y.append(y[frame + 1])

        if abs(dif_z) > threshold_z[joint] and dif_z > 0:
            new_z.append(z[frame] - threshold_z[joint])
            # print(joint)
        elif abs(dif_z) > threshold_z[joint] and dif_z < 0:
            new_z.append(z[frame] + threshold_z[joint])
            # print(joint)
        else:
            new_z.append(z[frame + 1])
        '''

        '''
        if abs(dif_x) > threshold_x[joint] and dif_x > 0:
            new_x.append(alpha*(x[frame] - threshold_x[joint]) + real[frame, joint, 0])
            # print(joint)
        elif abs(dif_x) > threshold_x[joint] and dif_x < 0:
            new_x.append(alpha*(x[frame] + threshold_x[joint]) + real[frame, joint, 0])
            # print(joint)
        else:
            new_x.append(alpha*x[frame+1] + real[frame, joint, 0])

        if abs(dif_y) > threshold_y[joint] and dif_y > 0:
            new_y.append(alpha * (y[frame] - threshold_y[joint]) + real[frame, joint, 1])
            # print(joint)
        elif abs(dif_y) > threshold_y[joint] and dif_y < 0:
            new_y.append(alpha * (y[frame] + threshold_y[joint]) + real[frame, joint, 1])
            # print(joint)
        else:
            new_y.append(alpha * y[frame + 1] + real[frame, joint, 1])

        if abs(dif_z) > threshold_z[joint] and dif_z > 0:
            new_z.append(alpha * (z[frame] - threshold_z[joint]) + real[frame, joint, 2])
            # print(joint)
        elif abs(dif_z) > threshold_z[joint] and dif_z < 0:
            new_z.append(alpha * (z[frame] + threshold_z[joint]) + real[frame, joint, 2])
            # print(joint)
        else:
            new_z.append(alpha * z[frame + 1] + real[frame, joint, 2])
        '''

        if abs(dif_x) > threshold_x[joint] and dif_x > 0:
            new_x.append(((2-alpha)*x[frame] + alpha*real[frame, joint, 0]) / 2 - threshold_x[joint])
            # print(joint)
        elif abs(dif_x) > threshold_x[joint] and dif_x < 0:
            new_x.append(((2-alpha)*x[frame] + alpha*real[frame, joint, 0]) / 2 + threshold_x[joint])
            # print(joint)
        else:
            new_x.append(((2-alpha)*x[frame + 1] + alpha*real[frame + 1, joint, 0]) / 2)

        if abs(dif_y) > threshold_y[joint] and dif_y > 0:
            new_y.append(((2-alpha)*y[frame] + alpha*real[frame, joint, 1]) / 2 - threshold_y[joint])
            # print(joint)
        elif abs(dif_y) > threshold_y[joint] and dif_y < 0:
            new_y.append(((2-alpha)*y[frame] + alpha*real[frame, joint, 1]) / 2 + threshold_y[joint])
            # print(joint)
        else:
            new_y.append(((2-alpha)*y[frame + 1] + alpha*real[frame + 1, joint, 1]) / 2)

        if abs(dif_z) > threshold_z[joint] and dif_z > 0:
            new_z.append(((2-alpha)*z[frame] + alpha*real[frame, joint, 2]) / 2 - threshold_z[joint])
            # print(joint)
        elif abs(dif_z) > threshold_z[joint] and dif_z < 0:
            new_z.append(((2-alpha)*z[frame] + alpha*real[frame, joint, 2]) / 2 + threshold_z[joint])
            # print(joint)
        else:
            new_z.append(((2-alpha)*z[frame + 1] + alpha*real[frame + 1, joint, 2]) / 2)

    fake_x.append(new_x)
    fake_y.append(new_y)
    fake_z.append(new_z)


fake_x = np.array(fake_x).transpose()
fake_y = np.array(fake_y).transpose()
fake_z = np.array(fake_z).transpose()

fake_pp = np.stack((fake_x, fake_y, fake_z), axis=-1)

img_show(fake_pp, 'fake_pp')
# '''