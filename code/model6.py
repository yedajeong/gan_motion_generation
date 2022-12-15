###############################################################
# using 1 class: 9 _ standing up / 33 _ check time from watch
# file read 시 마지막 프레임으로 채우지 않고 선형보간 (image resizing)
# data_class9(3): 엉덩이 중앙 관절(1번 joint) 원점 (0, 0, 0)으로 선형이동
###############################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile
import cv2
import re

from keras.models import load_model
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
from keras.models import model_from_json

from sklearn.preprocessing import minmax_scale
from matplotlib.pylab import rcParams

# GAN - train_step - loss_metric.result() 넘파이로 변경해서 loss값 출력하기
# due to AttributeError: 'Tensor' object has no attribute 'numpy' 에러 -> in tf.2x 버전부터
tf.config.run_functions_eagerly(True)

'''
# file read
missing_sk = []
reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'
f = open(reposDir+'/NTU_RGBD_samples_with_missing_skeletons.txt')

for line in f.readlines():
    filename = r'S\d{3}C\d{3}P\d{3}R\d{3}A009\n'
    if re.match(filename, line):
        missing_sk.append(re.match(filename, line).group().strip('\n') + '.skeleton')

# missing file except
fileDir = os.listdir(reposDir+'/nturgb+d_skeletons')
filename = [re.match(r'S\d{3}C\d{3}P\d{3}R\d{3}A009.skeleton', file) for file in fileDir if file not in missing_sk]

# drop None
filename = list(filter(None, filename))
filename = [file.group() for file in filename]

# max frame count
frame = []

for file in filename:
    f = open(reposDir+'/nturgb+d_skeletons/'+file, 'r')
    frameCnt = int(f.readline().strip('\n'))
    frame.append(frameCnt)

frame.sort()
max_frame = frame[-1]
print(max_frame)  # 120

max_frame = 128  # 원래 input 크기랑 동일하게 만들기

# data read from filename
data = np.empty((0, max_frame, 25, 3))
passCnt = 0

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
    sample = np.stack([x, y, z], axis=-1)
    sample_resize = cv2.resize(sample, dsize=(25, 128), interpolation=cv2.INTER_CUBIC)
    sample_resize = np.array(sample_resize)

    # 매 파일마다 data에 저장
    # np.expand_dims: 3d sample -> 4d 확장
    data = np.append(data, np.expand_dims(sample_resize, axis=0), axis=0)

    f.close()

print(data.shape)  # (_, 128, 128, 3)
print(passCnt)

np.save(reposDir+'/data_class9(2)', data)
# '''

# dataset load
reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'
# train_images = np.load(reposDir + '/data_class9(2).npy')
# train_images = np.load(reposDir + '/data_class33.npy')
# train_images = data
# train_images = np.load(reposDir + '/data_class9(3).npy')
# train_images = np.load(reposDir + '/data(3).npy')
# train_images = np.load(reposDir + '/data_class33(2).npy')
train_images = np.load(reposDir + '/data_class27(2).npy')

# image resizing
train_resize = [cv2.resize(train_images[i], dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for i in range(len(train_images))]
train_resize = np.array(train_resize)

train_resize_min = train_resize.min()
train_resize_max = train_resize.max()

# normalization [-1, 1]
train_normal = (2 * train_resize - train_resize_max - train_resize_min) / (train_resize_max - train_resize_min)

# normalization [0, 1]
# train_normal = (train_resize - train_resize_min) / (train_resize_max - train_resize_min)

# conver to tensor
BUFFER_SIZE = train_images.shape[0]  # 총 sample 개수 (8154)
BATCH_SIZE = 128  # iteration: BUFFER_SIZE / BATCH_SIZE

train_tensor = tf.convert_to_tensor(train_normal, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices(train_tensor).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)  # <BatchDataset element_spec=TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None)>

# # sample image
# for i, x in enumerate(train_normal):
#     if i < 10:
#         plt.axis("off")
#         plt.imshow((x * 255).astype("int32"))
#         plt.colorbar()
#         plt.show()
#         i += 1
#     else:
#         break


# discriminator
# BN -> input layer 제외 모든 층에
# ReLU의 변형체 (LeakyReLU와 같은 것들) discriminator에 사용하도록 권
discriminator = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),

        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        #layers.ReLU(),

        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        #layers.ReLU(),

        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        #layers.ReLU(),

        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid"),
        layers.BatchNormalization(),
    ],
    name="discriminator",
)
discriminator.summary()


# generator
# BN -> output layer 제외 모든 층에
# ReLU는 generator에만 권장 (discriminator에서는 x)
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),

        layers.Dense(16 * 16 * 128),
        layers.BatchNormalization(),

        layers.Reshape((16, 16, 128)),
        layers.BatchNormalization(),

        layers.UpSampling2D(interpolation='nearest'),
        layers.Conv2D(128, kernel_size=5, strides=1, padding="same"),
        layers.BatchNormalization(),
        #layers.LeakyReLU(alpha=0.2),
        layers.ReLU(),

        layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding="same"),
        layers.BatchNormalization(),
        #layers.LeakyReLU(alpha=0.2),
        layers.ReLU(),

        layers.UpSampling2D(interpolation='nearest'),
        layers.Conv2D(64, kernel_size=5, strides=1, padding="same"),
        layers.BatchNormalization(),
        #layers.LeakyReLU(alpha=0.2),
        layers.ReLU(),

        layers.Conv2DTranspose(64, kernel_size=5, strides=1, padding="same"),
        layers.BatchNormalization(),
        #layers.LeakyReLU(alpha=0.2),
        layers.ReLU(),

        layers.UpSampling2D(interpolation='nearest'),
        layers.Conv2D(32, kernel_size=5, strides=1, padding="same"),
        layers.BatchNormalization(),
        #layers.LeakyReLU(alpha=0.2),
        layers.ReLU(),

        layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
    ],
    name="generator",
)
generator.summary()


# compile(config training steps)
epochs = 150
PREDICTIONS = np.zeros(shape=(epochs, 128, 128, 3))  # generated image 값 저장용
history = {'disc_loss': [], 'gen_loss': []}
iteration = 0

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn  # BinaryCrossEntropy (fake: 0, real:1)
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")  # metrix의 평균값 -> 한 해폭 당 d_loss, g_loss 출력?
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick! (라벨 스무딩? 1->1.05, 0->0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator (오류 역전파?)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)  # real, fake image를 섞어서 input
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        # D, G는 각자 따로 학습
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))  # fake image를 input
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

            # for draw history
        # global iteration
        # iteration += 1
        # if iteration == 15:
        #     history['disc_loss'].append(self.d_loss_metric.result().numpy())
        #     history['gen_loss'].append(self.g_loss_metric.result().numpy())
        #     iteration = 0

        history['disc_loss'].append(self.d_loss_metric.result().numpy())
        history['gen_loss'].append(self.g_loss_metric.result().numpy())

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def get_config(self):
        return {'discriminator': self.discriminator,
                'generator': self.generator,
                'latent_dim': self.latent_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        x = self.generator(inputs)
        return self.discriminator(x)

# save generated image
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim), mean=0, stddev=0.02)  # 평균: 0 표준편차: 0.02
        generated_images = self.model.generator(random_latent_vectors)

        PREDICTIONS[epoch] = generated_images.numpy()[0]

            # generator 출력 범위 (-1, 1) _ tanh
            # (0, 1) 사이로 범위 변환
        generated_min = generated_images.numpy().min()
        generated_max = generated_images.numpy().max()
        generated_images = (generated_images - generated_min) / (generated_max - generated_min)

        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./generated_image/model6/generated_img_%03d_%d.png" % (epoch, i))  # 이어서 학습

#'''
# train
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),  # 모멘텀(beta_1) 0.5로 설
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(train_dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)])

# draw train loss history
# rcParams['figure.figsize'] = 15, 6
plt.plot(history['disc_loss'])
plt.plot(history['gen_loss'])
plt.title('model loss')
# plt.xlabel('epoch')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(['disc_loss', 'gen_loss'], loc='upper left')
# plt.xticks(range(0, epochs+1, 10), fontsize=7)  # iteration: 8번
# plt.show()
plt.savefig('./history/model6/train_history.png')

# '''
# save model
np.save('./history/model6/PREDICTIONS', PREDICTIONS)

    # 1. HDF5
try:
    # gan.save('./history/model6/model6', save_format="tf")
    gan.save('./history/model6/model6.h5')
    print("전체 모델 저장_tf??")

except Exception as e:
    print(e)

    # 2. JSON
gan_json = gan.to_json()
with open("./history/model6/model6.json", "w") as json_file:
    json_file.write(gan_json)
print("전체 모델 저장_json")


gan.save_weights('./history/model6/model6_weight.h5')
# gan.save_weights('./history/model6_weight')  # 뭔차이?

# show model architecture _ 왜 안됨 ? ㅠ
# try:
#     SVG(model_to_dot(gan, show_shapes=True).create(prog='dot', format='svg'))
# except Exception as e:
#     print(e)
#
# try:
#     plot_model(gan, show_shapes=True, show_layer_names=True, to_file='./history/model6/model6.png')
# except Exception as e:
#     print(e)

'''
# load model
with open("./history/model6/model6.json", "r") as model_json:
    loaded_model = model_json.read()  # json string encoding a model configuration

loaded_model = model_from_json(loaded_model, custom_objects={'GAN': GAN})

loaded_model.build((latent_dim, latent_dim))

loaded_model.load_weights("./history/model6/model6_weight.h5")
print("Loaded model from disk")

loaded_model.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

loaded_model.fit(train_dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)])

# save model
np.save('./history/model6/PREDICTIONS(2)', PREDICTIONS)

    # 1. HDF5
try:
    loaded_model.save('./history/model6/model6(2).h5')
    print("전체 모델 저장_tf??")

except Exception as e:
    print(e)

    # 2. JSON
gan_json = loaded_model.to_json()
with open("./history/model6/model6(2).json", "w") as json_file:
    json_file.write(gan_json)
print("전체 모델 저장_json")


loaded_model.save_weights('./history/model6/model6_weight(2).h5')
# '''
