import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import cv2

from IPython import display
from matplotlib.pylab import rcParams

# dataset load
reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'
train_images = np.load(reposDir + '/data.npy')

# image resizing
train_resize = [cv2.resize(train_images[i], dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for i in range(len(train_images))]
train_resize = np.array(train_resize)

# normalization [-1, 1]
train_resize_min = train_resize.min()
train_resize_max = train_resize.max()
train_normal = (2 * train_resize - train_resize_max - train_resize_min) / (train_resize_max - train_resize_min)

# random sample image
random_index = []
for i in range(8):
    random_index.append(np.random.randint(0, train_images.shape[0]))

plt.figure(figsize=(30, 10))

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow((train_normal[random_index[i]] + 1) / 2)
    plt.colorbar()
    plt.title(f'index: {random_index[i]}')

# plt.show()
plt.savefig('./train_data_random.png')

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 128

train_dataset = tf.data.Dataset.from_tensor_slices(train_resize).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# generator model
# output layer 제외 모든 층에 BN
def make_generator_model():

    # start
    model = tf.keras.Sequential()

    # first: Dense layer
    model.add(layers.Dense(32*32*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))

    # second: Reshape layer
        # model.output_shape = (None, 32, 32, 256)
    model.add(layers.Reshape((32, 32, 256)))
    model.add(layers.BatchNormalization())

    # third: Conv2DTranspose layer
        # model.output_shape = (None, 32, 32, 128)
    model.add(layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))

    # fourth: Conv2DTranspose layer
        # model.output_shape = (None, 64, 64, 64)
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))

    # fifth: Conv2DTranspose layer
        # model.output_shape = (None, 128, 128, 3)
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


# noise vector (latent vector)
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# plt.imshow((generated_image[0, :, :, 0]+1)/2)
# plt.colorbar()


# discrimnator model
# input layer 제외 모든 층에 BN
def make_discriminator_model():

    # start
    model = tf.keras.Sequential()

    # first: Conv2D layer
    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.3))

    # second: Conv2D layer
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.3))

    # third: Flatten layer
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())

    # fourth: Dense Layer
    model.add(layers.Dense(1, activation='sigmoid'))
    model.add(layers.BatchNormalization())
    # logits = model
    # model = tf.sigmoid(logits)

    return model


# raw output (decision of untrained discriminator)
discriminator = make_discriminator_model()

decision_fake = discriminator(generated_image)  # noise, generator로 만든 이미지
print(decision_fake)

decision_real = discriminator(np.expand_dims(train_resize[0], axis=0))
print(decision_real)


# define loss function & optimizer
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 레이블 스무딩 파라미터 -> real image에 1 대신 1*smooth 값으로 레이블
smooth = 0.1
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output)*(1-smooth), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


global PREDICTIONS
EPOCHS = 50
PREDICTIONS = np.zeros(shape=(EPOCHS, 128, 128, 3))


# image generate & save
def generate_and_save_images(model, epoch, test_input):
    global PREDICTIONS

    # `training`이 False로 맞춰짐
    # -> (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됨
    # epoch: 1-based
    predictions = model(test_input, training=False)
    PREDICTIONS[epoch-1] = predictions[0].numpy()

    fig = plt.figure(figsize=(30, 10))

    for i in range(predictions.shape[0]):
        plt.subplot(2, 4, i+1)
        plt.imshow((predictions[i, :, :, 0]+1)/2)
        plt.colorbar()

    plt.savefig('./generated_image/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close(fig)


# history visualization
rcParams['figure.figsize'] = 15, 6


def draw_train_history(history, epoch):

    # loss
    plt.subplot(211)
    plt.plot(history['gen_loss'])
    plt.plot(history['disc_loss'])
    plt.title('model loss')
    plt.xlabel('batch iters')
    plt.ylabel('loss')
    plt.legend(['gen_loss', 'disc_loss'], loc='upper left')

    # accuracy
    '''
    plt.subplot(212)
    plt.plot(history['fake_accuracy'])
    plt.plot(history['real_accuracy'])
    plt.title('discriminator accuracy')
    plt.xlabel('batch iters')
    plt.ylabel('accuracy')
    plt.legend(['fake_accuracy', 'real_accuracy'], loc='upper left')
    '''

    # epoch별 그래프 이미지 파일로 저장
    plt.savefig('./history/train_history_at_epoch_{:04d}.png'.format(epoch))

    # plt.show()


# trian loop
noise_dim = 100
num_examples_to_generate = 8

seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)


        # D_loss 계산 시 sigmoid 통과 전 logits값 사용
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # loss
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # accuracy
        # real_accuracy, fake_accuracy = discriminator_accuracy(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss


def train(dataset, epochs):

    history = {'gen_loss': [], 'disc_loss': []}

    for epoch in range(epochs):
        start = time.time()

        for it, image_batch in enumerate(dataset):
            gen_loss, disc_loss = train_step(image_batch)
            history['gen_loss'].append(gen_loss)
            history['disc_loss'].append(disc_loss)

        # GIF를 위한 이미지를 바로 생성합니다.
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # 마지막 에포크가 끝난 후 생성합니다.
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)

    # PREDICTIONS 결과 텐서 저장
    np.save(reposDir + '/PREDICTIONS', PREDICTIONS)

    display.clear_output(wait=True)
    draw_train_history(history, epochs)

# model training
train(train_dataset, epochs=EPOCHS)