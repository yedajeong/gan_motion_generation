import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile
import cv2


# discriminator.summary()
# dataset load
reposDir = '/Users/ydj89/Desktop/다도미/2022-2 소융캡스톤디자인/repos'
train_images = np.load(reposDir + '/data.npy')

# image resizing
train_resize = [cv2.resize(train_images[i], dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for i in range(len(train_images))]
train_resize = np.array(train_resize)

train_resize_min = train_resize.min()
train_resize_max = train_resize.max()

# normalization [-1, 1]
# train_normal = (2 * train_resize - train_resize_max - train_resize_min) / (train_resize_max - train_resize_min)

# normalization [0, 1] -> LeakyReLU 쓰는데 [0, 1]사이값이어도 되나?
train_normal = (train_resize - train_resize_min) / (train_resize_max - train_resize_min)

# conver to tensor
BUFFER_SIZE = train_images.shape[0]  # 총 sample 개수 (8154)
BATCH_SIZE = 32  # iteration: BUFFER_SIZE / BATCH_SIZE

train_tensor = tf.convert_to_tensor(train_normal, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices(train_tensor).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)  # <BatchDataset element_spec=TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None)>

# sample image
# for x in train_normal:
#     plt.axis("off")
#     plt.imshow((x * 255).astype("int32"))
#     plt.colorbar()
#     break


# discriminator
# BN -> input layer 제외 모든 층에
discriminator = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),

        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

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
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),

        layers.Dense(16 * 16 * 128),
        layers.BatchNormalization(),

        layers.Reshape((16, 16, 128)),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),


        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()


# compile(config training steps)
PREDICTIONS = np.zeros(shape=(100, 128, 128, 3))  # generated image 값 저장용
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
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
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
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
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
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


# save generated image
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255

        PREDICTIONS[epoch] = generated_images.numpy()[0]

        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./generated_image/model2/generated_img_%03d_%d.png" % (epoch, i))


# train
epochs = 10  # In practice, use ~100 epochs

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)
gan.fit(train_dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)])
np.save('./history/model2/PREDICTIONS', PREDICTIONS)
try:
    gan.save('./history/model2')
    print("전체 모델 저장")
except:
    gan.save_weights('./history/model2')
    print("모델 weight만 저장")