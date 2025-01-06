# Author: Jan Premus
# Wasserstaein Generative adversarial networks with gradient penalty fro dynamic inversion of seismic source
# Generator can be used in PTMCMC_seis as a prior PDF  
# Large amount of credit goes to:
# https://github.com/eriklindernoren/Keras-GAN/tree/master
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which was used as a reference for the implementation

from __future__ import print_function, division

from keras.layers import Average
from keras.layers import Input, Dense, Reshape, Flatten, Layer
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers.legacy import Adam
from functools import partial
import joblib
import scipy.stats as stats
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
import numpy as np

class RandomWeightedAverage(Average):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((16, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GradientWrap(Layer):
    def call (self,inputs ):
        return K.gradients (inputs[0], inputs[1])
    
class WGANGP():
    def __init__(self):
        self.img_rows = 15
        self.img_cols = 1
        self.channels = 1
        self.img_shape = (self.channels, self.img_rows, self.img_cols)
        self.latent_dim = 3

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])

        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10],experimental_run_tf_function=False)
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer, experimental_run_tf_function=False)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        grads = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(grads)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(16, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(15,activation="sigmoid"))
        model.add(Reshape((1,15, 1)))
        model.build(self.latent_dim)
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(inputs=noise, outputs=img)

    def build_critic(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.1))
        model.add((Dense(128)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(15))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(1))
        
        model.build(self.img_shape)
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        temp2 = np.loadtxt('learn_models.dat')
        #temp1=np.reshape(temp2,(len(temp2),24))
        #temp = np.reshape(temp1[0:len(temp2),0:8,0:3],(len(temp2),24))

        #define noise
        lower, upper = 0., 1.
        mu, sigma = 0.5, 0.33
        Ndist = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        
       # print(np.reshape(temp,(len(temp),3,8)))
        self.scaler = MinMaxScaler()
        self.scaler.fit(temp2)
        temp=self.scaler.transform(temp2)
        X_train = np.reshape(temp,(len(temp),1,15,1))
        scaler_filename = "scaler.save"
        joblib.dump(self.scaler, scaler_filename) 
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        f = open("losses.txt", "a")
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Critic
                # ---------------------
                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = Ndist.rvs((batch_size, self.latent_dim)) #np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])
            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            f.write("%d %f %f\n"% (epoch, d_loss[0], g_loss) )
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample(epoch)                
                filename2 = 'model_generator_%04d.tf' % (epoch)
                self.generator.save(filename2,save_format='tf')
        f.close
    def samples(self, epoch):
        #define noise
        lower, upper = 0., 1.
        mu, sigma = 0.5, 0.33
        Ndist = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        r, c = 10, 10
        noise = Ndist.rvs((r * c, self.latent_dim))  #np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        cnt = 0
        
        for i in range(r):
            for j in range(c):
                temp = gen_imgs[cnt, 0,:,:]
                tt=temp
                tt = self.scaler.inverse_transform(temp.reshape(1,-1))
                cnt += 1
                with open("GANmodels_%d.dat" % epoch, "ab") as f:
                    np.savetxt(f, tt)


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=4000001, batch_size=16, sample_interval=100000)
