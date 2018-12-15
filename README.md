# DoodleGAN

"Quick, Draw!" was released as an experimental game by Google to educate the public in a playful way about how AI works. The game prompts users to draw an image depicting a certain category, such as ”banana,” “table,” etc. Consequently, more than 1B drawings were generated, of which a subset was publicly released as the basis for this competition’s training set. That subset contains 50M drawings encompassing 340 label categories. (https://www.kaggle.com/c/quickdraw-doodle-recognition)

Convolutional Neural Networks (CNNs) have continually demonstrated their superiority over other modeling choices for image based datasets. Since this project we will be dealing with image classification, using deep learning tools like CNNs is a natural choice. Furthermore, in recent years, Generative Adversarial Networks (GANs) have shown to be extremely effective at modeling the distribution of the data.  (http://papers.nips.cc/paper/5423-generative-adversarial-nets) The modeled distribution can be used to generate samples of the data that closely resemble real/true data. We seek to train a GAN to learn how to draw doodles with a given label, i.e the output is an image from a certain class. If we manage to train a good enough generator we could use it as a data augmentation tool that could help to train an even better classifier.

![bee](https://storage.googleapis.com/kaggle-media/competitions/quickdraw/what-does-a-bee-look-like-1.png)



# Simple GAN

Simple_GAN.ipynb contains a simple and somewhat naive implementation of a GAN.  The class of doodles we chose to use are cats (go figure).  An randomly chosen example cat image:

![cat](https://user-images.githubusercontent.com/14242505/50039091-fef49880-ffe0-11e8-8e89-0e17b910cfeb.png)


The generator is composed of 5 dense layers separated by LeakyReLUs.  The descriminator is similar, except it has 4 dense layers.  The architecture is shown below.

```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 1024)              803840    
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               524800    
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 512)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               131328    
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 257       
=================================================================
Total params: 1,460,225
Trainable params: 1,460,225
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5 (Dense)              (None, 256)               25856     
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 512)               131584    
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 512)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 1024)              525312    
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 1024)              0         
_________________________________________________________________
dense_8 (Dense)              (None, 2048)              2099200   
_________________________________________________________________
leaky_re_lu_7 (LeakyReLU)    (None, 2048)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 784)               1606416   
=================================================================
Total params: 4,388,368
Trainable params: 4,388,368
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         (None, 100)               0         
_________________________________________________________________
model_2 (Model)              (None, 784)               4388368   
_________________________________________________________________
model_1 (Model)              (None, 1)                 1460225   
=================================================================
Total params: 5,848,593
Trainable params: 4,388,368
Non-trainable params: 1,460,225
_________________________________________________________________
```

Initially, the generated cat images aren't very good.
![simple_bad](https://user-images.githubusercontent.com/14242505/50039096-12076880-ffe1-11e8-90d9-082f49ab66a2.png)

After many epochs of training, they get much better.

![simple_good](https://user-images.githubusercontent.com/14242505/50039097-16cc1c80-ffe1-11e8-92f4-18b94e04c9c5.png)

As you can see, the loss of the discriminator and generator converge.  While the generated cats are noticeable better than before, they are still not what someone would probably actually draw.
![simple_loss](https://user-images.githubusercontent.com/14242505/50039098-1b90d080-ffe1-11e8-971e-c731a34dd427.png)


# DCGAN

The DCGAN performs much better.
