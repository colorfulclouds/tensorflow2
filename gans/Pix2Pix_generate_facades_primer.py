
# coding: utf-8

# In[1]:

from keras.models import Sequential , Model
from keras.layers import Dense ,  BatchNormalization , Reshape , Input , Flatten
from keras.layers import Conv2D , MaxPool2D , Conv2DTranspose , UpSampling2D , ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU , PReLU
from keras.layers import Activation
from keras.layers import Dropout

from keras.layers import Concatenate

from keras.initializers import truncated_normal , constant , random_normal

from keras.optimizers import Adam , RMSprop

#残差块使用
from keras.layers import Add

from keras.datasets import mnist

#导入存在的模型
from keras.applications import VGG16 , VGG19


# In[2]:

import os

import matplotlib as plt
import numpy as np

import gc

from glob import glob

import keras.backend as K

import scipy

#get_ipython().magic('matplotlib inline')


# In[3]:

WIDTH = 256
HEIGHT = 256
CHANNEL = 3

SHAPE = (WIDTH , HEIGHT , CHANNEL)


LATENT_DIM = 100 #latent variable z sample from normal distribution

BATCH_SIZE = 4 #crazy!!! slow turtle
EPOCHS = 10

PATH = '../dataset/facades/'

#生成多少个图像 长*宽
ROW = 3
COL = 3

TRAIN_PATH = glob(PATH + 'train/*')
TEST_PATH = glob(PATH + 'test/*')
VAL_PATH = glob(PATH + 'val/*')

#卷积使用 基卷积核大小
G_filters = 64
D_filters = 64


# In[4]:

patch = int(HEIGHT/(2**4)) #16
disc_patch = (patch , patch , 1) #16*16*1


# In[5]:

def load_image(batch_size = BATCH_SIZE , training = True):
    #随机在图片库中挑选
    if training:
        IMAGES_PATH = TRAIN_PATH
    else:
        IMAGES_PATH = TEST_PATH
        
    images = np.random.choice(IMAGES_PATH , size=batch_size)
    
    original_facades = []
    faded_facades = []
    
    for i in images:
        image = scipy.misc.imread(i , mode='RGB').astype(np.float)
        
        height , width , channel = image.shape
        
        origin = image[: , :int(width/2) , :] #样本图像的左侧
        fade = image[: , int(width/2): , :] #样本图像的右侧
        
        #尽管原图像不是指定的大小 下面将强制将图像resize
        origin = scipy.misc.imresize(origin , size=(256,256))
        fade = scipy.misc.imresize(fade , size=(256,256))
        
        #随机性地对训练样本进行 左右反转
        if training and np.random.random()<0.5:
            origin = np.fliplr(origin)
            fade = np.fliplr(fade)
        
        original_facades.append(origin)
        faded_facades.append(faded_facades)
        
    original_facades = np.array(original_facades)/127.5 - 1
    faded_facades = np.array(faded_facades)/127.5 - 1
    
    return original_facades , faded_facades


def write_image(epoch):
    #生成高分图像时 进行对比显示
    original_facades , faded_facades = load_image(batch_size=3 , training=False)
    fake_faded_facades = generator_i.predict(faded_facades) #使用G来生成高分图像 使用低分图像生成原始的高分图像 但是难免有偏差 细节表现
    
    original_facades = original_facades*0.5+0.5
    faded_facades = faded_facades*0.5+0.5
    fake_faded_facades = fake_faded_facades*0.5+0.5
    
    
    fig , axes = plt.pyplot.subplots(ROW , COL)
    count=0
    
    axes[0][0].imshow(faded_facades[0])
    axes[0][0].set_title('faded')
    axes[0][0].axis('off')

    axes[0][1].imshow(original_facades[0])
    axes[0][1].set_title('original')
    axes[0][1].axis('off')
    
    axes[0][2].imshow(fake_faded_facades[0])
    axes[0][2].set_title('fake faded')
    axes[0][2].axis('off')

    axes[1][0].imshow(faded_facades[1])
    axes[1][0].set_title('faded')
    axes[1][0].axis('off')

    axes[1][1].imshow(original_facades[1])
    axes[1][1].set_title('original')
    axes[1][1].axis('off')
    
    axes[1][2].imshow(fake_faded_facades[1])
    axes[1][2].set_title('fake faded')
    axes[1][2].axis('off')

    axes[2][0].imshow(faded_facades[2])
    axes[2][0].set_title('faded')
    axes[2][0].axis('off')
    
    axes[2][1].imshow(original_facades[2])
    axes[2][1].set_title('original')
    axes[2][1].axis('off')
    
    axes[2][2].imshow(fake_faded_facades[2])
    axes[2][2].set_title('fake faded')
    axes[2][2].axis('off')
            
    fig.savefig('facades_pix2pix/No.%d.png' % epoch)
    plt.pyplot.close()
    
    
#    for i in range(ROW):
#        fig = plt.pyplot.figure()
#        plt.pyplot.imshow(low_resolution_image[i])
#        fig.savefig('celeba_srgan/No.%d_low_resolution%d.png' % (epoch , i))


# In[6]:

def load_batch(batch_size = BATCH_SIZE , training = True):
    #随机在图片库中挑选
    if training:
        IMAGES_PATH = TRAIN_PATH
    else:
        IMAGES_PATH = VAL_PATH
    
    batch_num = int(len(IMAGES_PATH) / batch_size)
    
    
    
    for i in range(batch_num-1):
        batch = IMAGES_PATH[i*batch_size : (i+1)*batch_size]
        
        original_facades = []
        faded_facades = []
        
        for image_name in batch:
            image = scipy.misc.imread(image_name , mode='RGB').astype(np.float)
        
            height , width , channel = image.shape
        
            origin = image[: , :int(width/2) , :] #样本图像的左侧
            fade = image[: , int(width/2): , :] #样本图像的右侧
        
            #尽管原图像不是指定的大小 下面将强制将图像resize
            origin = scipy.misc.imresize(origin , size=(256,256))
            fade = scipy.misc.imresize(fade , size=(256,256))
        
            #随机性地对训练样本进行 左右反转
            if training and np.random.random()<0.5:
                origin = np.fliplr(origin)
                fade = np.fliplr(fade)
        
            original_facades.append(origin)
            faded_facades.append(faded_facades)
        
        original_facades = np.array(original_facades)/127.5 - 1
        faded_facades = np.array(faded_facades)/127.5 - 1
    
        yield original_facades , faded_facades


# In[7]:

#==============


# In[8]:

def conv2d(input_data , output_size , filter_size=4 , batch_norm = True):
    h = Conv2D(output_size , filter_size , strides=(2,2) , padding='same')(input_data)
    h = LeakyReLU(alpha=0.2)(h)
    
    if batch_norm:
        h = BatchNormalization(momentum=0.8)(h)
    
    return h


#实现U-Net使用 需要网络的跳连接
def deconv2d(input_data , skip_input , output_size , filter_size=4 , dropout_rate=0.0):
    h = UpSampling2D(size=2)(input_data)
    h = Conv2D(output_size , filter_size , strides=(1,1) , padding='same')(h)
    h = Activation('relu')(h)
    
    if dropout_rate:
        h = Dropout(rate=dropout_rate)(h)
    
    h = BatchNormalization(momentum=0.8)(h)
    h =  Concatenate()([h , skip_input]) #跳连接具体实现

    return h
    


# In[9]:

#G使用encoder-decoder结构 但是需要引入跳连接 即U-Net
def generator(G_filters):
    #输入为faded的图像 输出为还原后的图像
    faded_facades = Input(shape=SHAPE)
    
    #encoder
    d1 = conv2d(faded_facades , G_filters , batch_norm=False)
    d2 = conv2d(d1 , G_filters*2)
    d3 = conv2d(d2 , G_filters*4)
    d4 = conv2d(d3 , G_filters*8)
    d5 = conv2d(d4 , G_filters*8)
    d6 = conv2d(d5 , G_filters*8)
    d7 = conv2d(d6 , G_filters*8)

    #decoder
    u1 = deconv2d(d7 , d6 , G_filters*8)
    u2 = deconv2d(u1 , d5 , G_filters*8)
    u3 = deconv2d(u2 , d4 , G_filters*8)
    u4 = deconv2d(u3 , d3 , G_filters*4)
    u5 = deconv2d(u4 , d2 , G_filters*2)
    u6 = deconv2d(u5 , d1 , G_filters)
    
    u7 = UpSampling2D(size=(2,2))(u6)
    original_facades = Conv2D(filters=CHANNEL , kernel_size=(4,4) , strides=(1,1) , padding='same' , activation='tanh')(u7) #还原后的图像
    
    return Model(faded_facades , original_facades , name='generator_Model')


# In[10]:

def discriminator(D_filters):
    original_facades = Input(shape=SHAPE) #原始图像
    faded_facades = Input(shape=SHAPE) #fade的图像
    
    original_faded = Concatenate()([original_facades , faded_facades])
    
    h1 = conv2d(original_faded , output_size=D_filters , batch_norm=False)
    h2 = conv2d(h1 , output_size=D_filters*2)
    h3 = conv2d(h2 , output_size=D_filters*4)
    h4 = conv2d(h3 , output_size=D_filters*8)
    
    validity =  Conv2D(1 , kernel_size=(4,4) , strides=(1,1) , padding='same')(h4)
    
    return Model([original_facades , faded_facades] , validity , name='discriminator_Model')


# In[11]:

adam = Adam(lr = 0.0002 , beta_1=0.5)

discriminator_i = discriminator(D_filters)
discriminator_i.compile(optimizer = adam , loss='mse' , metrics=['accuracy'])


generator_i = generator(G_filters)

original_facades = Input(shape=SHAPE)
faded_facades = Input(shape=SHAPE)

fake_original_facades = generator_i(faded_facades) #使用G来将faded的图像生成为original的图像

#freeze D
discriminator_i.trainable = False

validity = discriminator_i([original_facades , fake_original_facades])

combined = Model([original_facades , faded_facades] , [validity , fake_original_facades])
combined.compile(optimizer=adam , loss=['mse' , 'mae'] , loss_weights=[1 , 100])


# In[ ]:

#tuple类型相加 相当于cat连接
real_labels = np.ones(shape=(BATCH_SIZE , )+disc_patch) #真实样本label为1
fake_labels = np.zeros(shape=(BATCH_SIZE , )+disc_patch) #假样本label为0

for i in range(1001):
    print('1')
    original_facades_ , faded_facades_ = load_batch() #真实的原始图像和faded图像都是来自真实样本
    print('2')
    fake_original_facades_ = generator_i.predict(faded_facades_) #使用G生成真faded样本的原始样本
    print('*')
    #训练判别器
    real_loss = discriminator_i.train_on_batch([original_facades_ , faded_facades_] , real_labels) #使用真实的原始图像 训练 label全1
    fake_loss = discriminator_i.train_on_batch([fake_original_facades_ , faded_facades_] , fake_labels) #使用G生成的假的原始图像 训练 label全0 
    print('**')

    loss = np.add(real_loss , fake_loss)/2

    #训练生成器
    generator_loss = combined_model_i.train_on_batch([original_facades_ , faded_facades_] , [real_labels , original_facades_])
    print('***')
    print('epoch:%d loss:%f accu:%f gene_loss[mse]:%f gene_loss[mae]:%f' % (i , loss[0] , loss[1] , generator_loss[0] , generator_loss[1]))

    if i % 50 == 0:
        write_image(i)
    #write_image_mnist(i)
    
write_image(999)
#write_image_mnist(999)


# In[70]:

real_labels.shape


# In[ ]:

gc.collect()


# In[ ]:

gc.collect()


# In[2]:

VGG19(weights='imagenet')


# In[ ]:



