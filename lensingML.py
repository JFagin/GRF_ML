# Written by: Joshua Fagin
# This code is for the paper "Measuring the substructure mass power spectrum of 23 SLACS strong galaxy–galaxy lenses with convolutional neural networks"


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from astropy.io import fits
import glob
from random import shuffle,randint
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow_probability.python.layers import DenseFlipout, Convolution2DFlipout
import tensorflow_probability as tfp
import scipy
import scipy.fftpack
import pandas as pd
from time import time
import gc
from matplotlib.patches import Rectangle
from tqdm import tqdm

import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.Profiles.interpolation import Interpol
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.kernel_util as kernel_util
from random import uniform
import scipy.signal as signal
import scipy



#For plot formatting
size = 13
plt.rc('font', size=size)          
plt.rc('axes', titlesize=size)     
plt.rc('axes', labelsize=size)   
plt.rc('xtick', labelsize=size)   
plt.rc('ytick', labelsize=size)    
plt.rc('legend', fontsize=size)    
plt.rc('figure', titlesize=size) 

tick_length_major = 7
tick_length_minor = 3
tick_width = 1


def load_image(file_path):
    with fits.open(file_path,memmap=False) as hdul:
        data = hdul[0].data
    return data



plot = False


#load the data with astropy, loads all files ending with .fits
#create_data_set11
#create_data_set_augment11

path_list = ["../MOLET/molet/output/create_data_set/molet_inputs/output", 
             ]


file_names = []
for path in path_list:
    file_names += glob.glob(f'{path}/lensed_image_galaxy*.fits')

print(len(file_names))

load = False #Load a pretrained model or traing from scratch, if true we do not train 
load_path = ""

A_max = -2
A_min = -5
beta_max = 8.0
beta_min = 3.0
num_classes=100


psf_name_list = glob.glob('psf/*/psf.fits') 
print(len(psf_name_list))

# parameter space of our mass model
q_max = 1.0
q_min = 0.5
pa_min = 0
pa_max = 180
x0_max = 0.1
x0_min = -0.1
y0_max = 0.1
y0_min = -0.1
# resolution of the image
resolution = 0.05
numPix = 110

def correlated_noise(sigma,L):
    # Generate n-by-n grid of spatially correlated noise
    width = numPix*resolution
    step_size = resolution
    x = np.arange(-width/2, width/2, step_size)
    y = np.arange(-width/2, width/2, step_size)
    xx, yy = np.meshgrid(x, y, sparse=True)
    filter_kernel = np.exp(-(xx**2+yy**2)/(2*L**2))
    #noise from normal distribution so sigma = 1, mu = 0
    random_noise = np.random.randn(len(x),len(x))
    noise = signal.fftconvolve(random_noise, filter_kernel, mode='same')
    noise = sigma*noise/noise.std()
    
    return noise


def add_lens_light(file):
    # We precomputed the lensed images, here we add the lens light to the images
    source = file[file.find("_source_")+8:file.find("_source_")+8+len('SDSSJ1630+4520')]
    theta_E = float(file[file.find("_theta_E")+len('_theta_E')+1:file.find("_q_")])
    q = float(file[file.find("_q_")+3:file.find("_pa_")])
    pa = float(file[file.find("_pa_")+4:file.find("_x0_")])
    x0 = float(file[file.find("_x0_")+4:file.find("_y0_")])
    y0 = float(file[file.find("_y0_")+4:file.find(".fits")])

    light_model_list = ['SERSIC_ELLIPSE']

    light_deviation = 0.025

    q_light = np.random.normal(q,(q_max-q_min)*light_deviation)
    pa_light = np.random.normal(pa,(pa_max-pa_min)*light_deviation)

    q_light = np.clip(q_light,q_min,q_max)
    pa_light = np.clip(pa_light,pa_min,pa_max)
    #convert from q,pa to e1,e2
    e1_light = (1.-q_light**2)/(1.+q_light**2) * np.cos(np.radians(pa_light))
    e2_light = (1.-q_light**2)/(1.+q_light**2) * np.sin(np.radians(pa_light))

    center_x_light = np.random.normal(x0,(x0_max-x0_min)*light_deviation)
    center_y_light = np.random.normal(y0,(y0_max-y0_min)*light_deviation)
    center_x_light = np.clip(center_x_light,x0_min,x0_max)
    center_y_light = np.clip(center_y_light,y0_min,y0_max)

    n_sersic = uniform(3.0,8.0)
    R_sersic = uniform(0.5,2.6)*theta_E
    kwargs_lens_light = [
        {'amp': 20, 'R_sersic': R_sersic, 'n_sersic': n_sersic, 'e1': e1_light, 'e2': e2_light, 
         'center_x': center_x_light, 'center_y': center_y_light}
    ]
    lensLightModel = LightModel(light_model_list=light_model_list)
    # evaluate the surface brightness of the unlensed coordinates
    x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=resolution)
    flux_lens_light = lensLightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light)
    flux_lens_light = util.array2image(flux_lens_light)
    flux_lens_light /=np.max(flux_lens_light)

    for psf_name in psf_name_list:
        if source in psf_name:
            psf = psf_name
    kernel = load_image(psf_name)
    
    flux_lens_light = signal.fftconvolve(flux_lens_light, kernel, mode='same')
    
    flux_lens_light /= np.max(flux_lens_light)


    image = load_image(file)

    image /=np.max(image)

    lens_light_ratio = 10.0**uniform(-0.1,2.0)
    image = image + lens_light_ratio*flux_lens_light
    image /= np.max(image) 

    L = uniform(0,0.05)
    
    background_rms = 10.0**uniform(-3.25,-2.25)

    noise = correlated_noise(background_rms, L)
    image = image + noise
    
    # Set the maximum value of the image to 1
    image /= np.max(image)
    
    return image


# We use 250,000 images, 230,000 for training and 20,000 for validation
# We also have a final test set of 25,000 images but I ran this after training.

num_files = min(250_000,len(file_names))
#randomize the file order
shuffle(file_names)
file_names = file_names[:num_files]
num_val = 20_000

num_copies = 1

# Note A is really log10(sigma^2) and beta is the slope of the power spectrum
num_val = num_val
image_train = []
A_train = []
beta_train = []
image_test = []
A_test = []
beta_test = []
A_train_labels = []
beta_train_labels = []
A_train_labels_test = []
beta_train_labels_test = []
file_names_test = file_names[:num_val]
file_names_train = file_names[num_val:]

print('Loading test set')

for j in tqdm(range(len(file_names_test))):
    file = file_names_test[j]
    for i in range(num_copies):
        image = add_lens_light(file)
        A = float(file[file.find("_SigmaPsi_")+len("_SigmaPsi_"):file.find("_beta_")])
        A = np.log10(A**2)
        beta = float(file[file.find("_beta_")+len("_beta_"):file.find("_theta")])
        image_test.append(image)
        A_test.append(A)     
        beta_test.append(beta) 

print('Loading training set')
for j in tqdm(range(len(file_names_train))):
    file = file_names_train[j]
    for i in range(num_copies):
        image = add_lens_light(file)
        A = float(file[file.find("_SigmaPsi_")+len("_SigmaPsi_"):file.find("_beta_")])
        A = np.log10(A**2)
        beta = float(file[file.find("_beta_")+len("_beta_"):file.find("_theta")])
        image_train.append(image)
        A_train.append(A)     
        beta_train.append(beta)
print()
print(f"{len(A_test)} test images")
print(f"{len(A_train)} training images")

# This is related to how large the uniform labels are
n_A = 0.9
n_beta = 0.9

def p_label(beta,A):
    # defines p for the labels
    P = (A-A_min)/(A_max-A_min)*(1-0.5*(beta-beta_min)/(beta_max-beta_min))
    p = 0.4*2**(-P)
    return np.clip(p,0,1)

x1, x2 = np.meshgrid(np.arange(beta_min,beta_max, (beta_max-beta_min)/5000),np.arange(A_min,A_max, (A_max-A_min)/5000))
p_grid = p_label(x1,x2)
plt.imshow(p_grid,extent=[beta_min,beta_max,A_min,A_max], origin='lower', aspect='auto')
plt.ylabel(r'$log_{10}(\sigma_{\delta\psi}^2)$')
plt.xlabel(r'$\beta$')
plt.colorbar(label='p')
plt.savefig('results/p_with_A_and_beta.pdf',bbox_inches='tight')
if plot:
    plt.show()
else:
    plt.close()

#create the labels
def binomial(n,A,beta):
    p = p_label(beta,A)
    
    Np = n*num_classes
    return np.random.binomial(round(Np),p)

def create_label(num,n,A,beta):
    label = np.zeros(num_classes)

    N_left = binomial(n,A,beta)
    N_right = binomial(n,A,beta)

    min_val = 3
    if N_left < min_val:
        N_left = min_val
    if N_right < min_val:
        N_right = min_val
    for j in range(num-N_left,num+N_right):
        if j > num_classes-1:
            j = num_classes-1
        elif j < 0:
            j = 0
        label[j] = 1.0
        
    #So we know where the true value is to get a MSE metric
    label[num] = 1.000001  
    label /= np.sum(label)

    return label


#create the test training labels
A_train_labels_test = []
beta_train_labels_test = []
for A,beta in zip(A_test,beta_test):
    #from 0 to num_classes-1 (0-49)
    A_num = round((num_classes-1)*np.clip((A-A_min)/(A_max-A_min),0,1))
    beta_num = round((num_classes-1)*np.clip((beta-beta_min)/(beta_max-beta_min),0,1))
    A_label = create_label(A_num,n_A,A,beta)
    beta_label = create_label(beta_num,n_beta,A,beta)
    A_train_labels_test.append(A_label)
    beta_train_labels_test.append(beta_label)


def plot_training_labels(num):
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(7,10))
    xaxis = np.linspace(A_min, A_max,num_classes)
    ax1.plot(xaxis,A_train_labels_test[num],label='prediction')
    ax1.axvline(A_test[num],linestyle='--')
    ax1.set_xlim(A_min,A_max)
    ax1.set_xlabel(r'$\log_{10}(\sigma_{\delta\psi}^2)$')
    ax1.set_ylabel("probability density")
    ax1.minorticks_on()
    ax1.tick_params(which='both',direction='in',top=True, right=True)

    xaxis = np.linspace(beta_min, beta_max,num_classes)
    ax2.plot(xaxis,beta_train_labels_test[num],label='prediction')
    ax2.axvline(beta_test[num],linestyle='--')
    ax2.set_xlim(beta_min,beta_max)
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("probability density")
    ax2.minorticks_on()
    ax2.tick_params(which='both',direction='in',top=True, right=True)

    plt.savefig(f"results/prediction_{num}.pdf",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
        

try:

    plot_training_labels(0)
    plot_training_labels(2)
    plot_training_labels(3)
    plot_training_labels(4)
except:
    plt.close('all')

try:
    #plot an example image
    num=0
    plt.imshow(image_train[num])
    print(A_train[num])
    print(beta_train[num])
    plt.colorbar()
    plt.savefig("results/example_image.pdf")
    if plot:
        plt.show()
    else:
        plt.close()
except:
    plt.close('all')


#plot a bunch of examples

try:

    xfig_num = 10
    yfig_num = 3
    fig, axs = plt.subplots(yfig_num,xfig_num, sharex=True,sharey=True, 
                            figsize=(2*xfig_num,2*yfig_num))
    plt.subplots_adjust(hspace=0,wspace=0)
    width = 5.5
    for i in range(yfig_num):
        for j in range(xfig_num):
            new_image = np.copy(image_train[i*xfig_num+j])
            new_image = np.clip(new_image,1e-5,1)
            new_image = np.log10(new_image)
            new_image = np.clip(new_image,-2,0)
            
            im = axs[i,j].imshow(new_image,extent=[-width/2,width/2,-width/2,width/2],
                                cmap="cividis",vmin = -2,vmax = 0)
            axs[i,j].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axs[i,j].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            
            scalebar = AnchoredSizeBar(axs[i,j].transData,
                                        1, r'$1\prime\prime$', 'lower left', 
                                        pad=0.01,
                                        borderpad=0.25,
                                        sep=4,
                                        color='white',
                                        frameon=False,
                                        size_vertical=0.02)
            if i ==0 and j==0:
                axs[i,j].add_artist(scalebar)
            
    for ax in axs.flat:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    for ax in axs.flat:
        ax.label_outer()
    cax = plt.axes([1.0, 0.15, 0.015, 0.7])
    cb = plt.colorbar(im,cax=cax)
    cb.set_label(label=r'$\log_{10}(I/I_{\mathrm{max}})$', size='18')
    cb.ax.tick_params(labelsize='large')
    plt.tight_layout()
    plt.savefig("results/Lensed_Image_Sample_new.pdf",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
except:
    plt.close('all')


A_train = np.array(A_train)
beta_train = np.array(beta_train)
A_test = np.array(A_test)
beta_test = np.array(beta_test)
A_train_labels_test = np.array(A_train_labels_test)
beta_train_labels_test = np.array(beta_train_labels_test)

#reshape for training
image_test = np.array(image_test)
if len(image_test.shape) == 3:
    image_test = image_test.reshape(len(image_test),len(image_test[0]),len(image_test[0]),1)

#reshape for training
image_train = np.array(image_train)
if len(image_train.shape) == 3:
    image_train = image_train.reshape(len(image_train),len(image_train[0]),len(image_train[0]),1)

def JS(Q, P):
    # This is the Jensen-Shannon divergence loss in tensorflow
    M = 0.5*(P+Q)
    KL1 = tf.keras.losses.kullback_leibler_divergence(P, M)
    KL2 = tf.keras.losses.kullback_leibler_divergence(Q, M)
    return 0.5*(KL1+KL2)

def H(Q,P):
    # This is the entropy loss in tensorflow
    return -K.mean(K.sum(P * K.log(P), axis=-1))

def combined_loss(y_true, y_pred):  
    # This is the combined loss function
    Q = K.clip(y_true, K.epsilon(), 1)
    P = K.clip(y_pred, K.epsilon(), 1) 
    
    lambda_val = 0.975 # #value from 0 to 1 choosing the relative weight of the terms in the loss function
    return lambda_val*JS(Q,P)+(1-lambda_val)*H(Q,P)

def MSE_metric(y_true, y_pred):
    #This is a MSE metric in tensorflow

    #This finds the ground truth argument using the trick in the training label
    max_true = K.argmax(y_true,axis=1)
    
    crng=np.arange(0,num_classes)
    crng2=np.expand_dims(crng,axis=1)
    crng3 = np.repeat(crng2,batch_size,axis=1)
    crng4 = K.transpose(K.constant(crng3))
    
    mean = K.sum(crng4*y_pred,axis=1)
    MSE = K.mean(K.square(K.cast(mean,'float32')-K.cast(max_true,'float32')))/(num_classes-1)**2
    return MSE

def entropy_metric(y_true, y_pred):  
    # This is the entropy metric in tensorflow

    Q = K.clip(y_true, K.epsilon(), 1)
    P = K.clip(y_pred, K.epsilon(), 1) 
    
    return H(Q,P)


#Randomly rotate and flip the test images. The training images are rotated and flipped in the generator.
for j in range(len(image_test)):
    #random rotation
    image_test[j] = np.rot90(image_test[j],randint(0,3))
    #random flip
    if randint(0,1) == 0:
        image_test[j] = np.flipud(image_test[j])
    if randint(0,1) == 0:
        image_test[j] = np.fliplr(image_test[j])


#number of epochs over which KL scaling stays 0 at the start of training
kl_start = 0
starting_value = 0.0
#number of epochs over which KL scaling is increased from 0 to 1
kl_annealtime = 100

cyclic_anneal = False # We did not use cyclic annealing in the final model but was testing this
num_cycles = 4
end_cycle = kl_start+2*num_cycles*kl_annealtime
min_weight = 1e-5 
class AnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight=tf.keras.backend.variable(starting_value), kl_start=kl_start, kl_annealtime=kl_annealtime,
                 cyclic_anneal=cyclic_anneal,end_cycle=end_cycle,min_weight=min_weight):
        self.weight = weight
        self.kl_start = kl_start
        self.kl_annealtime = kl_annealtime
        self.cyclic_anneal = cyclic_anneal
        self.end_cycle = end_cycle
        self.min_weight = min_weight
    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.kl_start:
            if self.cyclic_anneal:
                if epoch < self.end_cycle:
                    new_weight = ((epoch-self.kl_start) % (2*self.kl_annealtime)) / self.kl_annealtime
                    new_weight = min(1.,new_weight)
                else:
                    new_weight = 1.0
            else:
                new_weight = min((epoch-self.kl_start)/self.kl_annealtime, 1.)
            new_weight = max(new_weight,self.min_weight)
            tf.keras.backend.set_value(self.weight, new_weight)
        print("Current KL Weight is " + str(tf.keras.backend.get_value(self.weight)))


AC = AnnealingCallback()
w = AC.weight

kl_divergence_scaled = lambda q, p, _: w*tfp.distributions.kl_divergence(q, p)/tf.cast(len(file_names_train),dtype=tf.float32)

#See for example: https://github.com/WeilerWebServices/TensorFlow/blob/36eb6994d36674604973a06159e73187087f51c6/probability/tensorflow_probability/examples/models/bayesian_resnet.py
kernel_posterior_scale_mean=-9.0
kernel_posterior_scale_stddev=0.01
kernel_posterior_scale_constraint=0.2

reg = 1e-3
def _untransformed_scale_constraint(t):
    return tf.clip_by_value(t,-1000,tf.math.log(kernel_posterior_scale_constraint))

def L2(weights):
    return w*reg*tf.reduce_sum(tf.square(weights))

def L2_new(weights):
    return w*reg*tf.reduce_sum(tf.square(tf.math.exp(weights)))

kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
    untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
        mean=kernel_posterior_scale_mean,
        stddev=kernel_posterior_scale_stddev),
    #loc_regularizer= tf.keras.regularizers.L2(1e-5),
    untransformed_scale_constraint=_untransformed_scale_constraint,
    untransformed_scale_regularizer=L2_new)

#Learning rate for the Atam optimizer
learning_rate_initial = 1e-3

# Define the Squeeze and Excitation block
ratio = 8 
def se_block(x, num_filters, ratio=ratio):                     

    pool1 = GlobalAveragePooling2D()(x)
    flat = Reshape((1, 1, num_filters))(pool1)
    dense1 = DenseFlipout(num_filters//ratio, activation='relu',
                          kernel_posterior_fn=kernel_posterior_fn,
                          kernel_divergence_fn=kl_divergence_scaled)(flat)
    dense2 = DenseFlipout(num_filters, activation='sigmoid',
                          kernel_posterior_fn=kernel_posterior_fn,
                          kernel_divergence_fn=kl_divergence_scaled)(dense1)
    scale = multiply([x, dense2])

    return scale

# Define the residual block
def res_block(x, num_filters, strides,Resnet50):                             

    x_skip = x

    if strides > 1:

        if Resnet50:
            x_skip = Convolution2DFlipout(4*num_filters, kernel_size=(1, 1),padding='valid',strides=strides,
                                               kernel_posterior_fn=kernel_posterior_fn,
                                               kernel_divergence_fn=kl_divergence_scaled)(x_skip)
        else:
            x_skip = Convolution2DFlipout(num_filters, kernel_size=(1, 1),padding='valid',strides=strides,
                                               kernel_posterior_fn=kernel_posterior_fn,
                                               kernel_divergence_fn=kl_divergence_scaled)(x_skip)

    x_skip = BatchNormalization()(x_skip)

    if Resnet50:
        conv1 = Convolution2DFlipout(num_filters, kernel_size=(1, 1), padding='same',strides=strides,
                                     kernel_posterior_fn=kernel_posterior_fn,
                                     kernel_divergence_fn=kl_divergence_scaled)(x)
        norm1 = BatchNormalization()(conv1)
        relu1 = Activation('relu')(norm1)
        conv2 = Convolution2DFlipout(num_filters, kernel_size=(3, 3), padding='same',strides=1,
                                 kernel_posterior_fn=kernel_posterior_fn,
                                 kernel_divergence_fn=kl_divergence_scaled)(relu1)
        norm2 = BatchNormalization()(conv2)
        relu2 = Activation('relu')(norm2)
        conv3 = Convolution2DFlipout(4*num_filters, kernel_size=(1, 1), padding='same',strides=1,
                                     kernel_posterior_fn=kernel_posterior_fn,
                                     kernel_divergence_fn=kl_divergence_scaled)(relu2)
        norm3 = BatchNormalization()(conv3)
        se = se_block(norm3, num_filters=4*num_filters)
        #se = norm3
    else:
        conv1 = Convolution2DFlipout(num_filters, kernel_size=(3, 3), padding='same',strides=strides,
                                     kernel_posterior_fn=kernel_posterior_fn,
                                     kernel_divergence_fn=kl_divergence_scaled)(x)
        norm1 = BatchNormalization()(conv1)
        relu1 = Activation('relu')(norm1)
        conv2 = Convolution2DFlipout(num_filters, kernel_size=(3, 3), padding='same',strides=1,
                                 kernel_posterior_fn=kernel_posterior_fn,
                                 kernel_divergence_fn=kl_divergence_scaled)(relu1)
        norm2 = BatchNormalization()(conv2)
        se = se_block(norm2, num_filters=num_filters)
        #se = norm2
    sum = Add()([x_skip, se])
    relu2 = Activation('relu')(sum)

    return relu2

    
def initialize_model(summary=True):
    '''
    returns tensorflow model
    '''
    
    x_input = Input(image_train[0].shape)

    # Was also testing the Resnet50 architecture but did not use it in the final model    
    Resnet50 = False
    
    # Resnet18

    filters = [32,64,128,256] # smaller than the original Resnet18 to prevent overfitting

    if Resnet50:
        strides = [2, 2, 2, 2]
    else:
        strides = [1, 2, 2, 2] 
    num_layers = [2,2,2,2]

    x = Convolution2DFlipout(filters[0], kernel_size=(7, 7), activation='relu', padding='same',strides=2,
                             kernel_posterior_fn=kernel_posterior_fn,
                             kernel_divergence_fn=kl_divergence_scaled)(x_input)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)


    for i in range(len(filters)):
        for j in range(num_layers[i]):
            stride = strides[i] if j == 0 else 1
            x = res_block(x, filters[i],stride,Resnet50)
        
    x = GlobalAveragePooling2D()(x)

    A_layer = Dense(num_classes,activation='softmax',name='est_a',kernel_initializer='ones')(x) # We initialize the predicted labels uniformly across the classes

    beta_layer = Dense(num_classes,activation='softmax',name='est_b',kernel_initializer='ones')(x) # We initialize the predicted labels uniformly across the classes
  
    model = Model(inputs=x_input, outputs=[A_layer,beta_layer])
    
    # We combined the loss between our two parameters
    loss_weights = [0.5,0.5]
    optzr =  Adam(learning_rate=learning_rate_initial) 
    model.compile(loss=combined_loss,loss_weights=loss_weights,optimizer=optzr, metrics =[MSE_metric,entropy_metric]) 
    
    if summary:
        model.summary()

    return model



batch_size = 250 #Note the size of the training set must be divisible by the batch size

if load:
    # In case you need to load without training
    history = np.load(f'{load_path}/my_history.npy',allow_pickle='TRUE').item()

    loss = history['loss']
    val_loss = history['val_loss']
    mse_A = history['est_a_MSE_metric']
    mse_A_val = history['val_est_a_MSE_metric']
    mse_beta = history['est_b_MSE_metric']
    mse_beta_val = history['val_est_b_MSE_metric']


def MC_labels():
    #Function to make a new iteration of the training labels each epoch

    A_train_labels = []
    beta_train_labels = []
    
    #create the training labels
    for A,beta in zip(A_train,beta_train):
        A_num = round((num_classes-1)*np.clip((A-A_min)/(A_max-A_min),0,1))
        beta_num = round((num_classes-1)*np.clip((beta-beta_min)/(beta_max-beta_min),0,1))
        A_label = create_label(A_num,n_A,A,beta)
        beta_label =  create_label(beta_num,n_beta,A,beta)
        A_train_labels.append(A_label)
        beta_train_labels.append(beta_label)
    
    A_train_labels = np.array(A_train_labels)
    beta_train_labels = np.array(beta_train_labels)
    
    return A_train_labels,beta_train_labels

# number of epochs to train
epochs = 2000

if load:
    model = tf.keras.models.load_model(f"{load_path}/my_model",
                                       custom_objects={'combined_loss': combined_loss,
                                                       'MSE_metric':MSE_metric,
                                                       'entropy_metric':entropy_metric})
    model.summary()
    model.save("results/my_model")
else:
    model = initialize_model(summary=True)
    A_train_labels,beta_train_labels = MC_labels()
    class MySequence(tf.keras.utils.Sequence):

        def __init__(self,batch_size,x,y): # you can add parameters here
            self.batchSize = batch_size
            self.xTrain = x
            self.yTrain = y 

        def __len__(self):
            return self.xTrain.shape[0]//self.batchSize

        def __getitem__(self, index):
            #return self.xTrain[index*self.batchSize:(index+1)*self.batchSize:]
            x_batch = self.xTrain[index*self.batchSize:(index+1)*self.batchSize:]
            y_batch = []
            for label in self.yTrain:
                y_batch.append(label[index*self.batchSize:(index+1)*self.batchSize:])
    
            return x_batch,y_batch
      
        def on_epoch_end(self):
            
            for j in range(len(self.xTrain)):
                #random rotation
                self.xTrain[j] = np.rot90(self.xTrain[j],randint(0,3))
                #random flip
                if randint(0,1) == 0:
                    self.xTrain[j] = np.flipud(self.xTrain[j])
                if randint(0,1) == 0:
                     self.xTrain[j] = np.fliplr(self.xTrain[j])

            self.yTrain = MC_labels()
    
    class MySequence_val(tf.keras.utils.Sequence):

        def __init__(self,batch_size,x,y): # you can add parameters here
            self.batchSize = batch_size
            self.xTrain = x
            self.yTrain = y 

        def __len__(self):
            return self.xTrain.shape[0]//self.batchSize

        def __getitem__(self, index):
            #return self.xTrain[index*self.batchSize:(index+1)*self.batchSize:]
            x_batch = self.xTrain[index*self.batchSize:(index+1)*self.batchSize:]
            y_batch = []
            for label in self.yTrain:
                y_batch.append(label[index*self.batchSize:(index+1)*self.batchSize:])
            #y_batch = np.array(y_batch)
            return x_batch,y_batch
        def on_epoch_end(self):
            pass
    
    def scheduler(epoch, lr):
        return learning_rate_initial*10**(-epoch/epochs)

    schedule = LearningRateScheduler(scheduler)
    callbacks=[schedule,AnnealingCallback(w)]
    gen = MySequence(batch_size,image_train,MC_labels())
    gen_val = MySequence_val(batch_size,image_test,[A_train_labels_test,beta_train_labels_test])
    history = model.fit_generator(gen,
                                validation_data =gen_val,
                                epochs=epochs,
                                steps_per_epoch=int(len(image_train)/batch_size),
                                #validation_steps=int(len(image_test)/batch_size),
                                verbose=1,
                                shuffle=True,
                                callbacks=callbacks) 
    w = 1
    model.save("results/my_model")
    np.save('results/my_history.npy',history.history)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mse_A = history.history['est_a_MSE_metric']
    mse_A_val = history.history['val_est_a_MSE_metric']
    mse_beta = history.history['est_b_MSE_metric']
    mse_beta_val = history.history['val_est_b_MSE_metric']

#delete the training images to save memory
try:
    del image_train
except:
    pass

try:
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.legend(['training', 'validation'])
    plt.savefig('results/loss_vs_epoch.pdf',bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(mse_A,label=r"$\log_{10}(\sigma_{\delta\psi}^2)$ MSE",linestyle='dashed',color=cycle[0])
    plt.plot(mse_A_val,label=r"$\log_{10}(\sigma_{\delta\psi}^2)$ val MSE",color=cycle[0])
    plt.plot(mse_beta,label=r"$\beta$ MSE",linestyle='dashed',color=cycle[1])
    plt.plot(mse_beta_val,label=r"$\beta$ val MSE",color=cycle[1])
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.ylim(0,0.1)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.legend()
    plt.savefig("results/MSE_vs_epoch.pdf",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(mse_A,label=r"$\log_{10}(\sigma_{\delta\psi}^2)$ MSE",linestyle='dashed',color=cycle[0])
    plt.plot(mse_A_val,label=r"$\log_{10}(\sigma_{\delta\psi}^2)$ val MSE",color=cycle[0])
    plt.plot(mse_beta,label=r"$\beta$ MSE",linestyle='dashed',color=cycle[1])
    plt.plot(mse_beta_val,label=r"$\beta$ val MSE",color=cycle[1])
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.ylim(0,0.2)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.legend()
    plt.savefig("results/MSE_vs_epoch2.pdf",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
except:
    plt.close('all')
 



# test the amount of epistemic vs aleatoric uncertainty
#This is done by looking at the variance of the predictions for each test image
model = tf.keras.models.load_model("results/my_model",custom_objects={'combined_loss': combined_loss,
                                                                'MSE_metric':MSE_metric,
                                                                'entropy_metric':entropy_metric})

pred = model.predict(image_test)
A_pred = pred[0]
beta_pred = pred[1]

mean_A = np.sum(A_pred*np.expand_dims(np.linspace(A_min,A_max,num_classes),axis=0),axis=1)
mean_beta = np.sum(beta_pred*np.expand_dims(np.linspace(beta_min,beta_max,num_classes),axis=0),axis=1)

stdev_A = np.sqrt(np.sum(A_pred*( np.expand_dims(np.linspace(A_min,A_max,num_classes),axis=0) - np.expand_dims(mean_A,axis=1))**2, axis=1))
stdev_beta = np.sqrt(np.sum(beta_pred*( np.expand_dims(np.linspace(beta_min,beta_max,num_classes),axis=0) - np.expand_dims(mean_beta,axis=1))**2, axis=1))

num_val_iterations = 200

A_pred_list = []
beta_pred_list = []

for i in tqdm(range(num_val_iterations)):
    try:
        del model
    except:
        pass
    #This avoids a memory leak problem in tensorflow/Keras by saving a reloading the model every time
    gc.collect()
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model("results/my_model",custom_objects={'combined_loss': combined_loss,
                                                                        'MSE_metric':MSE_metric,
                                                                        'entropy_metric':entropy_metric})

    pred = model.predict(image_test)
    if i == 0:
        A_pred = pred[0]
        beta_pred = pred[1]
    else:
        A_pred += pred[0]
        beta_pred += pred[1]
    A_pred_list.append(pred[0][0])
    beta_pred_list.append(pred[1][0])

plt.close('all')

for i in range(10):
    plt.plot(np.linspace(A_min,A_max,num_classes),A_pred_list[i])
plt.xlim(A_min,A_max)
plt.ylim(0, 1.2*np.max(A_pred_list[0]))
plt.xlabel(r"$\log_{10}(\sigma_{\delta\psi}^2)$")
plt.ylabel("probability density")
plt.savefig("results/A_pred_sampled_TEST.pdf",bbox_inches='tight')
plt.close()

for i in range(10):
    plt.plot(np.linspace(beta_min,beta_max,num_classes),beta_pred_list[i])
plt.xlim(beta_min,beta_max)
plt.ylim(0, 1.2*np.max(beta_pred_list[0]))
plt.xlabel(r"$\beta$")
plt.ylabel("probability density")
plt.savefig("results/beta_pred_sampled_TEST.pdf",bbox_inches='tight')
plt.close()

A_pred /= num_val_iterations
beta_pred /= num_val_iterations

mean_A_sampled = np.sum(A_pred*np.expand_dims(np.linspace(A_min,A_max,num_classes),axis=0),axis=1)
mean_beta_sampled = np.sum(beta_pred*np.expand_dims(np.linspace(beta_min,beta_max,num_classes),axis=0),axis=1)

stdev_A_sampled = np.sqrt(np.sum(A_pred*( np.expand_dims(np.linspace(A_min,A_max,num_classes),axis=0) - np.expand_dims(mean_A_sampled,axis=1))**2, axis=1))
stdev_beta_sampled = np.sqrt(np.sum(beta_pred*( np.expand_dims(np.linspace(beta_min,beta_max,num_classes),axis=0) - np.expand_dims(mean_beta_sampled,axis=1))**2, axis=1))

#make the predictions
prob = True
num_val_iterations = 200
if prob:
    A_pred_list = []
    beta_pred_list = []
    for i in tqdm(range(num_val_iterations)):
        try:
            del model
        except:
            pass
        #This avoids a memory leak problem in tensorflow/Keras by saving a reloading the model every time
        gc.collect()
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model("results/my_model",custom_objects={'combined_loss': combined_loss,
                                                                      'MSE_metric':MSE_metric,
                                                                      'entropy_metric':entropy_metric})
        for k in range(image_test.shape[0]):
            if randint(0,1) == 0:
                image_test[k] = np.flipud(image_test[k])
            if randint(0,1) == 0:
                image_test[k] = np.fliplr(image_test[k])

            image_test[k] = np.rot90(image_test[k],randint(0,3))
        
        pred = model.predict(image_test)
        if i == 0:
            A_pred = pred[0]
            beta_pred = pred[1]
        else:
            A_pred += pred[0]
            beta_pred += pred[1]

        A_pred_list.append(pred[0][0])
        beta_pred_list.append(pred[1][0])

    plt.close('all')

    for i in range(10):
        plt.plot(np.linspace(A_min,A_max,num_classes),A_pred_list[i])
    plt.xlim(A_min,A_max)
    plt.ylim(0, 1.2*np.max(A_pred_list[0]))
    plt.xlabel(r"$\log_{10}(\sigma_{\delta\psi}^2)$")
    plt.ylabel("probability density")
    plt.savefig("results/A_pred_sampled_rot_TEST.pdf",bbox_inches='tight')
    plt.close()

    for i in range(10):
        plt.plot(np.linspace(beta_min,beta_max,num_classes),beta_pred_list[i])
    plt.xlim(beta_min,beta_max)
    plt.ylim(0, 1.2*np.max(beta_pred_list[0]))
    plt.xlabel(r"$\beta$")
    plt.ylabel("probability density")
    plt.savefig("results/beta_pred_sampled_rot_TEST.pdf",bbox_inches='tight')
    plt.close()

    A_pred /= num_val_iterations
    beta_pred /= num_val_iterations

else:
    model = tf.keras.models.load_model("results/my_model",custom_objects={'combined_loss': combined_loss,
                                                              'MSE_metric':MSE_metric,
                                                              'entropy_metric':entropy_metric})
    A_pred = model.predict(image_test)[0]
    beta_pred = model.predict(image_test)[1]      

mean_A_sampled_rot = np.sum(A_pred*np.expand_dims(np.linspace(A_min,A_max,num_classes),axis=0),axis=1)
mean_beta_sampled_rot = np.sum(beta_pred*np.expand_dims(np.linspace(beta_min,beta_max,num_classes),axis=0),axis=1)

stdev_A_sampled_rot = np.sqrt(np.sum(A_pred*(np.expand_dims(np.linspace(A_min,A_max,num_classes),axis=0) - np.expand_dims(mean_A_sampled_rot,axis=1))**2, axis=1))
stdev_beta_sampled_rot = np.sqrt(np.sum(beta_pred*(np.expand_dims(np.linspace(beta_min,beta_max,num_classes),axis=0) - np.expand_dims(mean_beta_sampled_rot,axis=1))**2, axis=1))

stdev_A_mean = np.mean(stdev_A)
stdev_beta_mean = np.mean(stdev_beta)
stdev_A_sampled_mean = np.mean(stdev_A_sampled)
stdev_beta_sampled_mean = np.mean(stdev_beta_sampled)
stdev_A_sampled_rot_mean = np.mean(stdev_A_sampled_rot)
stdev_beta_sampled_rot_mean = np.mean(stdev_beta_sampled_rot)

stdev_A_mean_frac = np.mean( (stdev_A_sampled-stdev_A) / stdev_A ) 
stdev_beta_mean_frac = np.mean( (stdev_beta_sampled-stdev_beta) / stdev_beta )

stdev_A_mean_rot_frac = np.mean( (stdev_A_sampled_rot-stdev_A) / stdev_A )
stdev_beta_mean_rot_frac = np.mean( (stdev_beta_sampled_rot-stdev_beta) / stdev_beta )

stdev_A_median_frac = np.median( (stdev_A_sampled-stdev_A) / stdev_A )
stdev_beta_median_frac = np.median( (stdev_beta_sampled-stdev_beta) / stdev_beta )

stdev_A_median_rot_frac = np.median( (stdev_A_sampled_rot-stdev_A) / stdev_A )
stdev_beta_median_rot_frac = np.median( (stdev_beta_sampled_rot-stdev_beta) / stdev_beta )


print(f"Mean A uncertainty 1: {stdev_A_mean}")
print(f"Mean beta uncertainty 1: {stdev_beta_mean}")
print(f"Mean A uncertainty sampled: {stdev_A_sampled_mean}")
print(f"Mean beta uncertainty sampled: {stdev_beta_sampled_mean}")
print(f"Mean A uncertainty sampled with rot: {stdev_A_sampled_rot_mean}")
print(f"Mean beta uncertainty sampled with rot: {stdev_beta_sampled_rot_mean}")
print(f"Mean A uncertainty frac: {stdev_A_mean_frac}")
print(f"Mean beta uncertainty frac: {stdev_beta_mean_frac}")
print(f"Mean A uncertainty rot frac: {stdev_A_mean_rot_frac}")
print(f"Mean beta uncertainty rot frac: {stdev_beta_mean_rot_frac}")
print(f"Median A uncertainty frac: {stdev_A_median_frac}")
print(f"Median beta uncertainty frac: {stdev_beta_median_frac}")
print(f"Median A uncertainty rot frac: {stdev_A_median_rot_frac}")
print(f"Median beta uncertainty rot frac: {stdev_beta_median_rot_frac}")

# save the metrics to text file
with open('results/uncertainty.txt','w') as f:
    f.write(f"Mean A uncertainty 1: {stdev_A_mean}\n")
    f.write(f"Mean beta uncertainty 1: {stdev_beta_mean}\n")
    f.write(f"Mean A uncertainty sampled: {stdev_A_sampled_mean}\n")
    f.write(f"Mean beta uncertainty sampled: {stdev_beta_sampled_mean}\n")
    f.write(f"Mean A uncertainty sampled with rot: {stdev_A_sampled_rot_mean}\n")
    f.write(f"Mean beta uncertainty sampled with rot: {stdev_beta_sampled_rot_mean}\n")
    f.write(f"Mean A uncertainty frac: {stdev_A_mean_frac}\n")
    f.write(f"Mean beta uncertainty frac: {stdev_beta_mean_frac}\n")
    f.write(f"Mean A uncertainty rot frac: {stdev_A_mean_rot_frac}\n")
    f.write(f"Mean beta uncertainty rot frac: {stdev_beta_mean_rot_frac}\n")
    f.write(f"Median A uncertainty frac: {stdev_A_median_frac}\n")
    f.write(f"Median beta uncertainty frac: {stdev_beta_median_frac}\n")
    f.write(f"Median A uncertainty rot frac: {stdev_A_median_rot_frac}\n")
    f.write(f"Median beta uncertainty rot frac: {stdev_beta_median_rot_frac}\n")


def plot_result(num):
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(7,10),sharey=True)
    
    max_prob = max(np.max(A_pred[num]),np.max(beta_pred[num]))
    
    xaxis = np.linspace(A_min, A_max,num_classes)
    ax1.plot(xaxis,A_pred[num],label='prediction')
    ax1.plot(xaxis,A_train_labels_test[num],linestyle='dotted',label='training label')
    ax1.axvline(A_test[num],linestyle='--',label='ground truth')
    ax1.set_xlabel(r"$\log_{10}(\sigma_{\delta\psi}^2)$")
    ax1.set_ylabel("probability density")
    ax1.set_xlim(A_min,A_max)
    ax1.set_ylim(0,1.1*max_prob)
    ax1.minorticks_on()
    ax1.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    ax1.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    
    ax1.legend()

    xaxis = np.linspace(beta_min, beta_max,num_classes)
    ax2.plot(xaxis,beta_pred[num],label='prediction')
    ax2.plot(xaxis,beta_train_labels_test[num],linestyle='dotted',label='training label')
    ax2.axvline(beta_test[num],linestyle='--',label='ground truth')
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("probability density")
    ax2.set_xlim(beta_min,beta_max)
    ax2.set_ylim(0,1.1*max_prob)
    ax2.minorticks_on()
    ax2.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    ax2.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    #ax2.legend()
    
    plt.savefig(f"results/prediction_{num}.pdf",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()


plot_result(0)
plot_result(1)
plot_result(2)
plot_result(7)
plot_result(8)
plot_result(12)
plot_result(16)
plot_result(20)
plot_result(51)
plot_result(100)
plot_result(101)
plot_result(104)
plot_result(105)
plot_result(110)
plot_result(111)
plot_result(150)
plot_result(155)
plot_result(160)
plot_result(165)



A_mean = []
A_width = []
beta_mean = []
beta_width = []
for A_pred_val,beta_pred_val in zip(A_pred,beta_pred):
    mean = np.sum(np.linspace(A_min, A_max,num_classes)*A_pred_val)
    #mean = np.sum(np.linspace(A_min, A_max,num_classes)*np.argmax(A_pred_val)/(num_classes-1))
    stdev = np.sqrt(np.sum(A_pred_val*(np.linspace(A_min, A_max,num_classes)-mean)**2))
    A_mean.append(mean)
    A_width.append(stdev)
    
    mean = np.sum(np.linspace(beta_min, beta_max,num_classes)*beta_pred_val)
    stdev = np.sqrt(np.sum(beta_pred_val*(np.linspace(beta_min, beta_max,num_classes)-mean)**2))
    beta_mean.append(mean)
    beta_width.append(stdev)

plt.plot(np.linspace(min(min(A_test),min(A_mean)),max(max(A_test),max(A_mean)), num=50),
         np.linspace(min(min(A_test),min(A_mean)),max(max(A_test),max(A_mean)), num=50))   
plt.hist2d(A_test,A_mean,bins=(20, 20),range=[[A_min,A_max],[A_min,A_max]])
plt.colorbar(label="number of lenses")
plt.ylabel("predicted "+"$\log_{10}(\sigma_{\delta\psi}^2)$")
plt.xlabel("true "+"$\log_{10}(\sigma_{\delta\psi}^2)$")
plt.xlim(A_min,A_max)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
plt.savefig("results/confusion_matrix_A.pdf",bbox_inches='tight')
plt.savefig("results/confusion_matrix_A_high_res.png",bbox_inches='tight',dpi=1200)
plt.savefig("results/confusion_matrix_A.png",bbox_inches='tight')
plt.savefig("results/confusion_matrix_A_high_res.pdf",bbox_inches='tight',dpi=1200)
if plot:
    plt.show()
else:
    plt.close()



diff = A_mean-A_test
try:
    np.save('results/A_test.npy',A_test)
except:
    pass
try:
    np.save('results/A_mean.npy',A_mean)
except:
    pass
y,x,_ = plt.hist(diff, bins=20) 
plt.xlabel(r"$\Delta \log_{10}(\sigma_{\delta\psi}^2)$")
plt.ylabel("number of lenses")
plt.xlim(-(A_max-A_min),A_max-A_min)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
mean = np.mean(diff)
stdev= np.std(diff)
textstr = '\n'.join((
    r'$\mu=%.2e$' % (mean, ),
    r'$\sigma=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
plt.text(0.35*np.max(x), 0.8*np.max(y), textstr,verticalalignment='top',
          bbox = dict(boxstyle = "square",facecolor='white',alpha = 1))
plt.savefig("results/diff_hist_A.pdf",bbox_inches='tight')
plt.savefig("results/diff_hist_A.png",bbox_inches='tight')
if plot:
    plt.show()
else:
    plt.close()

diff = A_mean-A_test
y,x,_ = plt.hist(diff, bins=20)
plt.xlabel(r"$\Delta \log_{10}(\sigma_{\delta\psi}^2)$")
plt.ylabel("number of lenses")
plt.xlim(-(A_max-A_min),A_max-A_min)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
mean = np.mean(diff)
stdev= np.std(diff)
plt.savefig("results/diff_hist_A_no_text.pdf",bbox_inches='tight')
plt.savefig("results/diff_hist_A_no_text.png",bbox_inches='tight')
if plot:
    plt.show()
else:
    plt.close()



plt.plot(np.linspace(min(min(beta_test),min(beta_mean)),max(max(beta_test),max(beta_mean)), num=50),
         np.linspace(min(min(beta_test),min(beta_mean)),max(max(beta_test),max(beta_mean)), num=50))    
plt.hist2d(beta_test,beta_mean,bins=(20, 20),range=[[beta_min,beta_max],[beta_min,beta_max]])
plt.colorbar(label="number of lenses")
plt.ylabel(r"predicted $\beta$")
plt.xlabel(r"true $\beta$")
plt.xlim(beta_min,beta_max)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
plt.savefig(f"results/confusion_matrix_beta.pdf",bbox_inches='tight')
plt.savefig(f"results/confusion_matrix_beta.png",bbox_inches='tight')
plt.savefig(f"results/confusion_matrix_beta_high_res.png",bbox_inches='tight',dpi=1200)
plt.savefig(f"results/confusion_matrix_beta_high_res.pdf",bbox_inches='tight',dpi=1200)
if plot:
    plt.show()
else:
    plt.close()


diff = beta_mean-beta_test

try:
    np.save('results/beta_test.npy',beta_test)
except:
    pass
try:
    np.save('results/beta_mean.npy',beta_mean)
except:
    pass

y,x,_ = plt.hist(diff, bins=20) 
plt.xlabel(r"$\Delta \beta$")
plt.ylabel("number of lenses")
plt.xlim(-(beta_max-beta_min),beta_max-beta_min)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
mean = np.mean(diff)
stdev= np.std(diff)
textstr = '\n'.join((
    r'$\mu=%.2e$' % (mean, ),
    r'$\sigma=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
plt.text(0.3*np.max(x), 0.8*np.max(y), textstr,verticalalignment='top',
          bbox = dict(boxstyle = "square",facecolor='white',alpha = 1))
plt.savefig("results/diff_hist_beta.pdf",bbox_inches='tight')
plt.savefig("results/diff_hist_beta.png",bbox_inches='tight')
if plot:
    plt.show()
else:
    plt.close()

diff = beta_mean-beta_test
y,x,_ = plt.hist(diff, bins=20)
plt.xlabel(r"$\Delta \beta$")
plt.ylabel("number of lenses")
plt.xlim(-(beta_max-beta_min),beta_max-beta_min)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
mean = np.mean(diff)
stdev= np.std(diff)
plt.savefig("results/diff_hist_beta_no_text.pdf",bbox_inches='tight')
plt.savefig("results/diff_hist_beta_no_text.png",bbox_inches='tight')
if plot:
    plt.show()
else:
    plt.close()

diff_A = A_mean-A_test
diff_beta = beta_mean-beta_test
plt.hist2d(diff_beta,diff_A, bins=(20,20)) 
mean = np.mean(diff_A)
stdev= np.std(diff_A)
textstr = '\n'.join((
    r'$\mu_{\sigma_{\delta\psi}}=%.2e$' % (mean, ),
    r'$\sigma_{\sigma_{\delta\psi}}=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
#plt.text(-.95*np.max(diff_beta), 0.9*np.max(diff_A), textstr,verticalalignment='top',color='white',
#          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
diff_beta = beta_mean-beta_test
mean = np.mean(diff_beta)
stdev= np.std(diff_beta)
textstr = '\n'.join((
    r'$\mu_\beta=%.2e$' % (mean, ),
    r'$\sigma_\beta=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
#plt.text(-0.95*np.max(diff_beta), -0.6*np.max(diff_A), textstr,verticalalignment='top',color='white',
#          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
plt.xlabel(r"$\Delta \beta$")
plt.ylabel(r"$\Delta \log_{10}(\sigma_{\delta\psi}^2)$")
plt.ylim(-(A_max-A_min),A_max-A_min)
plt.xlim(-(beta_max-beta_min),beta_max-beta_min)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
plt.colorbar(label="number of lenses",pad=0.02)
plt.savefig("results/diff_hist_2D_no_text.pdf",bbox_inches='tight')
plt.savefig("results/diff_hist_2D_no_text.png",bbox_inches='tight')
plt.savefig("results/diff_hist_2D_no_text_high_res.pdf",bbox_inches='tight',dpi=1200)
plt.savefig("results/diff_hist_2D_no_text_high_res.png",bbox_inches='tight',dpi=1200)
if plot:
    plt.show()
else:
    plt.close()

plt.hist2d(diff_beta,diff_A, bins=(20,20))
mean = np.mean(diff_A)
stdev= np.std(diff_A)
textstr = '\n'.join((
    r'$\mu_{\sigma_{\delta\psi}}=%.2e$' % (mean, ),
    r'$\sigma_{\sigma_{\delta\psi}}=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
#plt.text(-.95*np.max(diff_beta), 0.9*np.max(diff_A), textstr,verticalalignment='top',color='white',
#          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
diff_beta = beta_mean-beta_test
mean = np.mean(diff_beta)
stdev= np.std(diff_beta)
textstr = '\n'.join((
    r'$\mu_\beta=%.2e$' % (mean, ),
    r'$\sigma_\beta=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
#plt.text(-0.95*np.max(diff_beta), -0.6*np.max(diff_A), textstr,verticalalignment='top',color='white',
#          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
plt.xlabel(r"$\Delta \beta$")
plt.ylabel(r"$\Delta \log_{10}(\sigma_{\delta\psi}^2)$")
#plt.ylim(-(A_max-A_min),A_max-A_min)
#plt.xlim(-(beta_max-beta_min),beta_max-beta_min)
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
plt.colorbar(label="number of lenses",pad=0.02)
plt.savefig("results/diff_hist_2D_no_text2.pdf",bbox_inches='tight')
plt.savefig("results/diff_hist_2D_no_text2.png",bbox_inches='tight')
plt.savefig("results/diff_hist_2D_no_text2_high_res.png",bbox_inches='tight',dpi=1200)
plt.savefig("results/diff_hist_2D_no_text2_high_res.pdf",bbox_inches='tight',dpi=1200)
if plot:
    plt.show()
else:
    plt.close()


diff_A = A_mean-A_test
diff_beta = beta_mean-beta_test
plt.hist2d(diff_beta,diff_A, bins=(20,20)) 
mean = np.mean(diff_A)
stdev= np.std(diff_A)
textstr = '\n'.join((
    r'$\mu_{\sigma_{\delta\psi}}=%.2e$' % (mean, ),
    r'$\sigma_{\sigma_{\delta\psi}}=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
plt.text(-.95*np.max(diff_beta), 0.9*np.max(diff_A), textstr,verticalalignment='top',color='white',
          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
diff_beta = beta_mean-beta_test
mean = np.mean(diff_beta)
stdev= np.std(diff_beta)
textstr = '\n'.join((
    r'$\mu_\beta=%.2e$' % (mean, ),
    r'$\sigma_\beta=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
plt.text(-0.95*np.max(diff_beta), -0.6*np.max(diff_A), textstr,verticalalignment='top',color='white',
          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
plt.xlabel(r"$\Delta \beta$")
plt.ylabel(r"$\Delta A$")
plt.minorticks_on()
plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
plt.colorbar(label="number of lenses",pad=0.02)
plt.savefig("results/diff_hist_2D.pdf",bbox_inches='tight')
plt.savefig("results/diff_hist_2D.png",bbox_inches='tight')
if plot:
    plt.show()
else:
    plt.close()


rmse_A = np.sqrt(np.sum(((A_mean-A_test)/(A_max-A_min))**2)/len(A_mean))
rmse_beta = np.sqrt(np.sum(((beta_mean-beta_test)/(beta_max-beta_min))**2)/len(beta_mean))

print(f"SigmaPsi RMSE: {round(rmse_A*100,1)}%")
print(f"beta RMSE: {round(rmse_beta*100,1)}%")



fig, axs = plt.subplots(1, 3,figsize=(20,5))

axs[0].plot(np.linspace(min(A_test),max(A_test), num=20),
         np.linspace(min(A_test),max(A_test), num=20))   
hist = axs[0].hist2d(A_test,A_mean,bins=(20, 20))
fig.colorbar(hist[3],ax=axs[0],label="number lensed images")
axs[0].set_ylabel(r"predicted $A$")
axs[0].set_xlabel(r"true $A$")
axs[0].set_xlim(A_min,A_max)
axs[0].minorticks_on()
axs[0].tick_params(which='major',direction='out',top=True, right=True,length=tick_length_major,width=tick_width)
axs[0].tick_params(which='minor',direction='out',top=True, right=True,length=tick_length_minor,width=tick_width)

axs[1].plot(np.linspace(min(beta_test),max(beta_test), num=20),
         np.linspace(min(beta_test),max(beta_test), num=20))   
hist = axs[1].hist2d(beta_test,beta_mean,bins=(20, 20))
fig.colorbar(hist[3],ax=axs[1],label="number lensed images")
axs[1].set_ylabel(r"predicted $\beta$")
axs[1].set_xlabel(r"true $\beta$")
axs[1].set_xlim(beta_min,beta_max)
axs[1].minorticks_on()
axs[1].tick_params(which='major',direction='out',top=True, right=True,length=tick_length_major,width=tick_width)
axs[1].tick_params(which='minor',direction='out',top=True, right=True,length=tick_length_minor,width=tick_width)

diff_A = A_mean-A_test
diff_beta = beta_mean-beta_test
hist = plt.hist2d(diff_beta,diff_A, bins=(20,20)) 
mean = np.mean(diff_A)
stdev= np.std(diff_A)
textstr = '\n'.join((
    r'$\mu_A=%.2e$' % (mean, ),
    r'$\sigma_A=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
axs[2].text(-.75*np.max(diff_beta), 0.85*np.max(diff_A), textstr,verticalalignment='top',color='white',
          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
diff_beta = beta_mean-beta_test
mean = np.mean(diff_beta)
stdev= np.std(diff_beta)
textstr = '\n'.join((
    r'$\mu_\beta=%.2e$' % (mean, ),
    r'$\sigma_\beta=%.2e$' % (stdev, )))
#this is just for the to display the mean and width
axs[2].text(-0.75*np.max(diff_beta), -0.65*np.max(diff_A), textstr,verticalalignment='top',color='white',
          bbox = dict(boxstyle = "square",facecolor='none',edgecolor='none',alpha = 1))
axs[2].set_xlabel(r"$\Delta \beta$")
axs[2].set_ylabel(r"$\Delta A$")
axs[2].minorticks_on()
axs[2].tick_params(which='major',direction='inout',top=True, right=True,length=tick_length_major,width=tick_width)
axs[2].tick_params(which='minor',direction='inout',top=True, right=True,length=tick_length_minor,width=tick_width)
fig.colorbar(hist[3],ax=axs[2],label="number lensed images")
plt.savefig('results/Combined_confusion_matrices.pdf',bbox_inches='tight')
if plot:
    plt.show()
else:
    plt.close()
