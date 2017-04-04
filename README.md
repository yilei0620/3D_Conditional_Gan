# 3d_conditional_gan

We use Conditional Generative Adversarial Nets to process the reconstruction of 3D shape.

The data set is ModelNet40 (http://3dshapenets.cs.princeton.edu/). We translate the mesh data to 64 X 64 X 64 voxelized data, which is used for training our model.

The first goal is to reconstruct 3D objects according to an input label and a random 200-vector.

The codes in lib are heavily borrowed from [DCGAN](https://github.com/Newmu/dcgan_code).

The training process is:

Step1: Pre-train an Encoder-Decoder model based on the training set.

Step2: Use the trained Encoder-Decoder model to encode all the samples in the training set. Collect all the latent vectors of training samples and then find their distribution in the latent space by tting a multivariate Gaussian distribution. (BIG ASSUMPTION: the covariance matrix is diagonal which means the each dimensions are independent with other dimensions.)

Step3: Train the GAN model. Initialize the parameters of discriminator randomly. Initialize the parameters of generator by COPING the parameters of decoder. Different from usual GAN model, here we randomly generate noise vector z by the distribution found through Encoder-Decoder model in latent space. Carefully setting of learning rate is helpful to keep generator and discriminator trained in the same pace. If one is learning much ahead than another, it will make another's gradient keep large and not decay.

The examples of our GAN model:
![alt tag](https://github.com/yilei0620/3D_Conditional_Gan/blob/master/example.png)