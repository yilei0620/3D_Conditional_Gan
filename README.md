# 3d_conditional_gan

We use Conditional Generative Adversarial Nets to process the reconstruction of 3D shape.

The data set is ModelNet40 (http://3dshapenets.cs.princeton.edu/). We translate the mesh data to 64 X 64 X 64 voxelized data, which is used for training our model.

The first goal is to reconstruct 3D objects according to an input label and a random 200-vector.