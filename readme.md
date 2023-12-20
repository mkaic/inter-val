# Inter-Val

The idea here is to weight the gradients of each training step depending on whether they cause the model to do better or worse on the *next* training batch. So there's now an additional layer of indirect supervision on top of the direct supervision. No idea if it will work, that's why I'm making this repo.

I'm training an autoencoder on the CelebA dataset because I think it's a nontrivial toy problem that will actually let me test my idea properly as opposed to something super easy like MNIST which isn't super helpful for testing novel ideas.

Adapted from my [basic_training](https://github.com/mkaic/basic_training) repo, which itself borrows a lot of code from [AntixK's](https://github.com/AntixK/PyTorch-VAE) VAE implementation. 