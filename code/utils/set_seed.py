import numpy as np
import torch
import random


class SetSeed:
    """A class that sets the random seed for all the packages used in the
    project. You can either use it to set up a seed for a particular package or
    use the general set_seed method that will set the seed for all the
    packages"""

    def set_scikit_learn_seed(self):
        """Sets the random seed for scikit-learn that uses numpy in the
        background"""
        np.random.seed(42)

    def set_torch_seed(self):
        """Sets the random seed for pytorch"""
        torch.manual_seed(42)

    def set_python_random_seed(self):
        """Sets the random seed for python"""
        random.seed(42)

    def set_seed(self):
        """Sets the random seed for all the packages used in this project"""
        self.set_scikit_learn_seed()
        self.set_torch_seed()
        self.set_python_random_seed()


if __name__ == '__main__':
    SetSeed().set_seed()
