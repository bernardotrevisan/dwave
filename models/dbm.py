# Referenced to PyDeep for this implementation https://pydeep.readthedocs.io/en/latest/
import numpy as np
from models import pcd as pcd_mod
from models import estimator as estimator_mod

class DBM:
    np.random.seed(10)

    def __init__(self):
        # Set layer dimensions
        self.input = 54 # input layer: 9 nodes per coord, 2 coords per loc, 3 locs
        self.hidden = 64 # attention allocation layer: 4 attention levels, 3 characters, 64 combinations
        self.output = 18 # output (direction of movement) layer: 9 nodes per coord, 2 coords per loc, 1 locs

        # Create training set
        self.train_set = self.generate_bars_and_stripes(self.input, 100)

        # Training info
        self.batch_size = self.train_set.shape[0]
        self.epochs = 100
        self.k_pos = 3 # positive phase
        self.k_neg = 5 # negative phase
        self.epsilon = 0.005 * np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

        # Initialize weights
        # TODO: Specify initial weights according to apriori knowledge about the problem
        self.W1 = np.random.randn(self.input, self.hidden) * 0.01
        self.W2 = np.random.randn(self.hidden, self.output) * 0.01

         # Initialize offsets
        self.o1 = np.mean(self.train_set, axis=0).reshape(1, self.input)
        self.o2 = np.zeros((1, self.hidden)) + 0.5
        self.o3 = np.zeros((1, self.output)) + 0.5

        # Initialize biases
        self.b1 = self.inverse_sigmoid(np.clip(self.o1, 0.001, 0.999))
        self.b2 = self.inverse_sigmoid(np.clip(self.o2, 0.001, 0.999))
        self.b3 = self.inverse_sigmoid(np.clip(self.o3, 0.001, 0.999))

        # Initialize negative Markov chain
        self.m1 = self.o1 + np.zeros((self.batch_size, self.input))
        self.m2 = self.o2 + np.zeros((self.batch_size, self.hidden))
        self.m3 = self.o3 + np.zeros((self.batch_size, self.output))

    def train(self):
        trainer = pcd_mod.PCD(self, self.batch_size)

        # Set AIS betas / inv. temps for AIS
        a = np.linspace(0.0, 0.5, 100+1)
        a = a[0:a.shape[0]-1]
        b = np.linspace(0.5, 0.9, 800+1)
        b = b[0:b.shape[0]-1]
        c = np.linspace(0.9, 1.0, 2000)
        betas = np.hstack((a,b,c))

        # Start time measure and training
        for epoch in range(0, self.epochs+1):
            # update model
            for i in range(0, self.train_set.shape[0], self.batch_size):
                trainer.train(data=self.train_set[i:i + self.batch_size, :],
                                                 epsilon=self.epsilon,
                                                 k=[self.k_pos,self.k_neg])

            # estimate every 10 epochs
            if epoch % 10 == 0:
                print("Epoche: ", epoch)
                logZ, logZ_up, logZ_down = estimator_mod.partition_function_AIS(trainer.model, betas=betas)
                train_LL = np.mean(estimator_mod.LL_lower_bound(trainer.model, self.train_set, logZ))
                print("AIS  LL: ",2**(np.sqrt(self.input) + 1)* train_LL)

                logZ = estimator_mod.partition_function_exact(trainer.model)
                train_LL = np.mean(estimator_mod.LL_exact(trainer.model, self.train_set, logZ))
                print("True LL: ",2**(np.sqrt(self.input) + 1)* train_LL)
                print()
        
    def generate_bars_and_stripes(self, length, num_samples):
        data = np.zeros((num_samples, length))
        for i in range(num_samples):
            data[i, :] = np.random.randint(low=0, high=2, size=length)
        return data

    def inverse_sigmoid(self, x):
        return 2.0 * np.arctanh(2.0 * x - 1.0)
    
