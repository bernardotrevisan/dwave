# Referenced to PyDeep for this implementation https://pydeep.readthedocs.io/en/latest/
import numpy as np
from models import pcd as pcd_mod
from models import estimator as estimator_mod

class DBM:
    np.random.seed(10)

    def __init__(self):
        # Set layer dimensions
        self.input = 42 # input layer: 7 nodes per coord, 2 coords per loc, 3 locs
        self.hidden = 64 # attention allocation layer: 4 attention levels, 3 characters, 64 combinations
        self.output = 14 # output (direction of movement) layer: 7 nodes per coord, 2 coords per loc, 1 locs

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

    def unnormalized_log_probability_x(self, x):
        # Generate all possibel binary codes for h1 and h2
        all_h1 = self.generate_binary_code(self.W2.shape[0], batch_size_exp=5)
        all_h2 = self.generate_binary_code(self.W2.shape[1], batch_size_exp=5)
        # Center variables
        xtemp = x-self.o1
        h1temp = all_h1-self.o2
        h2temp = all_h2-self.o3
        # Bias term
        bias = np.dot(xtemp, self.b1.T)
        # Both quadratic terms
        part1 = np.exp(np.dot(np.dot(xtemp, self.W1) + self.b2, h1temp.T))
        part2 = np.exp(np.dot(np.dot(h1temp, self.W2) + self.b3, h2temp.T))
        # Dot product of all combination of all quadratic terms + bias
        return bias + np.log(np.sum(np.dot(part1,part2), axis = 1).reshape(x.shape[0],1))

    def inverse_sigmoid(self, x):
        return 2.0 * np.arctanh(2.0 * x - 1.0)

    def generate_binary_code(self, bit_length, batch_size_exp=None, batch_number=0):
        # No batch size is given, all data is returned
        if batch_size_exp is None:
            batch_size_exp = bit_length
        batch_size = 2 ** batch_size_exp
        # Generate batch
        bit_combinations = np.zeros((batch_size, bit_length))
        for number in range(batch_size):
            dividend = number + batch_number * batch_size
            bit_index = 0
            while dividend != 0:
                bit_combinations[number, bit_index] = np.remainder(dividend, 2)
                dividend = np.floor_divide(dividend, 2)
                bit_index += 1
        return bit_combinations
    
