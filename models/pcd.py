import numpy as np
from models import rbm as rbm_mod

class PCD:
    np.random.seed(10)

    def __init__(self, model, batch_size):
        # Set batch size
        self.batch_size = batch_size
        
        # Store model
        self.model = model
        
        self.rbm = rbm_mod.RBM(number_visibles = model.input + model.output, 
                                number_hiddens = model.hidden,  
                                initial_weights = np.vstack((model.W1, model.W2.T)), 
                                initial_visible_bias = np.hstack((model.b1, model.b3)), 
                                initial_hidden_bias = model.b2, 
                                initial_visible_offsets = np.hstack((model.o1, model.o3)), 
                                initial_hidden_offsets = model.o2,
                                data = None)
        
        self.sampler = rbm_mod.sampler(self.rbm, self.batch_size)

    def train(self, data, epsilon, k=[3,1]):
        #positive phase
        id1 = np.dot(data-self.model.o1, self.model.W1)
        d3 = np.copy(self.model.o3)
        d2 = np.copy(self.model.o2)
        for _ in range(k[0]):
            d2 = self.sigmoid(id1 + np.dot(d3 - self.model.o3, self.model.W2.T) + self.model.b2)
            d2 = np.float64(d2 > np.random.random(d2.shape))
            d3 = self.sigmoid(np.dot(d2 - self.model.o2, self.model.W2) + self.model.b3)
            d3 = np.float64(d3 > np.random.random(d3.shape))

        self.sampler.model = rbm_mod.RBM(number_visibles = self.model.input + self.model.output, 
                                        number_hiddens = self.model.hidden, 
                                        data = None, 
                                        initial_weights = np.vstack((self.model.W1, self.model.W2.T)), 
                                        initial_visible_bias = np.hstack((self.model.b1, self.model.b3)), 
                                        initial_hidden_bias = self.model.b2, 
                                        initial_visible_offsets = np.hstack((self.model.o1, self.model.o3)), 
                                        initial_hidden_offsets = self.model.o2)
        sample = self.sampler.sample(self.batch_size, k[1])
        self.m2 = self.sampler.model.probability_h_given_v(sample)
        self.m1 = sample[:, 0:self.model.input]
        self.m3 = sample[:, self.model.input:]
 
        # Estimate new means
        new_o1 = data.mean(axis=0)
        new_o2 = d2.mean(axis=0)
        new_o3 = d3.mean(axis=0)
        
        # Reparameterize
        self.model.b1 += epsilon[6] * np.dot(new_o2 - self.model.o2, self.model.W1.T)
        self.model.b2 += epsilon[5] * np.dot(new_o1 - self.model.o1, self.model.W1) + epsilon[7] * np.dot(new_o3 - self.model.o3, self.model.W2.T)
        self.model.b3 += epsilon[7] * np.dot(new_o2 - self.model.o2, self.model.W2)

        # Shift means
        self.model.o1 = (1.0 - epsilon[5]) * self.model.o1 + epsilon[5] * new_o1
        self.model.o2 = (1.0 - epsilon[6]) * self.model.o2 + epsilon[6] * new_o2
        self.model.o3 = (1.0 - epsilon[7]) * self.model.o3 + epsilon[7] * new_o3

        # Calculate gradients
        dW1 = (np.dot((data - self.model.o1).T, d2 - self.model.o2) - np.dot((self.m1 - self.model.o1).T, self.m2 - self.model.o2))
        dW2 = (np.dot((d2 - self.model.o2).T, d3 - self.model.o3) - np.dot((self.m2 - self.model.o2).T, self.m3 - self.model.o3))
        
        db1 = (np.sum(data - self.m1, axis = 0)).reshape(1, self.model.input)
        db2 = (np.sum(d2 - self.m2, axis = 0)).reshape(1, self.model.hidden)
        db3 = (np.sum(d3 - self.m3, axis = 0)).reshape(1, self.model.output)

        # Update Model
        self.model.W1 += epsilon[0] / self.batch_size*dW1
        self.model.W2 += epsilon[1] / self.batch_size*dW2
        
        self.model.b1 += epsilon[2] / self.batch_size*db1
        self.model.b2 += epsilon[3] / self.batch_size*db2
        self.model.b3 += epsilon[4] / self.batch_size*db3
    
    def sigmoid(self, x):
        return 0.5 + 0.5 * np.tanh(0.5 * x)
