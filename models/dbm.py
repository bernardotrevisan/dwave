# Referenced to PyDeep for this implementation https://pydeep.readthedocs.io/en/latest/
import numpy as np

class DBM:
    np.random.seed(10)

    def __init__(self):
        # Set layer dimensions
        self.input = 54 # input layer: 9 nodes per coord, 2 coords per loc, 3 locs
        self.hidden = 64 # attention allocation layer: 4 attention levels, 3 characters, 64 combinations
        self.output = 18 # output (direction of movement) layer: 9 nodes per coord, 2 coords per loc, 1 locs

        # Create training set
        self.train_set = self.generate_bars_and_stripes(input, 100)

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
        self.b1 = self.sigmoid(np.clip(self.o1, 0.001, 0.999))
        self.b2 = self.sigmoid(np.clip(self.o2, 0.001, 0.999))
        self.b3 = self.sigmoid(np.clip(self.o3, 0.001, 0.999))

        # Initialize negative Markov chain
        self.m1 = self.o1 + np.zeros((self.batch_size, self.input))
        self.m2 = self.o2 + np.zeros((self.batch_size, self.hidden))
        self.m3 = self.o3 + np.zeros((self.batch_size, self.output))

    def PCD(self):
        rbm = RBM(number_visibles = self.input + self.output, 
                    number_hiddens = self.hidden, 
                    initial_weights = np.vstack((self.W1, self.W2.T)), 
                    initial_visible_bias = np.hstack((self.b1, self.b3)), 
                    initial_hidden_bias = self.b2, 
                    initial_visible_offsets = np.hstack((self.o1, self.o3)), 
                    initial_hidden_offsets = self.o2)
        
        sampler = RBM_SAMPLER.PersistentGibbsSampler(self.rbm, self.batch_size)


    def sigmoid(self, input):
        return 2.0 * np.arctanh(2.0 * input - 1.0)
        
    def generate_bars_and_stripes(self, length, num_samples):
        data = np.zeros((num_samples, length * length))
        for i in range(num_samples):
            values = np.dot(np.random.randint(low=0, high=2,
                                                size=(length, 1)),
                            np.ones((1, length)))
            if np.random.random() > 0.5:
                values = values.T
            data[i, :] = values.reshape(length * length)
        return data

class RBM:
    np.random.seed(10)

    def __init__(self,
                number_visibles,
                number_hiddens,
                initial_weights,
                initial_visible_bias,
                initial_hidden_bias,
                initial_visible_offsets,
                initial_hidden_offsets):

        self.data_mean = 0.5 * np.ones((1, number_visibles), np.float64)
        self.data_std = np.ones((1, number_visibles), np.float64)

        self.ov = np.zeros((1, number_visibles))
        self.ov += initial_visible_offsets.reshape(1, number_visibles)
        self.ov = np.array(self.ov, dtype=np.float64)

        self.oh = np.zeros((1, number_hiddens))
        self.oh += initial_hidden_offsets.reshape(1, number_hiddens)
        self.oh = np.array(self.oh, dtype=np.float64)

        self.bv_base = self._getbasebias()
        self.fast_PT = True
    
    def _getbasebias(self):
        save_mean = np.clip(np.float64(self.data_mean), 0.00001, 0.99999).reshape(1, self.data_mean.shape[1])
        return np.log(save_mean) - np.log(1.0 - save_mean)

class PersistentGibbsSampler():
    def __init__(self, model, num_chains):
        # Check and set the model
        if not hasattr(model, 'probability_h_given_v'):
            raise ValueError("The model needs to implement the function probability_h_given_v!")
        if not hasattr(model, 'probability_v_given_h'):
            raise ValueError("The model needs to implement the function probability_v_given_h!")
        if not hasattr(model, 'sample_h'):
            raise ValueError("The model needs to implement the function sample_h!")
        if not hasattr(model, 'sample_v'):
            raise ValueError("The model needs to implement the function sample_v!")
        if not hasattr(model, 'input_dim'):
            raise ValueError("The model needs to implement the parameter input_dim!")
        self.model = model

        # Initialize persistent Markov chains to Gaussian random samples.
        if numx.isscalar(num_chains):
            self.chains = model.sample_v(numx.random.randn(num_chains, model.input_dim) * 0.01)
        else:
            raise ValueError("Number of chains needs to be an integer or None.")

    def sample(self,
               num_samples,
               k=1,
               betas=None,
               ret_states=True):
        """ Performs k steps persistent Gibbs-sampling.

        :param num_samples: The number of samples to generate.
                            .. Note:: Optimal performance is achieved if the number of samples and the number of chains
                            equal the batch_size.
        :type num_samples: int, numpy array

        :param k: The number of Gibbs sampling steps.
        :type k: int

        :param betas: Inverse temperature to sample from.(energy based models)
        :type betas: None, float, numpy array [num_betas,1]

        :param ret_states: If False returns the visible probabilities instead of the states.
        :type ret_states: bool

        :return: The visible samples of the Markov chains.
        :rtype: numpy array [num samples, input dimension]
        """
        # Sample k times
        for _ in range(k):
            hid = self.model.probability_h_given_v(self.chains, betas)
            hid = self.model.sample_h(hid, betas)
            vis = self.model.probability_v_given_h(hid, betas)
            self.chains = self.model.sample_v(vis, betas)
        if ret_states:
            samples = self.chains
        else:
            samples = vis

        if num_samples == self.chains.shape[0]:
            return samples
        else:
            # If more samples than chains,
            repeats = numx.int32(num_samples / self.chains.shape[0])

            for _ in range(repeats):

                # Sample k times
                for u in range(k):
                    hid = self.model.probability_h_given_v(self.chains, betas)
                    hid = self.model.sample_h(hid, betas)
                    vis = self.model.probability_v_given_h(hid, betas)
                    self.chains = self.model.sample_v(vis, betas)
                if ret_states:
                    samples = numx.vstack([samples, self.chains])
                else:
                    samples = numx.vstack([samples, vis])
            return samples[0:num_samples, :]
