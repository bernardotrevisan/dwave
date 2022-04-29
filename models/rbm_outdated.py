import numpy as np

class RBM:

    np.random.seed(10)

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 dataset,
                 initial_weights=None,
                 initial_visible_bias=None,
                 initial_hidden_bias=None,
                 dtype=np.float64):

        # Set internal datatype
        self.dtype = dtype

        # Set input and output dimension
        self.input_dim = number_visibles
        self.output_dim = number_hiddens

        # Transform dataset into appropriate inputs to the network
        dataset_input = dataset[:, :3, :]
        dataset_input_bin = []
        for i in range(len(dataset_input)):
            bin = []
            for j in range(3):
                for k in range(2):
                    bin.append([int(b) for b in list('{0:07b}'.format(int(dataset_input[i][j][k])))])
            dataset_input_bin.append(np.concatenate(np.array(bin)))
        dataset_input_bin = np.array(dataset_input_bin)
        self.dataset = dataset_input_bin

        # Initialize weights
        if initial_weights == None:
            self.weights = np.array((2.0 * np.random.rand(self.input_dim, self.output_dim) - 1.0) *
                                    (4.0 * np.sqrt(6.0 / (self.input_dim + self.output_dim))),
                                    dtype=dtype)
        else:
            self.weights = initial_weights

        # Think of a better way to initialize the visible and hidden biases
        if initial_visible_bias == None:
            self.visible_bias = np.array(self.inverse_sigmoid(np.random.rand(1, self.input_dim)), dtype=dtype)
        else:
            self.visible_bias = initial_visible_bias

        if initial_hidden_bias == None:
            self.hidden_bias = np.array(self.inverse_sigmoid(np.random.rand(1, self.input_dim)), dtype=dtype)
        else:
            self.hidden_bias = initial_hidden_bias

    def train(self):
        
#     def sample_v(self, v, beta=None, use_base_model=False):
#         return np.float64(v > np.random.random(v.shape))

#     def sample_h(self, h, beta=None, use_base_model=False):
#         return np.float64(h > np.random.random(h.shape))

#     def probability_v_given_h(self, h, beta=None, use_base_model=False):
#         activation = self._visible_pre_activation(h)
#         if beta is not None:
#             activation *= beta
#             if use_base_model is True:
#                 activation += (1.0 - beta) * self.bv_base
#         return self._visible_post_activation(activation)

#     def probability_h_given_v(self, v, beta=None, use_base_model=False):
#         activation = self._hidden_pre_activation(v)
#         if beta is not None:
#             activation *= beta
#         return self._hidden_post_activation(activation)
    
#     def _visible_pre_activation(self, h):
#         return np.dot(h - self.oh, self.w.T) + self.bv

#     def _visible_post_activation(self, pre_act_v):
#         return self.sigmoid(pre_act_v)

#     def _hidden_pre_activation(self, v):
#         return np.dot(v - self.ov, self.w) + self.bh

#     def _hidden_post_activation(self, pre_act_h):
#         return self.sigmoid(pre_act_h)
    
#     def _getbasebias(self):
#         save_mean = np.clip(np.float64(self._data_mean), 0.00001, 0.99999).reshape(1, self._data_mean.shape[1])
#         return np.log(save_mean) - np.log(1.0 - save_mean)

#     def unnormalized_log_probability_v(self,
#                                        v,
#                                        beta=None,
#                                        use_base_model=False):
#         temp_v = v - self.ov
#         activation = np.dot(temp_v, self.w) + self.bh
#         bias = np.dot(temp_v, self.bv.T).reshape(temp_v.shape[0], 1)
#         if beta is not None:
#             activation *= beta
#             bias *= beta
#             if use_base_model is True:
#                 bias += (1.0 - beta) * np.dot(temp_v, self.bv_base.T).reshape(temp_v.shape[0], 1)
#         return bias + np.sum(np.log(np.exp(activation * (1.0 - self.oh)) + np.exp(
#             -activation * self.oh)), axis=1).reshape(v.shape[0], 1)

#     def unnormalized_log_probability_h(self,
#                                        h,
#                                        beta=None,
#                                        use_base_model=False):
#         temp_h = h - self.oh
#         activation = np.dot(temp_h, self.w.T) + self.bv
#         bias = np.dot(temp_h, self.bh.T).reshape(h.shape[0], 1)
#         if beta is not None:
#             activation *= beta
#             bias *= beta
#             if use_base_model is True:
#                 activation += (1.0 - beta) * self.bv_base
#         return bias + np.sum(np.log(np.exp(activation * (1.0 - self.ov)) +
#                                         np.exp(-activation * self.ov)), axis=1).reshape(h.shape[0], 1)

#     def _base_log_partition(self, use_base_model=False):
#         if use_base_model is True:
#             return np.sum(np.log(np.exp(-self.ov * self.bv_base) + np.exp((1.0 - self.ov) * self.bv_base))
#                             ) + self.output_dim * np.log(2.0)
#         else:
#             return self.input_dim * np.log(2.0) + self.output_dim * np.log(2.0)

    def sigmoid(self, x):
        return 0.5 + 0.5 * np.tanh(0.5 * x)

    def inverse_sigmoid(self, x):
        return 2.0 * np.arctanh(2.0 * x - 1.0)

# class sampler:
#     np.random.seed(10)

#     def __init__(self, model, num_chains):
#         # Check and set the model
#         if not hasattr(model, 'probability_h_given_v'):
#             raise ValueError("The model needs to implement the function probability_h_given_v!")
#         if not hasattr(model, 'probability_v_given_h'):
#             raise ValueError("The model needs to implement the function probability_v_given_h!")
#         if not hasattr(model, 'sample_h'):
#             raise ValueError("The model needs to implement the function sample_h!")
#         if not hasattr(model, 'sample_v'):
#             raise ValueError("The model needs to implement the function sample_v!")
#         if not hasattr(model, 'input_dim'):
#             raise ValueError("The model needs to implement the parameter input_dim!")
#         self.model = model

#         # Initialize persistent Markov chains to Gaussian random samples.
#         if np.isscalar(num_chains):
#             self.chains = model.sample_v(np.random.randn(num_chains, model.input_dim) * 0.01)
#         else:
#             raise ValueError("Number of chains needs to be an integer or None.")

#     def sample(self,
#                num_samples,
#                k=1,
#                betas=None,
#                ret_states=True):
#         # Sample k times
#         for _ in range(k):
#             hid = self.model.probability_h_given_v(self.chains, betas)
#             hid = self.model.sample_h(hid, betas)
#             vis = self.model.probability_v_given_h(hid, betas)
#             self.chains = self.model.sample_v(vis, betas)
#         if ret_states:
#             samples = self.chains
#         else:
#             samples = vis

#         if num_samples == self.chains.shape[0]:
#             return samples
#         else:
#             # If more samples than chains,
#             repeats = np.int32(num_samples / self.chains.shape[0])

#             for _ in range(repeats):

#                 # Sample k times
#                 for _ in range(k):
#                     hid = self.model.probability_h_given_v(self.chains, betas)
#                     hid = self.model.sample_h(hid, betas)
#                     vis = self.model.probability_v_given_h(hid, betas)
#                     self.chains = self.model.sample_v(vis, betas)
#                 if ret_states:
#                     samples = np.vstack([samples, self.chains])
#                 else:
#                     samples = np.vstack([samples, vis])
#             return samples[0:num_samples, :]
    