import torch

class RBM:

    def __init__(self, visible_dim, hidden_dim):
        self.weights = torch.randn(hidden_dim, visible_dim)
        self.hidden_bias = torch.randn(1, hidden_dim)
        self.visible_bias = torch.randn(1, visible_dim)
    
    def sample_hidden(self, input):
        weighted_input = torch.mm(input, self.weights.t())
        activation = weighted_input + self.hidden_bias.expand_as(weighted_input)
        prob_h_given_v = torch.sigmoid(activation)
        return prob_h_given_v, torch.bernoulli(prob_h_given_v)

    def sample_visible(self, hidden_input):
        weighted_input = torch.mm(hidden_input, self.weights)
        activation = weighted_input + self.visible_bias.expand_as(weighted_input)
        prob_v_given_h = torch.sigmoid(activation)
        return prob_v_given_h, torch.bernoulli(prob_v_given_h)

    def train(self, initial_visible, curr_visible, initial_prob_h, curr_prob_h):
        self.weights += torch.mm(initial_visible.t(), initial_prob_h) - torch.mm(curr_visible.t(), curr_prob_h)
        self.visible_bias += torch.sum((initial_visible - curr_visible), 0)
        self.hidden_bias += torch.sum((initial_prob_h, curr_prob_h), 0)
