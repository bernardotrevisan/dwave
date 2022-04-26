import numpy as np
from models import rbm as rbm_mod

def partition_function_exact(model, batchsize_exponent='AUTO'):

    # We transform the DBM to an RBM with restricted connections.
    rbm = rbm_mod.RBM(number_visibles = model.input + model.output, 
                        number_hiddens = model.hidden,
                        initial_weights=np.vstack((model.W1,model.W2.T)),
                        initial_visible_bias=np.hstack((model.b1,model.b3)),
                        initial_hidden_bias=model.b2,
                        initial_visible_offsets=np.hstack((model.o1,model.o3)),
                        initial_hidden_offsets=model.o2,
                        data=None)
    return partition_function_factorize_h(rbm)

def partition_function_AIS(model, num_chains = 100, k = 1, betas = 10000, status = False):
    # We transform the DBM to an RBM with restricted connections.
    rbm = rbm_mod.RBM(number_visibles = model.input + model.output,
                        number_hiddens = model.hidden,
                        initial_weights=np.vstack((model.W1,model.W2.T)),
                        initial_visible_bias=np.hstack((model.b1,model.b3)),
                        initial_hidden_bias=model.b2,
                        initial_visible_offsets=np.hstack((model.o1,model.o3)),
                        initial_hidden_offsets=model.o2,
                        data=None)
    # Run AIS for the transformed DBM
    return annealed_importance_sampling(model = rbm,
                                        num_chains =  num_chains,
                                        k = k, betas= betas,
                                        status = status)

def partition_function_factorize_h(model,
                                   beta=None,
                                   batchsize_exponent='AUTO',
                                   status=False):
    if status is True:
        print("Calculating the partition function by factoring over h: ")
        print('%3.2f%%' % 0.0)

    bit_length = model.output_dim
    if batchsize_exponent == 'AUTO' or batchsize_exponent > 20:
        batchsize_exponent = np.min([model.output_dim, 12])
    batchsize = np.power(2, batchsize_exponent)
    #num_combinations = 2 ** bit_length
    num_combinations = 1024

    num_batches = num_combinations // batchsize
    log_prob_vv_all = np.zeros(num_combinations)

    for batch in range(1, num_batches + 1):
        # Generate current batch
        bitcombinations = generate_binary_code(bit_length, batchsize_exponent, batch - 1)

        # calculate LL
        log_prob_vv_all[(batch - 1) * batchsize:batch * batchsize] = model.unnormalized_log_probability_h(
            bitcombinations, beta).reshape(bitcombinations.shape[0])

        # print status if wanted
        if status is True:
            print('%3.2f%%' % (100 * np.double(batch) / np.double(num_batches)))

    # return the log_sum of values
    return log_sum_exp(log_prob_vv_all)

def generate_binary_code(bit_length, batch_size_exp=None, batch_number=0):
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

def annealed_importance_sampling(model,
                                 num_chains=100,
                                 k=1,
                                 betas=10000,
                                 status=False):
    # Setup temerpatures if not given
    if np.isscalar(betas):
        betas = np.linspace(0.0, 1.0, betas)

    # Sample the first time from the base model
    v = model.probability_v_given_h(np.zeros((num_chains, model.output_dim)), betas[0], True)
    v = model.sample_v(v, betas[0], True)

    # Calculate the unnormalized probabilties of v
    lnpvsum = -model.unnormalized_log_probability_v(v, betas[0], True)

    if status is True:
        t = 1
        print("Calculating the partition function using AIS: ")
        print('%3.2f%%' % 0.0)
        print('%3.2f%%' % (100.0 * np.double(t) / np.double(betas.shape[0])))

    for beta in betas[1:betas.shape[0] - 1]:

        if status is True:
            t += 1
            print('%3.2f%%' % (100.0 * np.double(t) / np.double(betas.shape[0])))
        # Calculate the unnormalized probabilties of v
        lnpvsum += model.unnormalized_log_probability_v(v, beta, True)

        # Sample k times from the intermidate distribution
        for _ in range(0, k):
            h = model.sample_h(model.probability_h_given_v(v, beta, True), beta, True)
            v = model.sample_v(model.probability_v_given_h(h, beta, True), beta, True)

        # Calculate the unnormalized probabilties of v
        lnpvsum -= model.unnormalized_log_probability_v(v, beta, True)

    # Calculate the unnormalized probabilties of v
    lnpvsum += model.unnormalized_log_probability_v(v, betas[betas.shape[0] - 1], True)

    lnpvsum = np.longdouble(lnpvsum)

    # Calculate an estimate of logz .
    logz = log_sum_exp(lnpvsum) - np.log(num_chains)

    # Calculate +/- 3 standard deviations
    lnpvmean = np.mean(lnpvsum)
    lnpvstd = np.log(np.std(np.exp(lnpvsum - lnpvmean))) + lnpvmean - np.log(num_chains) / 2.0
    lnpvstd = np.vstack((np.log(3.0) + lnpvstd, logz))

    # Calculate partition function of base distribution
    baselogz = model._base_log_partition(True)

    # Add the base partition function
    logz = logz + baselogz
    logz_up = log_sum_exp(lnpvstd) + baselogz
    logz_down = log_diff_exp(lnpvstd) + baselogz

    if status is True:
        print('%3.2f%%' % 100.0)

    return logz, logz_up, logz_down

def LL_exact(model, x, lnZ):
    return model.unnormalized_log_probability_x(x) - lnZ

def LL_lower_bound(model, x, lnZ, conv_thres= 0.0001, max_iter=1000):
    # Pre calc activation from x since it is constant
    id1 = np.dot(x-model.o1,model.W1)
    # Initialize mu3 with its mean
    d2 = np.zeros((x.shape[0],model.hidden))
    # While convergence of max number of iterations not reached,
    # run mean field estimation
    for i in range(x.shape[0]):
        d3_temp = np.copy(model.o3)
        d2_temp = 0.0
        d2_new = sigmoid(id1[i,:] + np.dot(d3_temp-model.o3,model.W2.T) + model.b2)
        d3_new = sigmoid(np.dot(d2_new-model.o2,model.W2) + model.b3)
        while np.max(np.abs(d2_new-d2_temp )) > conv_thres:
            d2_temp  = d2_new
            d3_temp  = d3_new
            d2_new = sigmoid( id1[i,:]  + np.dot(d3_new-model.o3,model.W2.T) + model.b2)
            d3_new = sigmoid(np.dot(d2_new-model.o2,model.W2) + model.b3)
        d2[i] = np.clip(d2_new,0.0000000000000001,0.9999999999999999).reshape(1,model.hidden)

    # Foactorize over h2
    xtemp = x-model.o1
    h1temp = d2-model.o2
    e2 = np.sum(np.log(np.exp(-(np.dot(h1temp, model.W2)+model.b3)*(model.o3))+np.exp((np.dot(h1temp, model.W2)+model.b3)*(1.0-model.o3))), axis = 1).reshape(x.shape[0],1)
    e1 =  np.dot(xtemp, model.b1.T)\
        + np.dot(h1temp, model.b2.T) \
        + np.sum(np.dot(xtemp, model.W1) * h1temp ,axis=1).reshape(h1temp.shape[0], 1) + e2
    # Return energy of states + the entropy of h1 due to the mean field approximation
    return e1-lnZ - np.sum(d2*np.log(d2) + (1.0-d2)*np.log(1.0-d2) ,axis = 1).reshape(x.shape[0], 1)

def log_sum_exp(x, axis=0):
    alpha = x.max(axis) - np.log(np.finfo(np.float64).max) / 2.0
    if axis == 1:
        return np.squeeze(alpha + np.log(np.sum(np.exp(x.T - alpha), axis=0)))
    else:
        return np.squeeze(alpha + np.log(np.sum(np.exp(x - alpha), axis=0)))

def log_diff_exp(x, axis=0):
    alpha = x.max(axis) - np.log(np.finfo(np.float64).max) / 2.0
    if axis == 1:
        return np.squeeze(alpha + np.log(np.diff(np.exp(x.T - alpha), n=1, axis=0)))
    else:
        return np.squeeze(alpha + np.log(np.diff(np.exp(x - alpha), n=1, axis=0)))

def sigmoid(x):
        return 0.5 + 0.5 * np.tanh(0.5 * x)