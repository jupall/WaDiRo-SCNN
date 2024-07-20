import matplotlib.pylab as pl
import numpy as np
import copy
import ot

#TODO: Add comments, Add multiprocessing
def sliced_wasserstein_outlier_introducer(signal, domain, in_sample_function, domain_dimensions, n_points, distance_min, n_projections, k_multiplier, L, seed, rng):
    domain_min = domain[0]
    domain_max = domain[1]
    copy_signal = copy.deepcopy(signal)
    for n in range(n_points):
        new_original_t = rng.uniform(domain_min, domain_max, size=(domain_dimensions))
        new_original_y = in_sample_function(new_original_t)
        new_point = np.hstack((new_original_t, new_original_y))
        temp_signal = np.vstack((copy_signal, new_point))
        print(temp_signal.shape)
        
        distance = 0
        k = 1
        while distance < distance_min:
            new_random_t =  rng.uniform(domain_min, domain_max, size=(1, domain_dimensions))
            new_random_y = rng.uniform(-L*k,L*k, size=(1,1))
            new_random_point = np.hstack((new_random_t, new_random_y))
            temp_perturb_signal = np.vstack((copy_signal, new_random_point))
            n = temp_perturb_signal.shape[0]
            a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
            dist = ot.sliced_wasserstein_distance(temp_perturb_signal, temp_signal, a, b, n_projections=n_projections, seed=seed)
            distance = np.mean(dist)
            k *= k_multiplier
        copy_signal = copy.deepcopy(temp_perturb_signal)
        
        
    return copy_signal



def sliced_wasserstein_outlier_filter():
    return None