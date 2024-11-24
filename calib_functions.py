import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from joblib import delayed, Parallel

#Calculates the V matrix of quantile differences between adjacent PIT values. 
#The PIT values are given by valid_Fts, and the input CDFs are Gaussians with mean and variance from test_means, test_stds
def get_V(valid_Fts, test_means, test_stds):
  N = len(valid_Fts)
  Fts_tile = np.tile(valid_Fts.reshape(1,-1), (len(test_means),1))
  means_tile = np.tile(test_means.reshape(-1,1), (1, N))
  stds_tile = np.tile(test_stds.reshape(-1,1), (1, N))
  V_test = norm.ppf(Fts_tile, means_tile, stds_tile)
  return V_test

#Gets the PCE for the test CDFs given by the rows of V and C for the confidence levels in ps. 
def get_PCE(V, C, Fts, y):
  ubs = get_F_invs(V, C, Fts)
  phat = (ubs > np.tile(y.reshape(-1,1), (1, ubs.shape[1]))).mean(0)
  err = ((phat - Fts) ** 2).mean()
  return err

#Calculates the mean of the CDFs given by the rows of V and C. 
def get_means(V_test, C_test):
  N = C_test.shape[1]-1
  probs = C_test[:,1:(N+1)] - C_test[:,0:N]
  new_means = np.sum(V_test*probs, axis = 1)
  return new_means

#Calculates the ps-level quantiles of the CDFs given by the rows and V_test, C_test, using linear interpolation.
def get_F_invs(V_test, C_test, ps):
  F_invs = np.tile(V_test[:,0].reshape(-1,1), (1, len(ps)))
  idcs = np.apply_along_axis(
    np.searchsorted, 1, C_test, v = ps, side = "right")-1
  ps_tile = np.tile(ps.reshape(1,-1), (len(idcs),1))
  left_prob = np.take_along_axis(C_test, idcs, axis = 1)
  right_prob = np.take_along_axis(C_test, idcs+1, axis = 1)
  cond = (left_prob <= ps_tile)
  left_val = np.take_along_axis(V_test, idcs - 1, axis = 1)
  

  right_val = np.take_along_axis(V_test, idcs, axis = 1)
  new_val = left_val + ((ps_tile - left_prob)/(right_prob - left_prob))*(right_val - left_val)
  np.putmask(F_invs, cond, new_val)
  return F_invs

#Sorts the observations by PIT value, removes duplicates and values equal to 1 and 0
def filter_valid_data(valid_means, valid_stds, validy):
  valid_Fts = norm.cdf(validy, loc = valid_means, scale = valid_stds)
  
  unique_idcs = np.unique(valid_Fts, return_index = True)[1]
  valid_Fts = valid_Fts[unique_idcs]
  
  fil = ~(
    (valid_Fts == 1) 
    |  (valid_Fts == 0)
    # | (validy[unique_idcs] > 5)
  )
  valid_Fts = valid_Fts[fil]
  order = np.argsort(valid_Fts)
  
  return (valid_means[unique_idcs[fil][order]],
          valid_stds[unique_idcs[fil][order]],
          validy[unique_idcs[fil][order]],
          valid_Fts[order])

#Returns triangular matrices H and Hc that are used to compute h_j and w_j in the CRPS/binned CRPS/softweighted CRPS solution.
def get_triangular(valid_means, valid_stds, valid_Fts):
  N = len(valid_means)

  V = get_V(valid_Fts, valid_means, valid_stds)
  D = V[:,1:N] - V[:,0:(N-1)]

  ind_i = np.tile(np.arange(N).reshape(-1,1), (1, N-1))
  ind_j = np.tile(np.arange(N-1).reshape(1,-1), (N, 1))
  
  #We find using a strict inequality produces better results.
  H = D*(ind_j > ind_i)
  Hc = D*(ind_j <= ind_i)
  return H, Hc

#Computes the c vector for each test observation in the CRPS/binned CRPS/softweighted CRPS solution. 
#Note that for basic CRPS, wts_test is a matrix of ones. 
def get_C_triangular(H, Hc, wts_test):
  N = len(H)
  C = np.zeros((wts_test.shape[1], N+1))
  prodH = (H.T @ wts_test).T
  prodHc = (Hc.T @ wts_test).T
  C[:,1:N] = prodH/(prodH + prodHc)
  C[:,N] = 1
  return C

#Returns the recalibrated test CDFs using the basic CRPS method. 
#The test CDFs are discrete distributions with probability mass given by the rows of C at locations given by the rows of V.
def CRPS_single(valid_means, valid_stds, validy, test_means, test_stds):
  
  #we remove observations that produce duplicate PIT values in the validation dataset
  (valid_means, valid_stds, 
  validy, valid_Fts) = filter_valid_data(valid_means, valid_stds, validy)
  
  H,Hc = get_triangular(valid_means, valid_stds, valid_Fts)
  
  #In the basic CRPS method, we assign each observation to the same bin. 
  test_bin_numbers, valid_bin_numbers = np.zeros(len(test_means)).astype(int), np.zeros(len(valid_means)).astype(int)
  wts = get_bin_arr(valid_bin_numbers).astype(int).T
  wts_test = np.take(wts, test_bin_numbers, axis = 0).T
  C_test = get_C_triangular(H, Hc, wts_test)
  
  V_test = get_V(valid_Fts, test_means, test_stds)
  return V_test, C_test

#Calculate bin boundaries that divide x into bins of equal size. 
#Code taken from https://stackoverflow.com/questions/37649342/matplotlib-how-to-make-a-histogram-with-bins-of-equal-area
def equalObs(x, n_bins):
  return np.interp(np.linspace(0, len(x), n_bins + 1),
                    np.arange(len(x)),
                    np.sort(x))

#Returns a one-hot encoding of the bin numbers
def get_bin_arr(bin_numbers):
  M = len(bin_numbers)
  n_bins = np.max(bin_numbers) + 1
  idx_arr = np.tile(np.expand_dims(np.arange(n_bins), 0), (M,1))
  bin_numbers_tile = np.tile(np.expand_dims(bin_numbers,1), (1,n_bins))
  test_bins = (bin_numbers_tile == idx_arr)
  return test_bins

#Returns the bin number of each element of m
def get_bin_number(m, intervals):
  bin_numbers = np.ones(len(m))*-1
  m_tile = np.tile(np.expand_dims(m,1), (1, len(intervals)))
  intervals_tile = np.tile(np.expand_dims(intervals,0), (len(m),1))
  
  bin_numbers = np.argmin(m_tile - intervals_tile > 0, axis = 1) - 1
  bin_numbers[m <= intervals[0]] = 0
  bin_numbers[m >= intervals[-1]] = len(intervals) - 2
  
  return bin_numbers

#Computes bin boundaries based on valid_means and returns bin numbers for the valid_means and test_means
def get_test_valid_bin_numbers(test_means, valid_means, n_bins):
  valid_intervals = np.array(equalObs(valid_means, n_bins))
  test_bin_numbers = get_bin_number(test_means, valid_intervals)
  valid_bin_numbers = get_bin_number(valid_means, valid_intervals)
  return test_bin_numbers, valid_bin_numbers

#Returns the recalibrated test CDFs using the binned CRPS method. 
def CRPS_bin(valid_means, valid_stds, validy, test_means, test_stds, optimal_n_bins = None, n_jobs = 4):
  
  (valid_means, valid_stds, 
  validy, valid_Fts) = filter_valid_data(valid_means, valid_stds, validy)
  
  H,Hc = get_triangular(valid_means, valid_stds, valid_Fts)
  V_valid = get_V(valid_Fts, valid_means, valid_stds)
  
  #Find number of bins that minimizes the PCE on the validation dataset
  if optimal_n_bins is None:
    n_bins_arr = np.arange(1, 21, 1)
    rng = np.random.default_rng(seed = 0)
    rand_ps = rng.uniform(0, 1, size = 3000)

    #In the binned CRPS method, the weights are 1 if the observation belongs to the same bin and 0 otherwise.
    def helper_(i):
      n_bins = n_bins_arr[i]
      _, valid_bin_numbers = get_test_valid_bin_numbers(valid_means, valid_means, n_bins)
      
      wts = get_bin_arr(valid_bin_numbers).astype(int).T
      wts_valid = np.take(wts, valid_bin_numbers, axis = 0).T
      C_valid = get_C_triangular(H, Hc, wts_valid)
      return get_PCE(V_valid, C_valid, rand_ps, validy)

    PCEs = np.array([Parallel(n_jobs=n_jobs)(delayed(helper_)(i) for i in range(len(n_bins_arr)))])
    optimal_n_bins = n_bins_arr[np.argmin(PCEs)]
  # print(optimal_n_bins)
  #Return binned solution
  test_bin_numbers, valid_bin_numbers = get_test_valid_bin_numbers(test_means, valid_means, optimal_n_bins)
  wts = get_bin_arr(valid_bin_numbers).astype(int).T
  wts_test = np.take(wts, test_bin_numbers, axis = 0).T
  C_test = get_C_triangular(H, Hc, wts_test)
  
  V_test = get_V(valid_Fts, test_means, test_stds)  
  return V_test, C_test

def epa(t):
  return 0.75*(1 - t**2)*(t**2 <= 1).astype(int)

#Returns the recalibrated test CDFs using the soft-weighted CRPS method. 
def CRPS_weighted(valid_means, valid_stds, validy, test_means, test_stds, optimal_n_bins = None, n_jobs = 4):
  valid_means, valid_stds, validy, valid_Fts = filter_valid_data(valid_means, valid_stds, validy)
  H,Hc = get_triangular(valid_means, valid_stds, valid_Fts)  
  V_valid = get_V(valid_Fts, valid_means, valid_stds)
  
  valid_means_tile = np.tile(np.expand_dims(valid_means,0), (len(valid_means),1))
  valid_dists = np.abs(valid_means_tile - valid_means_tile.T)
  valid_dists_sorted = np.sort(valid_dists, axis = 1)
  
  # print('optimal_n_bins: ' + str(optimal_n_bins))
  if optimal_n_bins is None:
    rng = np.random.default_rng(seed = 0)  
    rand_ps = rng.uniform(0, 1, size = 3000)
    n_bins_arr = np.arange(20) + 1
    
    #In the soft-weighted CRPS method, the weights are given by the distance to the given mean.
    def helper_(i):
      n_bins = n_bins_arr[i]
      n_nbrs = int(valid_dists.shape[1] / n_bins) - 1
      dists_nn = valid_dists_sorted[:,0:int(n_nbrs)]
      max_dists = dists_nn[:,-1]
      max_dists_tile = np.tile(np.expand_dims(max_dists,1), (1, valid_dists.shape[1]))
      wts_test = epa(valid_dists/max_dists_tile).T
      C_test = get_C_triangular(H, Hc, wts_test)
      return get_PCE(V_valid, C_test, rand_ps, validy)
    PCEs = np.array([Parallel(n_jobs=n_jobs)(delayed(helper_)(i) for i in range(len(n_bins_arr)))])
    optimal_n_bins = n_bins_arr[np.argmin(PCEs)]
  # print(optimal_n_bins)
  
  test_dists = np.abs(np.tile(np.expand_dims(test_means, 1), (1,len(valid_means)))- np.tile(np.expand_dims(valid_means,0), (len(test_means),1)))
  test_dists_sorted = np.sort(test_dists, axis = 1)
  n_nbrs = int(test_dists.shape[1] / optimal_n_bins) - 1
  test_dists_nn = test_dists_sorted[:,0:int(n_nbrs)]
  max_test_dist_tile = np.tile(np.expand_dims(test_dists_nn[:,-1],1), (1, test_dists.shape[1]))
  wts_test = epa(test_dists/max_test_dist_tile).T
  C_test = get_C_triangular(H, Hc, wts_test)
  
  V_test = get_V(valid_Fts, test_means, test_stds)    
  return V_test, C_test

#Fills nan's with the previous value. 
def interp_nans(data):
  mask = np.isnan(data)
  idx = np.where(~mask,np.arange(len(mask)),0)
  idx = np.maximum.accumulate(idx)
  out = np.take(data, idx)
  return out

def kuleshov_helper(points):
  N = len(points)
  C_bin = np.ones(N)*np.nan
  n_points = points.sum()
  C_bin[points] = np.arange(1, n_points+1)/n_points
  C_bin = np.append(0, C_bin)
  C_bin = interp_nans(C_bin)
  return C_bin

#Returns the recalibrated test CDFs using the basic NLL method (quantile recalibration)
def NLL_single(valid_means, valid_stds, validy, test_means, test_stds):
  (valid_means, valid_stds, 
  validy, valid_Fts) = filter_valid_data(valid_means, valid_stds, validy)
  
  #All observations belong to the same bin.
  test_bin_numbers, valid_bin_numbers = np.zeros(len(test_means)).astype(int), np.zeros(len(valid_means)).astype(int)
  bin_arr = get_bin_arr(valid_bin_numbers)
  C_bin = np.apply_along_axis(kuleshov_helper, axis = 0, arr = bin_arr).T
  C_test = np.take(C_bin, test_bin_numbers, axis = 0)
  V_test = get_V(valid_Fts, test_means, test_stds)
  return V_test, C_test

#Returns the recalibrated test CDFs using the binned NLL method.
def NLL_bin(valid_means, valid_stds, validy, test_means, test_stds, optimal_n_bins = None, n_jobs = 4):
  
  (valid_means, valid_stds, 
  validy, valid_Fts) = filter_valid_data(valid_means, valid_stds, validy)
  
  V_valid = get_V(valid_Fts, valid_means, valid_stds)
  
  if optimal_n_bins is None:
    n_bins_arr = np.arange(1, 21, 1)
    rng = np.random.default_rng(seed = 0)
    rand_ps = rng.uniform(0, 1, size = 3000)

    def helper_(i):
      n_bins = n_bins_arr[i]
      _, valid_bin_numbers = get_test_valid_bin_numbers(valid_means, valid_means, n_bins)
      
      bin_arr = get_bin_arr(valid_bin_numbers)
      C_bin = np.apply_along_axis(kuleshov_helper, axis = 0, arr = bin_arr).T
      C_valid = np.take(C_bin, valid_bin_numbers, axis = 0)
      return get_PCE(V_valid, C_valid, rand_ps, validy)

    PCEs = np.array([Parallel(n_jobs=n_jobs)(delayed(helper_)(i) for i in range(len(n_bins_arr)))])
    optimal_n_bins = n_bins_arr[np.argmin(PCEs)]
  
  test_bin_numbers, valid_bin_numbers = get_test_valid_bin_numbers(test_means, valid_means, optimal_n_bins)
  bin_arr = get_bin_arr(valid_bin_numbers)
  C_bin = np.apply_along_axis(kuleshov_helper, axis = 0, arr = bin_arr).T
  C_test = np.take(C_bin, test_bin_numbers, axis = 0)
  
  V_test = get_V(valid_Fts, test_means, test_stds)  
  return V_test, C_test

def get_CRPS(V_test, C_test, testy):
  def helper_(i):
    V_i = V_test[i,:]
    C_i = C_test[i,:]
    test_i = testy[i]
    N = len(V_i)
    CRPS = (
      ((C_i[1:N] - (test_i <= V_i[0:(N-1)]).astype(float)) ** 2)
      *(V_i[1:N] - V_i[0:(N-1)])
    ).sum()
    return CRPS
  CRPSs = np.array([helper_(i) for i in range(len(testy))])
  return np.mean(CRPSs)

def get_CRPS_base(test_means, test_stds, testy):
  def helper_(i):
    mean_i = test_means[i]
    std_i = test_stds[i]
    test_i = testy[i]
    return std_i*(
      ((test_i - mean_i)/std_i)*(
        2*norm.cdf(((test_i - mean_i)/std_i)) - 1
      ) + 2*norm.pdf(((test_i - mean_i)/std_i)) - 1/np.sqrt(np.pi)
    )
  CRPSs = np.array([helper_(i) for i in range(len(testy))])
  return np.mean(CRPSs)

#Computes evaluation metrics of recalibrated test CDFs given by V_test and C_test.
def get_metrics(V_test, C_test, testy):
  ps = np.arange(0.05,1,0.05)
  test_means = get_means(V_test, C_test)
  test_F_invs = get_F_invs(V_test, C_test, ps)
  testy_tile = np.tile(testy.reshape(-1,1),(1, len(ps)))
  ps_tile = np.tile(np.expand_dims(ps,0), (len(testy), 1))

  MSE = ((test_means - testy) ** 2).mean()
  PCE = np.square((testy_tile < test_F_invs).mean(0)-ps).mean()
  PBL = (
      (test_F_invs > testy_tile)*(test_F_invs - testy_tile)*(1 - ps_tile) + 
      (test_F_invs < testy_tile)*(testy_tile - test_F_invs)*(ps_tile) 
    ).mean()
  CRPS = get_CRPS(V_test, C_test, testy)
  return np.array([MSE, PCE, PBL, CRPS])  

#Computes evaluation metrics of base forecaster.
def get_metrics_base(test_means, test_stds, testy):
  ps = np.arange(0.05,1,0.05)
  ps_tile = np.tile(ps.reshape(1,-1), (len(test_means),1))
  testy_tile = np.tile(testy.reshape(-1,1),(1, len(ps)))
  means_tile = np.tile(test_means.reshape(-1,1), (1, len(ps)))
  stds_tile = np.tile(test_stds.reshape(-1,1), (1, len(ps)))
  test_F_invs = norm.ppf(ps_tile, means_tile, stds_tile)
  MSE = ((test_means - testy) ** 2).mean()
  PCE = np.square((testy_tile < test_F_invs).mean(0)-ps).mean()
  PBL = (
      (test_F_invs > testy_tile)*(test_F_invs - testy_tile)*(1 - ps_tile) + 
      (test_F_invs < testy_tile)*(testy_tile - test_F_invs)*(ps_tile) 
    ).mean()
  CRPS = get_CRPS_base(test_means, test_stds, testy)
  return np.array([MSE, PCE, PBL, CRPS])