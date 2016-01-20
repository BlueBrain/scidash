import numpy as np
from scipy import stats
from sciunit.comparators import assert_dimensionless

def ttest(exp_mean, model_mean, exp_sd, model_sd, exp_n, model_n):
	m1 = exp_mean
	m2 = model_mean
	v1 = exp_sd**2
	v2 = model_sd**2
	n1 = exp_n
	n2 = model_n

	vn1 = v1 / n1
	vn2 = v2 / n2

	df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

	denom = np.sqrt(vn1 + vn2)
	d = m1 - m2
	t = np.divide(d, denom)

	prob = stats.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail

	return prob


def ttest_calc(observation, prediction):

	exp_means=[observation['mean_threshold'], observation['mean_prox_threshold'], observation['mean_dist_threshold'], observation['mean_peak_deriv'], observation['mean_nonlin_at_th'], observation['mean_nonlin_suprath'],  observation['mean_amp_at_th'], observation['mean_time_to_peak'], observation['mean_async_nonlin']]
	exp_SDs=[observation['threshold_std'], observation['prox_threshold_std'], observation['dist_threshold_std'], observation['peak_deriv_std'], observation['nonlin_at_th_std'], observation['nonlin_suprath_std'],  observation['amp_at_th_std'], observation['time_to_peak_std'], observation['async_nonlin_std']]
	exp_Ns=[observation['exp_n'], observation['prox_n'], observation['dist_n'], observation['exp_n'], observation['exp_n'], observation['exp_n'], observation['exp_n'], observation['exp_n'], observation['async_n']]

	model_means = [prediction['model_mean_threshold'], prediction['model_mean_prox_threshold'], prediction['model_mean_dist_threshold'], prediction['model_mean_peak_deriv'], prediction['model_mean_nonlin_at_th'], prediction['model_mean_nonlin_suprath'],  prediction['model_mean_amp_at_th'], prediction['model_mean_time_to_peak'], prediction['model_mean_async_nonlin']]
	model_SDs = [prediction['model_threshold_std'], prediction['model_prox_threshold_std'], prediction['model_dist_threshold_std'], prediction['model_peak_deriv_std'], prediction['model_nonlin_at_th_std'], prediction['model_nonlin_suprath_std'], prediction['model_amp_at_th_std'], prediction['model_time_to_peak_std'], prediction['model_async_nonlin_std']]
	model_N= prediction['model_n']

	p_values=[]

	for i in range (0, len(exp_means)):

		try:
			ttest_result = ttest(exp_means[i], model_means[i], exp_SDs[i], model_SDs[i], exp_Ns[i], model_N)
			ttest_result = assert_dimensionless(ttest_result)
			p_values.append(ttest_result)

		except (TypeError,AssertionError) as e:
			ttest_result = e

	return p_values


def zscore3(observation, prediction):
	"""Computes sum of z-scores from observation and prediction."""

	feature_error_means=np.array([])
	feature_error_stds=np.array([])
	features_names=(list(observation.keys()))
	feature_results_dict={}

	for i in range (0, len(features_names)):
		if "Apic" not in features_names[i]:

			p_value = prediction[features_names[i]]['feature values']
			o_mean = float(observation[features_names[i]]['Mean'])
			o_std = float(observation[features_names[i]]['Std'])

			p_std = prediction[features_names[i]]['feature sd']


			try:
				feature_error = abs(p_value - o_mean)/o_std
				feature_error = assert_dimensionless(feature_error)
				feature_error_mean = np.mean(feature_error)
				feature_error_sd = np.std(feature_error)

			except (TypeError, AssertionError) as e:
				feature_error = e
			feature_error_means = np.append(feature_error_means,feature_error_mean)
			feature_result = {features_names[i]:{'mean feature error':feature_error_mean,
											     'feature error sd':feature_error_sd}}

		feature_results_dict.update(feature_result)


	score_sum = np.sum(feature_error_means)

	return score_sum, feature_results_dict, features_names