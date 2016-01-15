# Python standard library
import inspect
import collections
import json
import pickle
import gzip
#import multiprocessing
#import functools
import math
import os
#import copyreg
#from types import MethodType
from functools import partial
from multiprocessing import Pool

# Very common packages
import numpy
from scipy import stats
import matplotlib.pyplot as plt
from quantities.quantity import Quantity
from quantities import mV, nA, ms, V, s

# Not so common packages
from neuron import h
import efel

# Sciunit
import sciunit
from sciunit import Test, Score, ObservationError
from sciunit.comparators import compute_zscore, assert_dimensionless
from sciunit.scores import ErrorScore, BooleanScore, ZScore
from sciunit.utils import printd

# Neurounit
from neuronunit.capabilities import ProducesMembranePotential
from neuronunit.capabilities import ReceivesCurrent

# Hippounit
from capabilities import Hoc, CurrentClamp, DendriticSynapse
from . import plot_utils as putils



class DictScore(Score):
	"""
	A dictionary of scores.
	"""
	def __init__(self, scores, related_data={}):
		if not isinstance(scores, dict):
			raise InvalidScoreError("score must be a dict")
		for key, val in scores.items():
			if not isinstance(val, Score):
				raise InvalidScoreError("Score with key " + str(key) + " is not a valid sciunit Score")

		super(DictScore, self).__init__(scores, related_data=related_data)

	def __str__(self):
		rstr = ""
		for key, val in self.score.items():
			rstr += str(key) + ":" + str(val) + ","
		if rstr.endswith(','):
			rstr = rstr.rstrip(',') # remove the trailing comma
		return rstr

	def getScore(self, key):
		return self.score[key]


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

	denom = numpy.sqrt(vn1 + vn2)
	d = m1 - m2
	t = numpy.divide(d, denom)

	prob = stats.t.sf(numpy.abs(t), df) * 2  # use np.abs to get upper tail

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


class P_Value(Score):
	"""
	A p value from t-test.
	"""

	def __init__(self, score, related_data={}):
		self.score_l = []
		for i in range(0, len(score)):
			if not isinstance(score[i], Exception) and not isinstance(score[i], float):
				raise InvalidScoreError("Score must be a float.")
			else:
				super(P_Value,self).__init__(score[i], related_data=related_data)
				self.score_l.append(score[i])

	def __str__(self):

		return '\n p_value_threshold = %.2f,\n p_value_prox_threshold  = %.2f,\n p_value_dist_threshold = %.2f,\n p_value_peak_dV/dt_at_threshold = %.2f,\n p_value_nonlin_at_th = %.2f,\n p_value_suprath_nonlin = %.2f,\n p_value_amplitude_at_th = %.2f,\n p_value_time_to_peak_at = %.2f,\n p_value_nonlin_at_th_asynch = %.2f\n' % (self.score_l[0], self.score_l[1], self.score_l[2], self.score_l[3],self.score_l[4], self.score_l[5],self.score_l[6], self.score_l[7],self.score_l[8])

def zscore3(observation, prediction):
	"""Computes sum of z-scores from observation and prediction."""

	feature_error_means=numpy.array([])
	feature_error_stds=numpy.array([])
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
				feature_error_mean=numpy.mean(feature_error)
				feature_error_sd=numpy.std(feature_error)

			except (TypeError,AssertionError) as e:
				feature_error = e
			feature_error_means=numpy.append(feature_error_means,feature_error_mean)
			feature_result={features_names[i]:{'mean feature error':feature_error_mean,
											'feature error sd':feature_error_sd}}

		feature_results_dict.update(feature_result)


	score_sum=numpy.sum(feature_error_means)

	return score_sum, feature_results_dict, features_names



def run_cclamp(path, force_run, model, amp):
	"""Runs a current clamp on the model and return membrane voltage trace.

	"""
	trace_file = path + 'cclamp_' + str(amp) + '.p'
	if force_run or (os.path.isfile(trace_file) is False):
		# load cell and run current clamp
		printd("running amplitude: ", amp)
		trace = {}
		traces = []
		model.initialise()
		model.set_cclamp(amp)
		t, v = model.run_cclamp()

		# Use efel to count the number of action potentials
		trace['T'] = t
		trace['V'] = v
		trace['stim_start'] = [500]
		trace['stim_end'] = [500+1000]
		traces.append(trace)
		traces_results = efel.getFeatureValues(traces, ['Spikecount'])
		traces.append(traces_results)

		# Store results in trace_file
		pickle.dump(traces, gzip.GzipFile(trace_file, "wb"))

	else:
		# load existing trace
		traces = pickle.load(gzip.GzipFile(trace_file, "rb"))

	return traces


class DepolarizationBlockTest(Test):
	"""
	Tests if the model enters depolarization block (maximum number of action potentials) under 
	current injection of increasing amplitudes.
	"""

	def __init__(self, observation = {'mean_Ith':None, 'Ith_std':None, 'mean_Veq': None, 'Veq_std': None},
				 name="Depolarization block test",
				 force_run=False,
				 show_plot=True,
				 data_directory='temp_data/',
				 fig_directory='figs/',
				 NProcesses=4):

		super(DepolarizationBlockTest, self).__init__(observation, name)
		
		self.required_capabilities += (Hoc, CurrentClamp)
		#self.required_capabilities += (ProducesMembranePotential, ReceivesCurrent)

		self.force_run = force_run
		self.show_plot = show_plot		
		self.data_directory = data_directory
		if not self.data_directory.endswith('/'):
			self.data_directory += '/'

		self.fig_directory = fig_directory
		if not self.fig_directory.endswith('/'):
			self.fig_directory += '/'

		self.NProcesses = NProcesses
	

	
	description = "Tests if the model enters depolarization block under current injection of increasing amplitudes."
	score_type = DictScore


	def validate_observation(self, observation):
		try:
			assert type(observation['mean_Ith']) is Quantity
			assert type(observation['Ith_std']) is Quantity
			assert type(observation['mean_Veq']) is Quantity
			assert type(observation['Veq_std']) is Quantity
		except Exception as e:
			raise ObservationError(("Observation must be of the form {'mean':float*mV,'std':float*mV}"))


	def generate_prediction(self, model):		
		# Run the model, in parallel, over a range of input currents
		pool = Pool(self.NProcesses)
		amps = numpy.arange(0, 1.65, 0.05) # 0 to 1.65 Amps, in steps of .05 A

		path = self.data_directory + model.name + '/cclamp/'
		if not os.path.exists(path):
			os.makedirs(path)

		cclamp_ = partial(run_cclamp, path, self.force_run, model)
		result = pool.map_async(cclamp_, amps)
		results = result.get()

		Ith, Veq = self.find_Ith_Veq(model, results, amps)
		prediction = {'model_Ith':float(Ith)*nA, 'model_Veq':float(Veq)*mV}

		return prediction


	def compute_score(self, observation, prediction):

		# Calculate z-Score for depolarizing current
		predI = {'mean':prediction['model_Ith']}
		obsI = {'mean':observation['mean_Ith'], 'std':observation['Ith_std']}
		zscoreI = compute_zscore(obsI, predI)

		# Calculate z-score for voltage at which depolarization occurs
		predV = {'mean':prediction['model_Veq']}
		obsV = {'mean':observation['mean_Veq'], 'std':observation['Veq_std']}
		zscoreV = compute_zscore(obsV, predV)

		return DictScore({'Ith':zscoreI, 'Veq':zscoreV})


	def find_Ith_Veq(self, model, results, amps):
		"""Determines the current at which the maximum number of action potentials are produced. 

		Returns
		=======
		Ith : nA 
		      The current which produces the peak number of APs. If maximum isn't achieved before
		      the largest input then NaN is returned.
		Veq : mV
			  The averaged membrane depolarization value when clamped at Ith nA. If Ith isn't reached
			  then returns NaN.
		"""

		spikecount_array = numpy.array([])
		for i, amp in enumerate(amps):
			spikecount_array = numpy.append(spikecount_array, results[i][1][0]['Spikecount'])

		max = numpy.amax(spikecount_array)
		Ith_index = numpy.where(spikecount_array==max)[0]
		if type(Ith_index) is numpy.ndarray: # If Ith index has multiple values...  
			Ith_index = Ith_index[0] # ...choose the first one.  

		# TODO: Using the absolute count of spikes could be problematic, as an extra spike could be the
		# difference between reaching Depolarization block or not. May be better to look at a relative 
		# number of APs with some margin for noise.
		if Ith_index == (spikecount_array.size-1):
			# If the max num AP is the last element, it didn`t enter depol. block     
			Ith = float('NaN')
			Veq = float('NaN')
			Veq_index = Ith_index

		else:
			Ith = amps[Ith_index]
			Veq_index = Ith_index+1

			Veq_trace = results[Veq_index][0]['V']
			time = numpy.array(results[Veq_index][0]['T'])
			indices1 = numpy.where(1400<=time)[0]           # we need the last 100ms of the current pulse
			indices2 = numpy.where(1500>=time)[0]
			trace_end_index_beginning = numpy.amin(indices1)
			trace_end_index_end = numpy.amax(indices2)
			trace_end = Veq_trace[trace_end_index_beginning:trace_end_index_end]      # THIS CONTAINS the last 100ms of the current pulse
			Veq = numpy.average(trace_end)

		# Save the plots 
		if self.show_plot:
			path_figs = self.fig_directory + '/' + model.name + '/depol_block/'
			if not os.path.exists(path_figs):
				os.makedirs(path_figs)

			printd("The figures are saved in the directory: ", path_figs)
			putils.plot_somatic_response_depol(path_figs, results, spikecount_array, Ith_index, Veq_index)
			putils.plot_num_aps(path_figs, amps, spikecount_array)
			putils.plot_somatic_response_Ith(path_figs, results, Ith_index)
			putils.plot_Ith(path_figs, self.observation, model, Ith)
			putils.plot_Veq(path_figs, self.observation, model, Veq)
			putils.plot_num_APs_at_Ith(path_figs, model, spikecount_array, Ith_index)


		# Print values and errors
		printd("Ith (the current intensity for which the model exhibited the maximum number of APs): ", Ith)
		printd("Veq (the equilibrium value during the depolarization block): ", Veq)
		if not math.isnan(Veq):
			Veq_error = abs(Veq*mV-self.observation['mean_Veq'])/self.observation['Veq_std']
			printd("The error of Veq in units of the experimental SD: ", Veq_error.base)

		if not math.isnan(Ith):
			Ith_error = abs(Ith*nA-self.observation['mean_Ith'])/self.observation['Ith_std']
			printd("The error of Ith in units of the experimental SD: ", Ith_error.base)

		return Ith, Veq



def analyse_syn_traces(t, v, v_dend, threshold):

	trace = {}
	trace['T'] = t
	trace['V'] = v
	trace['stim_start'] = [150]
	trace['stim_end'] = [150+500]  # should be changed
	traces = [trace]

	trace_dend = {}
	trace_dend['T'] = t
	trace_dend['V'] = v_dend
	trace_dend['stim_start'] = [150]
	trace_dend['stim_end'] = [150+500]
	traces_dend = [trace_dend]

	efel.setThreshold(threshold)
	traces_results_dend = efel.getFeatureValues(traces_dend,['Spikecount'])
	traces_results = efel.getFeatureValues(traces,['Spikecount'])
	spikecount_dend=traces_results_dend[0]['Spikecount']
	spikecount=traces_results[0]['Spikecount']

	result = [traces, traces_dend, spikecount, spikecount_dend]

	return result





def binsearch(model, path_bin_search, force_run, dend_loc0):
	"""Finds the synaptic weight (stimuluation???) threshold for which a spike on the dendritic synapse.

	Parameters
	==========
	dend_loc0 : [section, float]
				contains the id of the dendritic section and it's location along the branch (0 to 1)
	Returns
	=======
	found : True/False/None
			True indicates threshold found and it produced a spike at the soma. None if threshold
			was found, but didn't produce spike at the soma. False if no threshold found.
	c_stim : float
	         The weight/stim threshold value.
	"""

	interval = 0.1
	file_name = path_bin_search + 'weight_' +str(dend_loc0[0])+ '_' + str(dend_loc0[1]) + '.p'
	if force_run or (os.path.isfile(file_name) is False):
		c_minmax = model.c_minmax
		c_step_start = model.c_step_start
		c_step_stop = model.c_step_stop

		model.initialise()
		model.set_ampa_nmda(dend_loc0)
		model.set_netstim_netcon(interval)

		found = False
		while c_step_start >= c_step_stop and not found:
			c_stim = numpy.arange(c_minmax[0], c_minmax[1], c_step_start)

			first = 0
			last = numpy.size(c_stim, axis=0) - 1
			while first <= last and not found:
				midpoint = (first + last)//2
				result=[]
				for n in [4, 5]:
					# run with 4 and 5 presynaptic spikes, determine the number of spikes at dendrite
					model.set_num_weight(n, c_stim[midpoint])
					t, v, v_dend = model.run_syn()
					result.append(analyse_syn_traces(t, v, v_dend, model.threshold))

				if result[0][3]==0 and result[1][3]>=1:
					# Increase in n created a dendritic spike
					found = True
				else:
					# TODO: move this section into the while loop above, reduce the number of operations
					if result[0][3]>=1 and result[1][3]>=1:
						# Both values of n created a dendritic spike
						last = midpoint-1
					elif result[0][3]==0 and result[1][3]==0:
						# Neither value of n created a dendritic spike
						first = midpoint+1

			c_step_start=c_step_start/2

			if found:
				if result[1][2] >= 1:
					found = None
					break

		binsearch_result=[found, c_stim[midpoint]]
		pickle.dump(binsearch_result, gzip.GzipFile(file_name, "wb"))

	else:
		binsearch_result = pickle.load(gzip.GzipFile(file_name, "rb"))

	return binsearch_result



def run_synapse(model, path, force_run, dend_loc_num_weight, interval):
	"""Stimulate a dendritic synapse at record dendrite/soma polarization.

	Parameters
	==========
	interval : time
	           length of time of presynaptic stimuluation.
	dend_loc_num_weight : [section, location, num, weight]
						  dendrite section ID, location of synapse, num of presynaptic pulses, and the 
						  weight of the synapse.
	Returns
	=======
	traces : list(float)
	         voltage at soma.
	traces_dend : list(float)
				  voltages at synapse.
	spikecount : int
				 number of action potentials at soma.
	spikecount_dend : int
					  number of APs at synapse.
	"""
	ndend, xloc, num, weight = dend_loc_num_weight

	if interval > 0.1:
		file_name = path + 'synapse_' + str(num)+ '_' + str(ndend)+ '_' + str(xloc) + '.p'
	else:
		file_name = path + 'synapse_async_' + str(num)+ '_' + str(ndend)+ '_' + str(xloc) + '.p'

	if force_run or (os.path.isfile(file_name) is False):
		printd("- number of inputs:", num, "dendrite:", ndend, "xloc", xloc)

		model.initialise()
		model.set_ampa_nmda([ndend, xloc])
		model.set_netstim_netcon(interval)
		model.set_num_weight(num, weight)
		t, v, v_dend = model.run_syn()

		result = analyse_syn_traces(t, v, v_dend, model.threshold)
		pickle.dump(result, gzip.GzipFile(file_name, "wb"))

	else:
		result = pickle.load(gzip.GzipFile(file_name, "rb"))

	return result


class ObliqueIntegrationTest(Test):
	"""Tests the signal integration in oblique dendrites for increasing number of synchronous and asynchronous inputs"""

	def __init__(self,
				 observation = {'mean_threshold':None,'threshold_sem':None, 'threshold_std': None,
								'mean_prox_threshold':None,'prox_threshold_sem':None, 'prox_threshold_std': None,
								'mean_dist_threshold':None,'dist_threshold_sem':None, 'dist_threshold_std': None,
								'mean_nonlin_at_th':None,'nonlin_at_th_sem':None, 'nonlin_at_th_std': None,
								'mean_nonlin_suprath':None,'nonlin_suprath_sem':None, 'nonlin_suprath_std': None,
								'mean_peak_deriv':None,'peak_deriv_sem':None, 'peak_deriv_std': None,
								'mean_amp_at_th':None,'amp_at_th_sem':None, 'amp_at_th_std': None,
								'mean_time_to_peak':None,'time_to_peak_sem':None, 'time_to_peak_std': None,
								'mean_async_nonlin':None,'async_nonlin_sem':None, 'async_nonlin_std': None}  ,
				 name="Oblique integration test" ,
				 force_run_synapse=False,
				 force_run_bin_search=False,
				 show_plot=True,
				 data_directory='temp_data',
				 fig_directory='figs/',
				 NProcesses=4):


		Test.__init__(self, observation, name)

		self.required_capabilities += (Hoc, DendriticSynapse)
		#self.required_capabilities += (ProducesMembranePotential, ReceivesCurrent)

		self.show_plot = show_plot
		self.force_run_synapse = force_run_synapse
		self.force_run_bin_search = force_run_bin_search

		self.data_directory = data_directory
		if not self.data_directory.endswith('/'):
			self.data_directory += '/'

		self.fig_directory = fig_directory
		if not self.fig_directory.endswith('/'):
			self.fig_directory += '/'

		self.NProcesses = NProcesses

	description = "Tests the signal integration in oblique dendrites for increasing number of synchronous and asynchronous inputs"
	score_type = P_Value


	def calcs_plots(self, model, results, dend_loc000, dend_loc_num_weight, path_figs):

		experimental_mean_threshold=self.observation['mean_threshold']
		threshold_SEM=self.observation['threshold_sem']
		threshold_SD=self.observation['threshold_std']

		threshold_prox=self.observation['mean_prox_threshold']
		threshold_prox_SEM=self.observation['prox_threshold_sem']
		threshold_prox_SD=self.observation['prox_threshold_std']

		threshold_dist=self.observation['mean_dist_threshold']
		threshold_dist_SEM=self.observation['dist_threshold_sem']
		threshold_dist_SD=self.observation['dist_threshold_std']

		exp_mean_nonlin=self.observation['mean_nonlin_at_th']
		nonlin_SEM=self.observation['nonlin_at_th_sem']
		nonlin_SD=self.observation['nonlin_at_th_std']

		suprath_exp_mean_nonlin=self.observation['mean_nonlin_suprath']
		suprath_nonlin_SEM=self.observation['nonlin_suprath_sem']
		suprath_nonlin_SD=self.observation['nonlin_suprath_std']

		exp_mean_peak_deriv=self.observation['mean_peak_deriv']
		deriv_SEM=self.observation['peak_deriv_sem']
		deriv_SD=self.observation['peak_deriv_std']

		exp_mean_amp=self.observation['mean_amp_at_th']
		amp_SEM= self.observation['amp_at_th_sem']
		amp_SD=self.observation['amp_at_th_std']

		exp_mean_time_to_peak=self.observation['mean_time_to_peak']
		exp_mean_time_to_peak_SEM=self.observation['time_to_peak_sem']
		exp_mean_time_to_peak_SD=self.observation['time_to_peak_std']


		stop=len(dend_loc_num_weight)+1
		sep=numpy.arange(0,stop,11)
		sep_results=[]

		max_num_syn=10
		num = numpy.arange(0,max_num_syn+1)

		for i in range (0,len(dend_loc000)):
			sep_results.append(results[sep[i]:sep[i+1]])             # a list that contains the results of the 10 locations seperately (in lists)

		# sep_results[0]-- the first location
		# sep_results[0][5] -- the first location at 5 input
		# sep_results[0][1][0] -- the first location at 1 input, SOMA
		# sep_results[0][1][1] -- the first location at 1 input, dendrite
		# sep_results[0][1][1][0] -- just needed

		soma_depol=numpy.array([])
		soma_depols=[]
		sep_soma_depols=[]
		dV_dt=[]
		sep_dV_dt=[]
		soma_max_depols=numpy.array([])
		soma_expected=numpy.array([])
		sep_soma_max_depols=[]
		sep_soma_expected=[]
		max_dV_dt=numpy.array([])
		sep_max_dV_dt=[]
		max_dV_dt_index=numpy.array([],dtype=numpy.int64)
		sep_threshold=numpy.array([])
		prox_thresholds=numpy.array([])
		dist_thresholds=numpy.array([])
		peak_dV_dt_at_threshold=numpy.array([])
		nonlin=numpy.array([])
		suprath_nonlin=numpy.array([])
		amp_at_threshold=[]
		sep_time_to_peak=[]
		time_to_peak_at_threshold=numpy.array([])
		time_to_peak=numpy.array([])
		threshold_index=numpy.array([])

		for i in range (0, len(sep_results)):
			for j in range (0,max_num_syn+1):

		# calculating somatic depolarization and first derivative
				soma_depol=sep_results[i][j][0][0]['V'] - sep_results[i][0][0][0]['V']
				soma_depols.append(soma_depol)

				soma_max_depols=numpy.append(soma_max_depols,numpy.amax(soma_depol))

				dt=numpy.diff(sep_results[i][j][0][0]['T'] )
				dV=numpy.diff(sep_results[i][j][0][0]['V'] )
				deriv=dV/dt
				dV_dt.append(deriv)

				max_dV_dt=numpy.append(max_dV_dt, numpy.amax(dV_dt))

				diff_max_dV_dt=numpy.diff(max_dV_dt)

				if j==0:
					soma_expected=numpy.append(soma_expected,0)
				else:
					soma_expected=numpy.append(soma_expected,soma_max_depols[1]*j)

				if j!=0:
					peak=numpy.amax(soma_depol)

					peak_index=numpy.where(soma_depol==peak)[0]
					peak_time=sep_results[i][j][0][0]['T'][peak_index]
					t_to_peak=peak_time-150
					time_to_peak = numpy.append(time_to_peak, t_to_peak)
				else:
					time_to_peak = numpy.append(time_to_peak, 0)

				#print time_to_peak

			threshold_index0=numpy.where(diff_max_dV_dt==numpy.amax(diff_max_dV_dt[1:]))[0]
			threshold_index0=numpy.add(threshold_index0,1)

			if sep_results[i][threshold_index0][3] > 1 and sep_results[i][threshold_index0-1][3]==1:    #double spikes can cause bigger jump in dV?dt than the first single spike, to find the threshol, we want to eliminate this, but we also need the previous input level to generate spike
				threshold_index=numpy.where(diff_max_dV_dt==numpy.amax(diff_max_dV_dt[1:threshold_index0-1]))
				threshold_index=numpy.add(threshold_index,1)
			else:
				threshold_index=threshold_index0

			threshold=soma_expected[threshold_index]

			sep_soma_depols.append(soma_depols)
			sep_dV_dt.append(dV_dt)
			sep_soma_max_depols.append(soma_max_depols)
			sep_soma_expected.append(soma_expected)
			sep_max_dV_dt.append(max_dV_dt)
			sep_threshold=numpy.append(sep_threshold, threshold)
			peak_dV_dt_at_threshold=numpy.append(peak_dV_dt_at_threshold,max_dV_dt[threshold_index])
			nonlin=numpy.append(nonlin, soma_max_depols[threshold_index]/ soma_expected[threshold_index]*100)  #degree of nonlinearity
			suprath_nonlin=numpy.append(suprath_nonlin, soma_max_depols[threshold_index+1]/ soma_expected[threshold_index+1]*100)  #degree of nonlinearity
			amp_at_threshold=numpy.append(amp_at_threshold, soma_max_depols[threshold_index])
			sep_time_to_peak.append(time_to_peak)
			time_to_peak_at_threshold=numpy.append(time_to_peak_at_threshold, time_to_peak[threshold_index])

			soma_depols=[]
			dV_dt=[]
			soma_max_depols=numpy.array([])
			soma_expected=numpy.array([])
			max_dV_dt=numpy.array([])
			threshold_index=numpy.array([])
			threshold_index0=numpy.array([])
			time_to_peak=numpy.array([])


		prox_thresholds=sep_threshold[0::2]
		dist_thresholds=sep_threshold[1::2]

		threshold_errors = numpy.array([abs(experimental_mean_threshold - threshold_errors*mV)/threshold_SD  for threshold_errors in sep_threshold])     # does the same calculation on every element of a list  #x = [1,3,4,5,6,7,8] t = [ t**2 for t in x ]
		prox_threshold_errors=numpy.array([abs(threshold_prox - prox_threshold_errors*mV)/threshold_prox_SD  for prox_threshold_errors in prox_thresholds])        # and I could easily make it a numpy array : t = numpy.array([ t**2 for t in x ])
		dist_threshold_errors=numpy.array([abs(threshold_dist - dist_threshold_errors*mV)/threshold_dist_SD  for dist_threshold_errors in dist_thresholds])
		peak_deriv_errors=numpy.array([abs(exp_mean_peak_deriv - peak_deriv_errors*mV /ms )/deriv_SD  for peak_deriv_errors in peak_dV_dt_at_threshold])
		nonlin_errors=numpy.array([abs(exp_mean_nonlin- nonlin_errors)/nonlin_SD  for nonlin_errors in nonlin])
		suprath_nonlin_errors=numpy.array([abs(suprath_exp_mean_nonlin- suprath_nonlin_errors)/suprath_nonlin_SD  for suprath_nonlin_errors in suprath_nonlin])
		amplitude_errors=numpy.array([abs(exp_mean_amp- amplitude_errors*mV)/amp_SD  for amplitude_errors in amp_at_threshold])
		time_to_peak_errors=numpy.array([abs(exp_mean_time_to_peak- time_to_peak_errors*ms)/exp_mean_time_to_peak_SD  for time_to_peak_errors in time_to_peak_at_threshold])


		# means and SDs
		mean_threshold_errors=numpy.mean(threshold_errors)
		mean_prox_threshold_errors=numpy.mean(prox_threshold_errors)
		mean_dist_threshold_errors=numpy.mean(dist_threshold_errors)
		mean_peak_deriv_errors=numpy.mean(peak_deriv_errors)
		mean_nonlin_errors=numpy.mean(nonlin_errors)
		suprath_mean_nonlin_errors=numpy.mean(suprath_nonlin_errors)
		mean_amplitude_errors=numpy.mean(amplitude_errors)
		mean_time_to_peak_errors=numpy.mean(time_to_peak_errors)

		sd_threshold_errors=numpy.std(threshold_errors)
		sd_prox_threshold_errors=numpy.std(prox_threshold_errors)
		sd_dist_threshold_errors=numpy.std(dist_threshold_errors)
		sd_peak_deriv_errors=numpy.std(peak_deriv_errors)
		sd_nonlin_errors=numpy.std(nonlin_errors)
		suprath_sd_nonlin_errors=numpy.std(suprath_nonlin_errors)
		sd_amplitude_errors=numpy.std(amplitude_errors)
		sd_time_to_peak_errors=numpy.std(time_to_peak_errors)


		mean_sep_threshold=float(numpy.mean(sep_threshold)) *mV
		mean_prox_thresholds=float(numpy.mean(prox_thresholds)) *mV
		mean_dist_thresholds=numpy.mean(dist_thresholds) *mV
		mean_peak_dV_dt_at_threshold=numpy.mean(peak_dV_dt_at_threshold) *V /s
		mean_nonlin=numpy.mean(nonlin)
		suprath_mean_nonlin=numpy.mean(suprath_nonlin)
		mean_amp_at_threshold=numpy.mean(amp_at_threshold) *mV
		mean_time_to_peak_at_threshold=numpy.mean(time_to_peak_at_threshold) *ms

		sd_sep_threshold=float(numpy.std(sep_threshold)) *mV
		sd_prox_thresholds=float(numpy.std(prox_thresholds)) *mV
		sd_dist_thresholds=numpy.std(dist_thresholds) *mV
		sd_peak_dV_dt_at_threshold=numpy.std(peak_dV_dt_at_threshold) *V /s
		sd_nonlin=numpy.std(nonlin)
		suprath_sd_nonlin=numpy.std(suprath_nonlin)
		sd_amp_at_threshold=numpy.std(amp_at_threshold) *mV
		sd_time_to_peak_at_threshold=numpy.std(time_to_peak_at_threshold) *ms


		depol_input=numpy.array([])
		mean_depol_input=[]
		SD_depol_input=[]
		SEM_depol_input=[]

		expected_depol_input=numpy.array([])
		expected_mean_depol_input=[]

		prox_depol_input=numpy.array([])
		prox_mean_depol_input=[]
		prox_SD_depol_input=[]
		prox_SEM_depol_input=[]

		prox_expected_depol_input=numpy.array([])
		prox_expected_mean_depol_input=[]

		dist_depol_input=numpy.array([])
		dist_mean_depol_input=[]
		dist_SD_depol_input=[]
		dist_SEM_depol_input=[]

		dist_expected_depol_input=numpy.array([])
		dist_expected_mean_depol_input=[]

		peak_deriv_input=numpy.array([])
		mean_peak_deriv_input=[]
		SD_peak_deriv_input=[]
		SEM_peak_deriv_input=[]
		n=len(sep_soma_max_depols)

		prox_sep_soma_max_depols=sep_soma_max_depols[0::2]
		dist_sep_soma_max_depols=sep_soma_max_depols[1::2]
		prox_n=len(prox_sep_soma_max_depols)
		dist_n=len(dist_sep_soma_max_depols)

		prox_sep_soma_expected=sep_soma_expected[0::2]
		dist_sep_soma_expected=sep_soma_expected[1::2]


		for j in range (0, max_num_syn+1):
			for i in range (0, len(sep_soma_max_depols)):
				depol_input=numpy.append(depol_input,sep_soma_max_depols[i][j])
				expected_depol_input=numpy.append(expected_depol_input,sep_soma_expected[i][j])
				peak_deriv_input=numpy.append(peak_deriv_input,sep_max_dV_dt[i][j])
			mean_depol_input.append(numpy.mean(depol_input))
			expected_mean_depol_input.append(numpy.mean(expected_depol_input))
			mean_peak_deriv_input.append(numpy.mean(peak_deriv_input))
			SD_depol_input.append(numpy.std(depol_input))
			SEM_depol_input.append(numpy.std(depol_input)/math.sqrt(n))
			SD_peak_deriv_input.append(numpy.std(peak_deriv_input))
			SEM_peak_deriv_input.append(numpy.std(peak_deriv_input)/math.sqrt(n))
			depol_input=numpy.array([])
			expected_depol_input=numpy.array([])
			peak_deriv_input=numpy.array([])

		for j in range (0, max_num_syn+1):
			for i in range (0, len(prox_sep_soma_max_depols)):
				prox_depol_input=numpy.append(prox_depol_input,prox_sep_soma_max_depols[i][j])
				prox_expected_depol_input=numpy.append(prox_expected_depol_input,prox_sep_soma_expected[i][j])
			prox_mean_depol_input.append(numpy.mean(prox_depol_input))
			prox_expected_mean_depol_input.append(numpy.mean(prox_expected_depol_input))
			prox_SD_depol_input.append(numpy.std(prox_depol_input))
			prox_SEM_depol_input.append(numpy.std(prox_depol_input)/math.sqrt(prox_n))
			prox_depol_input=numpy.array([])
			prox_expected_depol_input=numpy.array([])

		for j in range (0, max_num_syn+1):
			for i in range (0, len(dist_sep_soma_max_depols)):
				dist_depol_input=numpy.append(dist_depol_input,dist_sep_soma_max_depols[i][j])
				dist_expected_depol_input=numpy.append(dist_expected_depol_input,dist_sep_soma_expected[i][j])
			dist_mean_depol_input.append(numpy.mean(dist_depol_input))
			dist_expected_mean_depol_input.append(numpy.mean(dist_expected_depol_input))
			dist_SD_depol_input.append(numpy.std(dist_depol_input))
			dist_SEM_depol_input.append(numpy.std(dist_depol_input)/math.sqrt(dist_n))
			dist_depol_input=numpy.array([])
			dist_expected_depol_input=numpy.array([])


		if self.show_plot:
			putils.plot_somatic_traces_sync(path_figs, dend_loc000, num, sep_results)
			putils.plot_input_output_curves_sync(path_figs, sep_results, sep_soma_expected, sep_soma_max_depols, dend_loc000)
			putils.plot_summary_input_output_curve_sync(path_figs, expected_mean_depol_input, mean_depol_input, SD_depol_input, SEM_depol_input, prox_SD_depol_input, prox_SEM_depol_input, dist_SD_depol_input, dist_SEM_depol_input, prox_expected_mean_depol_input, dist_expected_mean_depol_input, prox_mean_depol_input, dist_mean_depol_input)
			putils.plot_peak_derivative_plots_sync(path_figs, sep_results, num, sep_max_dV_dt, dend_loc000, mean_peak_deriv_input, SD_peak_deriv_input, SEM_peak_deriv_input)
			putils.plot_values_sync(path_figs, threshold_SD, experimental_mean_threshold, sep_results, sep_threshold, dend_loc000, threshold_prox_SD, threshold_prox, prox_thresholds, threshold_dist, threshold_dist_SD, dist_thresholds, deriv_SD, exp_mean_peak_deriv, peak_dV_dt_at_threshold, nonlin_SD, exp_mean_nonlin, nonlin, suprath_nonlin_SD, suprath_exp_mean_nonlin, suprath_nonlin, amp_SD, exp_mean_amp, amp_at_threshold, exp_mean_time_to_peak_SD, exp_mean_time_to_peak, time_to_peak_at_threshold)
			putils.plot_errors_sync(path_figs, sep_results, dend_loc000, threshold_errors, dist_threshold_errors, prox_threshold_errors, peak_deriv_errors, nonlin_errors, suprath_nonlin_errors, amplitude_errors, time_to_peak_errors)
			putils.plot_mean_values_sync(path_figs, threshold_SD, sd_sep_threshold, threshold_prox_SD, sd_prox_thresholds, threshold_dist_SD, sd_dist_thresholds, amp_SD, sd_amp_at_threshold, experimental_mean_threshold, mean_sep_threshold, threshold_prox, mean_prox_thresholds, threshold_dist, mean_dist_thresholds, exp_mean_amp, mean_amp_at_threshold, deriv_SD, sd_peak_dV_dt_at_threshold, exp_mean_peak_deriv, mean_peak_dV_dt_at_threshold, nonlin_SD, sd_nonlin, suprath_nonlin_SD, suprath_sd_nonlin, exp_mean_time_to_peak_SD, sd_time_to_peak_at_threshold, exp_mean_time_to_peak, mean_time_to_peak_at_threshold, exp_mean_nonlin, mean_nonlin, suprath_exp_mean_nonlin, suprath_mean_nonlin)
			putils.plot_mean_errors_sync(path_figs, sd_threshold_errors, sd_prox_threshold_errors, sd_dist_threshold_errors, sd_peak_deriv_errors,	sd_nonlin_errors, suprath_sd_nonlin_errors, sd_amplitude_errors, sd_time_to_peak_errors, mean_threshold_errors,  mean_prox_threshold_errors, mean_dist_threshold_errors, mean_peak_deriv_errors, mean_nonlin_errors, suprath_mean_nonlin_errors, mean_amplitude_errors, mean_time_to_peak_errors)

		exp_n=92
		n_prox=33
		n_dist=44

		exp_means=[experimental_mean_threshold, threshold_prox, threshold_dist, exp_mean_peak_deriv, exp_mean_nonlin, suprath_exp_mean_nonlin,  exp_mean_amp, exp_mean_time_to_peak]
		exp_SDs=[threshold_SD, threshold_prox_SD, threshold_dist_SD, deriv_SD, nonlin_SD, suprath_nonlin_SD,  amp_SD, exp_mean_time_to_peak_SD ]
		exp_Ns=[exp_n, n_prox, n_dist, exp_n, exp_n, exp_n, exp_n, exp_n]

		model_means = [mean_sep_threshold, mean_prox_thresholds, mean_dist_thresholds, mean_peak_dV_dt_at_threshold , mean_nonlin, suprath_mean_nonlin,  mean_amp_at_threshold, mean_time_to_peak_at_threshold]
		model_SDs = [sd_sep_threshold, sd_prox_thresholds, sd_dist_thresholds, sd_peak_dV_dt_at_threshold , sd_nonlin, suprath_sd_nonlin, sd_amp_at_threshold, sd_time_to_peak_at_threshold]
		model_N= len(sep_results)


		return model_means, model_SDs, model_N


	def calcs_plots_async(self, model, results, dend_loc000, dend_loc_num_weight, path_figs):

		async_nonlin=self.observation['mean_async_nonlin']
		async_nonlin_SEM=self.observation['async_nonlin_sem']
		async_nonlin_SD=self.observation['async_nonlin_std']

		stop=len(dend_loc_num_weight)+1
		sep=numpy.arange(0,stop,11)
		sep_results=[]

		max_num_syn=10
		num = numpy.arange(0,max_num_syn+1)

		for i in range (0,len(dend_loc000)):
			sep_results.append(results[sep[i]:sep[i+1]])             # a list that contains the results of the 10 locations seperately (in lists)

		# sep_results[0]-- the first location
		# sep_results[0][5] -- the first location at 5 input
		# sep_results[0][1][0] -- the first location at 1 input, SOMA
		# sep_results[0][1][1] -- the first location at 1 input, dendrite
		# sep_results[0][1][1][0] -- just needed

		soma_depol=numpy.array([])
		soma_depols=[]
		sep_soma_depols=[]
		dV_dt=[]
		sep_dV_dt=[]
		soma_max_depols=numpy.array([])
		soma_expected=numpy.array([])
		sep_soma_max_depols=[]
		sep_soma_expected=[]
		max_dV_dt=numpy.array([])
		sep_max_dV_dt=[]
		nonlin=numpy.array([])
		sep_nonlin=[]

		nonlins_at_th=numpy.array([])


		for i in range (0, len(sep_results)):
			for j in range (0,max_num_syn+1):
				# calculating somatic depolarization and first derivative
				soma_depol=sep_results[i][j][0][0]['V'] - sep_results[i][0][0][0]['V']
				soma_depols.append(soma_depol)

				soma_max_depols=numpy.append(soma_max_depols,numpy.amax(soma_depol))

				dt=numpy.diff(sep_results[i][j][0][0]['T'] )
				dV=numpy.diff(sep_results[i][j][0][0]['V'] )
				deriv=dV/dt
				dV_dt.append(deriv)

				max_dV_dt=numpy.append(max_dV_dt, numpy.amax(dV_dt))

				if j==0:
					soma_expected=numpy.append(soma_expected,0)
					nonlin=numpy.append(nonlin,100)
				else:
					soma_expected=numpy.append(soma_expected,soma_max_depols[1]*j)
					nonlin=numpy.append(nonlin, soma_max_depols[j]/ soma_expected[j]*100)

			sep_soma_depols.append(soma_depols)
			sep_dV_dt.append(dV_dt)
			sep_soma_max_depols.append(soma_max_depols)
			sep_soma_expected.append(soma_expected)
			sep_max_dV_dt.append(max_dV_dt)
			sep_nonlin.append(nonlin)
			nonlins_at_th=numpy.append(nonlins_at_th, nonlin[5])      #degree of nonlin at 4 inputs, that is the threshold in the synchronous case

			soma_depols=[]
			dV_dt=[]
			soma_max_depols=numpy.array([])
			soma_expected=numpy.array([])
			max_dV_dt=numpy.array([])
			nonlin=numpy.array([])

		expected_depol_input=numpy.array([])
		expected_mean_depol_input=[]
		depol_input=numpy.array([])
		mean_depol_input=[]
		SD_depol_input=[]
		SEM_depol_input=[]

		peak_deriv_input=numpy.array([])
		mean_peak_deriv_input=[]
		SD_peak_deriv_input=[]
		SEM_peak_deriv_input=[]
		n=len(sep_soma_max_depols)


		for j in range (0, max_num_syn+1):
			for i in range (0, len(sep_soma_max_depols)):
				depol_input=numpy.append(depol_input,sep_soma_max_depols[i][j])
				expected_depol_input=numpy.append(expected_depol_input,sep_soma_expected[i][j])
				peak_deriv_input=numpy.append(peak_deriv_input,sep_max_dV_dt[i][j])
			mean_depol_input.append(numpy.mean(depol_input))
			expected_mean_depol_input.append(numpy.mean(expected_depol_input))
			mean_peak_deriv_input.append(numpy.mean(peak_deriv_input))
			SD_depol_input.append(numpy.std(depol_input))
			SEM_depol_input.append(numpy.std(depol_input)/math.sqrt(n))
			SD_peak_deriv_input.append(numpy.std(peak_deriv_input))
			SEM_peak_deriv_input.append(numpy.std(peak_deriv_input)/math.sqrt(n))
			depol_input=numpy.array([])
			expected_depol_input=numpy.array([])
			peak_deriv_input=numpy.array([])

		mean_nonlin_at_th=numpy.mean(nonlins_at_th)
		SD_nonlin_at_th=numpy.std(nonlins_at_th)
		SEM_nonlin_at_th=SD_nonlin_at_th/math.sqrt(n)

		if self.show_plot:
			putils.plot_traces_async(path_figs, dend_loc000, sep_results, num)
			putils.plot_somatic_traces_async(path_figs, num, dend_loc000, sep_results)
			putils.plot_input_output_curves_async(path_figs, sep_results, sep_soma_expected, sep_soma_max_depols, dend_loc000, expected_mean_depol_input, mean_depol_input, SD_depol_input, SEM_depol_input)
			putils.peak_derivative_plots_async(path_figs, sep_results, dend_loc000, num, mean_peak_deriv_input, SD_peak_deriv_input, SEM_peak_deriv_input, sep_max_dV_dt)
			putils.plot_nonlin_values_async(path_figs, dend_loc000, async_nonlin_SD, async_nonlin, sep_nonlin)
			putils.plot_nonlin_errors_async(path_figs, sep_nonlin, async_nonlin, async_nonlin_SD, n, dend_loc000)

		return mean_nonlin_at_th, SD_nonlin_at_th



	def validate_observation(self, observation):
		try:
			assert type(observation['mean_threshold']) is Quantity
			assert type(observation['threshold_std']) is Quantity
			assert type(observation['mean_prox_threshold']) is Quantity
			assert type(observation['prox_threshold_std']) is Quantity
			assert type(observation['mean_dist_threshold']) is Quantity
			assert type(observation['dist_threshold_std']) is Quantity
			assert type(observation['mean_peak_deriv']) is Quantity
			assert type(observation['peak_deriv_std']) is Quantity
			assert type(observation['mean_amp_at_th']) is Quantity
			assert type(observation['amp_at_th_std']) is Quantity
			assert type(observation['mean_time_to_peak']) is Quantity
			assert type(observation['time_to_peak_std']) is Quantity

		except Exception as e:
			raise ObservationError(("Observation must be of the form {'mean':float*mV,'std':float*mV}"))


	def generate_prediction(self, model):
		
		# Bin search - find dendrites which 		
		traces = []
		path_bin_search = self.data_directory + model.name + '/bin_search/'
		if not os.path.exists(path_bin_search):
			os.makedirs(path_bin_search)

		pool = Pool(self.NProcesses)
		binsearch_ = partial(binsearch, model, path_bin_search, self.force_run_bin_search)
		result = pool.map_async(binsearch_, model.dend_loc)
		results = result.get()
		
		# List of dendrite sections that didn't produce spike at soma (at any point)
		dend0 = []		
		for i in range(0, len(model.dend_loc)):
			if results[i][0] == None or results[i][0] == False :
				if model.dend_loc[i][0] not in dend0:
					dend0.append(model.dend_loc[i][0])
			if results[i][0]==None :
				printd('the dendritic spike on at least one of the locations of dendrite ',  model.dend_loc[i][0], 'generated somatic AP')
			if results[i][0]==False :
				printd('No dendritic spike could be generated on at least one of the locations of dendrite',  model.dend_loc[i][0])

		# All points along any dendrite capable of causing somatic spike
		dend_loc00 = []
		for k in range(0, len(dend0)):
			for i in range(0, len(model.dend_loc)):
				if model.dend_loc[i][0] == dend0[k]:
					dend_loc00.append(model.dend_loc[i])

		#dend_loc000 does not contain the dendrites in which spike causes somatic AP
		dend_loc000=list(model.dend_loc)
		for i in range(0, len(dend_loc00)):
			dend_loc000.remove(dend_loc00[i])       

		dend_loc_num_weight = []
		max_num_syn = 10
		num = numpy.arange(0, max_num_syn+1)
		for i in range(0, len(dend_loc000)):
			for j in num:
				e=list(dend_loc000[i])
				e.append(num[j])
				e.append(results[i][1])
				dend_loc_num_weight.append(e)

		
		# run_synapse - record synaptic/somatic depolarization for a range of presynaptic stimuli		
		path = self.data_directory + model.name + '/synapse/'
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)

		interval_sync = 0.1
		pool = Pool(self.NProcesses)
		run_synapse_ = partial(run_synapse, model, path, self.force_run_synapse, interval=interval_sync)
		result = pool.map_async(run_synapse_, dend_loc_num_weight)
		results = result.get()


		interval_async = 2
		pool1 = Pool(self.NProcesses)
		run_synapse_ = partial(run_synapse, model, path, self.force_run_synapse, interval=interval_async)
		result_async = pool1.map_async(run_synapse_,dend_loc_num_weight)
		results_async = result_async.get()


		path_figs = self.fig_directory + 'oblique/' + model.name + '/'
		if self.show_plot:			
			if not os.path.exists(path_figs):
				os.makedirs(path_figs)
			printd("The figures are saved in the directory: ", path_figs)

		model_means, model_SDs, model_N = self.calcs_plots(model, results, dend_loc000, dend_loc_num_weight, path_figs)
		mean_nonlin_at_th_asynch, SD_nonlin_at_th_asynch = self.calcs_plots_async(model, results_async, dend_loc000, dend_loc_num_weight, path_figs)

		prediction = {'model_mean_threshold':model_means[0], 'model_threshold_std': model_SDs[0],
						'model_mean_prox_threshold':model_means[1], 'model_prox_threshold_std': model_SDs[1],
						'model_mean_dist_threshold':model_means[2], 'model_dist_threshold_std': model_SDs[2],
						'model_mean_peak_deriv':model_means[3],'model_peak_deriv_std': model_SDs[3],
						'model_mean_nonlin_at_th':model_means[4], 'model_nonlin_at_th_std': model_SDs[4],
						'model_mean_nonlin_suprath':model_means[5], 'model_nonlin_suprath_std': model_SDs[5],
						'model_mean_amp_at_th':model_means[6],'model_amp_at_th_std': model_SDs[6],
						'model_mean_time_to_peak':model_means[7], 'model_time_to_peak_std': model_SDs[7],
						'model_mean_async_nonlin':mean_nonlin_at_th_asynch, 'model_async_nonlin_std': SD_nonlin_at_th_asynch,
						'model_n': model_N }
		return prediction

	def compute_score(self, observation, prediction):
		score0 = ttest_calc(observation,prediction)

		return P_Value(score0)





def run_stim(model, path, force_run, stimuli_list):

	stimulus_name, amplitude, delay, duration, stim_section_name, stim_location_x, stim_type, rec_section_name, rec_location_x = stimuli_list

	traces_result={}

	if stim_type == "SquarePulse":
		file_name = path + stimulus_name + '.p'

		if force_run or (os.path.isfile(file_name) is False):

			printd("running stimulus: " + stimulus_name)

			model.initialise()
			stim_section_name = model.translate(stim_section_name)
			rec_section_name = model.translate(rec_section_name)
			model.set_cclamp(float(amplitude), float(delay), float(duration), stim_section_name, stim_location_x)
			#model.set_cclamp_somatic_feature(float(amplitude), float(delay), float(duration), stim_section_name, stim_location_x)
			#t, v = model.run_cclamp_somatic_feature(rec_section_name, rec_location_x)
			t, v = model.run_cclamp(rec_section_name, rec_location_x)

			traces_result[stimulus_name]=[t,v]

			pickle.dump(traces_result, gzip.GzipFile(file_name, "wb"))

		else:
			traces_result = pickle.load(gzip.GzipFile(file_name, "rb"))

	else:
		traces_result=None

	return traces_result

def analyse_traces(stimuli_list, traces_results, features_list):

	feature_name, target_sd, target_mean, stimulus, feature_type = features_list
	target_sd=float(target_sd)
	target_mean=float(target_mean)

	feature_result={}
	trace = {}
	for i in range (0, len(traces_results)):
		for key, value in traces_results[i].items():
			stim_name=key
		if stim_name == stimulus:

			trace['T'] = traces_results[i][key][0]
			trace['V'] = traces_results[i][key][1]

	for i in range (0, len(stimuli_list)):
		if stimuli_list[i][0]==stimulus:

			trace['stim_start'] = [float(stimuli_list[i][2])]
			trace['stim_end'] = [float(stimuli_list[i][2])+float(stimuli_list[i][3])]

	traces = [trace]
	#print traces

	efel_results = efel.getFeatureValues(traces,[feature_type])

	feature_values=efel_results[0][feature_type]

	if feature_values is None:
		feature_values = []
	feature_mean=numpy.mean(feature_values)
	feature_sd=numpy.std(feature_values)

	feature_result={feature_name:{'traces':traces,
								  'feature values': feature_values,
								  'feature mean': feature_mean,
								  'feature sd': feature_sd}}
	return feature_result



class SomaticFeaturesTest(Test):
	"""Tests some somatic features under current injection of increasing amplitudes."""

	def __init__(self,
				 observation = {}  ,
				 name="Somatic features test",
				 stimuli={},
				 force_run=False,
				 show_plot=True,
				 data_directory='temp_data/',
				 fig_directory='figs/',
				 NProcesses=4):

		Test.__init__(self, observation, name)

		self.required_capabilities += (Hoc, CurrentClamp)
		#self.required_capabilities += (ProducesMembranePotential, ReceivesCurrent,)

		self.stimuli = stimuli

		self.force_run = force_run
		self.show_plot = show_plot		
		self.data_directory = data_directory
		if not self.data_directory.endswith('/'):
			self.data_directory += '/'

		self.fig_directory = fig_directory
		if not self.fig_directory.endswith('/'):
			self.fig_directory += '/'

		self.NProcesses = NProcesses

	description = "Tests some somatic features under current injection of increasing amplitudes."
	score_type = ZScore

	def create_stimuli_list(self):
		stimulus_list=[]
		stimuli_list=[]
		stimuli_names=list(self.stimuli.keys())

		for i in range (0, len(stimuli_names)):
			if "Apic" not in stimuli_names[i]:
				stimulus_list.append(stimuli_names[i])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['Amplitude'])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['Delay'])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['Duration'])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['StimSectionName'])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['StimLocationX'])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['Type'])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['RecSectionName'])
				stimulus_list.append(self.stimuli[stimuli_names[i]]['RecLocationX'])
				stimuli_list.append(stimulus_list)
				stimulus_list=[]

		return stimuli_list

	def create_features_list(self, observation):

		feature_list=[]
		features_list=[]
		features_names=(list(observation.keys()))


		for i in range (0, len(features_names)):
			if "Apic" not in features_names[i]:
				feature_list.append(features_names[i])
				feature_list.append(observation[features_names[i]]['Std'])
				feature_list.append(observation[features_names[i]]['Mean'])
				feature_list.append(observation[features_names[i]]['Stimulus'])
				feature_list.append(observation[features_names[i]]['Type'])
				features_list.append(feature_list)
				feature_list=[]

		return features_names, features_list


	def generate_prediction(self, model):

		path = self.data_directory + model.name + '/soma/'
		if not os.path.exists(path):
			os.makedirs(path)

		pool = Pool(self.NProcesses)

		stimuli_list=self.create_stimuli_list()
		run_stim_ = partial(run_stim, model, path, self.force_run)
		traces_result = pool.map_async(run_stim_, stimuli_list)
		traces_results = traces_result.get()

		features_names, features_list = self.create_features_list(self.observation)

		analyse_traces_ = partial(analyse_traces, stimuli_list, traces_results)
		feature_result = pool.map_async(analyse_traces_, features_list)
		feature_results = feature_result.get()

		feature_results_dict={}
		for i in range (0,len(feature_results)):
			feature_results_dict.update(feature_results[i])  #concatenate dictionaries

		if self.show_plot:
			path_figs = self.fig_directory + 'soma/' + model.name + '/'
			if not os.path.exists(path_figs):
				os.makedirs(path_figs, exist_ok=True)
			printd("The figures are saved in the directory: ", path_figs)

			putils.plot_traces(path_figs, traces_results)
			putils.plot_traces_subplots(path_figs, traces_results)
			putils.plot_absolute_features(path_figs, features_names, feature_results_dict, observation)


		#self.create_figs(model, traces_results, features_names, feature_results_dict, self.observation)

		prediction = feature_results_dict

		return prediction

	def compute_score(self, observation, prediction):
		score_sum, feature_results_dict, features_names  = zscore3(observation, prediction)
		return ZScore(score_sum)


