import matplotlib.pyplot as plt
import numpy as np
import math

from neuronunit import plottools




######## DepolarizationBlockTest ###########

def plot_somatic_response_depol(path_figs, results, spikecount_array, Ith_index, Veq_index):
	
	plt.close()
	#plt.figure()
	if Ith_index == spikecount_array.size-1:
		plt.plot(results[spikecount_array.size-1][0]['T'], results[spikecount_array.size-1][0]['V'])
		plt.title("somatic response to the highest current intensity\n (The model did not enter depol. block.)")
	else:
		plt.plot(results[Veq_index][0]['T'],results[Veq_index][0]['V'])
		plt.title("somatic response at Ith + 0.05 nA")
		plt.xlabel("time (ms)")
		plt.ylabel("Somatic voltage (mV)")

	plt.savefig(path_figs + 'somatic_resp_at_depol_block' + '.pdf', dpi=300)
	

def plot_num_aps(path_figs, amps, spikecount_array):
	plt.close()	
	#plt.figure()
	plt.plot(amps,spikecount_array,'.-')
	plt.xlabel("I (nA)")
	plt.ylabel("number of APs")
	plt.savefig(path_figs + 'number_of_APs' + '.pdf', dpi=600)

def plot_somatic_response_Ith(path_figs, results, Ith_index):
	plt.close()
	#plt.figure()
	plt.plot(results[Ith_index][0]['T'],results[Ith_index][0]['V'])
	plt.title("somatic response at Ith")
	plt.xlabel("time (ms)")
	plt.ylabel("Somatic voltage (mV)")
	plt.savefig(path_figs + 'somatic_resp_at_Ith' + '.pdf', dpi=600)

def plot_Ith(path_figs, observation, model, Ith):
	plt.figure()
	x = np.array([1, 2])
	Ith_array = np.array([observation['mean_Ith'], Ith])
	labels = ['Target Ith with SD', model.name]

	x2 = np.array([1])
	y2 = np.array([observation['mean_Ith']])
	e = np.array([observation['Ith_std']])

	plt.plot(x, Ith_array, 'o')
	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=10)
	plt.margins(0.2)
	plt.ylabel("Ith (nA)")
	plt.savefig(path_figs + 'Ith' + '.pdf', dpi=600)

def plot_Veq(path_figs, observation, model, Veq):
	plt.figure()
	x = np.array([1, 2])
	Veq_array = np.array([observation['mean_Veq'], Veq])
	labels = ['Target Veq with SD', model.name]
	e = np.array([observation['Veq_std'], 0.0])

	x2 = np.array([1])
	y2 = np.array([observation['mean_Veq']])
	e = np.array([observation['Veq_std']])

	plt.figure(5)
	plt.plot(x, Veq_array, 'o')
	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=10)
	plt.margins(0.2)
	plt.ylabel("Veq (mV)")
	plt.savefig(path_figs + 'Veq' + '.pdf', dpi=600)

def plot_num_APs_at_Ith(path_figs, model, spikecount_array, Ith_index):
	plt.figure()
	num_AP_min = 13
	num_AP_max = 82
	labels2 = [model.name]

	plt.figure(7)
	plt.axhline(y=num_AP_min, label='Min. num.of AP\n observed\n experimentally')
	plt.axhline(y=num_AP_max, label='Max. num.of AP\n observed\n experimentally')
	plt.legend()

	plt.plot([1], spikecount_array[Ith_index], 'o')
	plt.title("For the models that doesn't enter depol block,\n the plot shows the num of AP-s for the highest current intensity")
	plt.xticks([1], labels2)
	plt.margins(0.2)
	plt.ylabel("Number of AP at Ith")
	plt.savefig(path_figs + 'num_of_APs_at_Ith' + '.pdf', dpi=600)


###### ObliqueIntegrationTest #########
def plot_somatic_traces_sync(path_figs, dend_loc000, num, sep_results):
	plt.close()
	#plt.figure()
	fig0, axes0 = plt.subplots(nrows=2, ncols=1)
	fig0.tight_layout()
	fig0.suptitle('Synchronous inputs (red: dendritic trace, black: somatic trace)')
	for i in range (0,len(dend_loc000)):
		plt.subplot(5,2,i+1)
		plt.subplots_adjust(hspace = 0.5)
		for j, number in enumerate(num):
			plt.plot(sep_results[i][j][0][0]['T'],sep_results[i][j][0][0]['V'], 'k')       # somatic traces
			plt.plot(sep_results[i][j][1][0]['T'],sep_results[i][j][1][0]['V'], 'r')        # dendritic traces
		plt.title('Input in dendrite '+str(dend_loc000[i][0])+ ' at location: ' +str(dend_loc000[i][1]))

		plt.xlabel("time (ms)")
		plt.ylabel("Voltage (mV)")
		plt.xlim(140, 250)

	fig0 = plt.gcf()
	fig0.set_size_inches(12, 18)
	plt.savefig(path_figs + 'traces_sync' + '.pdf', dpi=600,)

	fig0, axes0 = plt.subplots(nrows=2, ncols=1)
	fig0.tight_layout()
	fig0.suptitle('Synchronous inputs')
	for i in range (0,len(dend_loc000)):
		plt.subplot(5,2,i+1)
		plt.subplots_adjust(hspace = 0.5)
		for j, number in enumerate(num):
			plt.plot(sep_results[i][j][0][0]['T'],sep_results[i][j][0][0]['V'], 'k')       # somatic traces
		plt.title('Input in dendrite '+str(dend_loc000[i][0])+ ' at location: ' +str(dend_loc000[i][1]))

		plt.xlabel("time (ms)")
		plt.ylabel("Somatic voltage (mV)")
		plt.xlim(140, 250)
	fig0 = plt.gcf()
	fig0.set_size_inches(12, 18)
	plt.savefig(path_figs + 'somatic_traces_sync' + '.pdf', dpi=600,)

def plot_input_output_curves_sync(path_figs, sep_results, sep_soma_expected, sep_soma_max_depols, dend_loc000):
	#plt.figure()
	plt.close()
	plt.title('Synchronous inputs')

	# Expected EPSP - Measured EPSP plot
	colormap = plt.cm.spectral      #http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
	plt.gca().set_color_cycle([colormap(j) for j in np.linspace(0, 0.9, len(sep_results))])
	for i in range (0, len(sep_results)):

		plt.plot(sep_soma_expected[i],sep_soma_max_depols[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
		plt.plot(sep_soma_expected[i],sep_soma_expected[i], 'k--')         # this gives the linear line
		plt.xlabel("expected EPSP (mV)")
		plt.ylabel("measured EPSP (mV)")
		plt.legend(loc=2, prop={'size':10})
	fig = plt.gcf()
	fig.set_size_inches(12, 12)
	plt.savefig(path_figs + 'input_output_curves_sync' + '.pdf', dpi=600,)

def plot_summary_input_output_curve_sync(path_figs, expected_mean_depol_input, mean_depol_input, SD_depol_input, SEM_depol_input, prox_SD_depol_input, prox_SEM_depol_input, dist_SD_depol_input, dist_SEM_depol_input, prox_expected_mean_depol_input, dist_expected_mean_depol_input, prox_mean_depol_input, dist_mean_depol_input):
	#plt.figure()
	plt.close()
	plt.suptitle('Synchronous inputs')

	plt.subplot(3,1,1)
	plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	plt.plot(expected_mean_depol_input,expected_mean_depol_input, 'k--')         # this gives the linear line
	plt.margins(0.1)
	plt.legend(loc=2)
	plt.title("Summary plot of mean input-output curve for every locations")
	plt.xlabel("expected EPSP (mV)")
	plt.ylabel("measured EPSP (mV)")

	plt.subplot(3,1,2)
	plt.errorbar(prox_expected_mean_depol_input, prox_mean_depol_input, yerr=prox_SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	plt.errorbar(prox_expected_mean_depol_input, prox_mean_depol_input, yerr=prox_SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	plt.plot(prox_expected_mean_depol_input,prox_expected_mean_depol_input, 'k--')         # this gives the linear line
	plt.margins(0.1)
	plt.legend(loc=2)
	plt.title("Summary plot of mean input-output curve for proximal locations")
	plt.xlabel("expected EPSP (mV)")
	plt.ylabel("measured EPSP (mV)")

	plt.subplot(3,1,3)
	plt.errorbar(dist_expected_mean_depol_input, dist_mean_depol_input, yerr=dist_SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	plt.errorbar(dist_expected_mean_depol_input, dist_mean_depol_input, yerr=dist_SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	plt.plot(dist_expected_mean_depol_input,dist_expected_mean_depol_input, 'k--')         # this gives the linear line

	plt.margins(0.1)
	plt.legend(loc=2)
	plt.title("Summary plot of mean input-output curve for distal locations")
	plt.xlabel("expected EPSP (mV)")
	plt.ylabel("measured EPSP (mV)")

	fig = plt.gcf()
	fig.set_size_inches(12, 15)
	plt.savefig(path_figs + 'summary_input_output_curve_sync' + '.pdf', dpi=600,)


def plot_peak_derivative_plots_sync(path_figs, sep_results, num, sep_max_dV_dt, dend_loc000, mean_peak_deriv_input, SD_peak_deriv_input, SEM_peak_deriv_input):
		plt.close()
		#plt.figure(5)
		plt.subplot(2,1,1)
		plt.title('Synchronous inputs')
		#Derivative plot
		colormap = plt.cm.spectral
		plt.gca().set_color_cycle([colormap(j) for j in np.linspace(0, 0.9, len(sep_results))])
		for i in range (0, len(sep_results)):

			plt.plot(num,sep_max_dV_dt[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
			plt.xlabel("# of inputs")
			plt.ylabel("dV/dt (V/s)")
			plt.legend(loc=2, prop={'size':10})

		plt.subplot(2,1,2)

		plt.errorbar(num, mean_peak_deriv_input, yerr=SD_peak_deriv_input, linestyle='-', marker='o', color='red', label='SD')
		plt.errorbar(num, mean_peak_deriv_input, yerr=SEM_peak_deriv_input, linestyle='-', marker='o', color='blue', label='SEM')
		plt.margins(0.1)
		plt.legend(loc=2)
		plt.title("Summary plot of mean peak dV/dt amplitude")
		plt.xlabel("# of inputs")
		plt.ylabel("dV/dt (V/s)")

		fig = plt.gcf()
		fig.set_size_inches(12, 12)
		plt.savefig(path_figs + 'peak_derivative_plots_sync' + '.pdf', dpi=600,)


def plot_values_sync(path_figs, threshold_SD, experimental_mean_threshold, sep_results, sep_threshold, 
	dend_loc000, threshold_prox_SD, threshold_prox, prox_thresholds, threshold_dist, threshold_dist_SD, 
	dist_thresholds, deriv_SD, exp_mean_peak_deriv, peak_dV_dt_at_threshold, 
	nonlin_SD, exp_mean_nonlin, nonlin,
	suprath_nonlin_SD, suprath_exp_mean_nonlin, suprath_nonlin,
	amp_SD, exp_mean_amp, amp_at_threshold,
	exp_mean_time_to_peak_SD, exp_mean_time_to_peak, time_to_peak_at_threshold):

	plt.close()
	#plt.figure()

	#VALUES PLOT
	fig, axes = plt.subplots(nrows=4, ncols=2)
	fig.tight_layout()
	fig.suptitle('Synchronous inputs')

	plt.subplot(4, 2, 1)
	# plot of thresholds
	x =np.array([])
	labels = ['exp mean with SD']
	e = np.array([threshold_SD])
	x2 =np.array([1])
	y2 = np.array([experimental_mean_threshold])
	for i in range (0, len(sep_results)+1):
		x=np.append(x, i+1)
	for i in range (0, len(sep_results)):
		labels.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
		plt.plot(x[i+1], sep_threshold[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Threshold (mV)")

	plt.subplot(4, 2, 2)

	# plot of proximal thresholds
	x_prox =np.array([])
	labels_prox = ['exp mean with SD']
	e = np.array([threshold_prox_SD])
	x2 =np.array([1])
	y2 = np.array([threshold_prox])
	for i in range (0, len(dend_loc000),2):
		labels_prox.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	for i in range (0, len(prox_thresholds)+1):
		x_prox=np.append(x_prox, i+1)
	for i in range (0, len(prox_thresholds)):
		plt.plot(x_prox[i+1], prox_thresholds[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x_prox, labels_prox, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Proximal threshold (mV)")

	plt.subplot(4, 2, 3)

	# plot of distal thresholds
	x_dist =np.array([])
	labels_dist = ['exp mean with SD']
	e = np.array([threshold_dist_SD])
	x2 =np.array([1])
	y2 = np.array([threshold_dist])
	for i in range (1, len(dend_loc000),2):
		labels_dist.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	for i in range (0, len(dist_thresholds)+1):
		x_dist=np.append(x_dist, i+1)
	for i in range (0, len(dist_thresholds)):
		plt.plot(x_dist[i+1], dist_thresholds[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x_dist, labels_dist, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Distal threshold (mV)")

	plt.subplot(4, 2, 4)

	# plot of peak derivateives at threshold
	e = np.array([deriv_SD])
	x2 =np.array([1])
	y2 = np.array([exp_mean_peak_deriv])
	for i in range (0, len(sep_results)):
		plt.plot(x[i+1], peak_dV_dt_at_threshold[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("peak derivative at threshold (V/s)")

	plt.subplot(4, 2, 5)

	# plot of degree of nonlinearity at threshold

	e = np.array([nonlin_SD])
	x2 =np.array([1])
	y2 = np.array([exp_mean_nonlin])
	for i in range (0, len(sep_results)):
		plt.plot(x[i+1], nonlin[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("degree of nonlinearity (%)")

	plt.subplot(4, 2, 6)

	# plot of suprathreshold degree of nonlinearity

	e = np.array([suprath_nonlin_SD])
	x2 =np.array([1])
	y2 = np.array([suprath_exp_mean_nonlin])
	for i in range (0, len(sep_results)):
		plt.plot(x[i+1], suprath_nonlin[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("suprath. degree of nonlinearity (%)")

	plt.subplot(4, 2, 7)

	# plot of amplitude at threshold
	e = np.array([amp_SD])
	x2 =np.array([1])
	y2 = np.array([exp_mean_amp])
	for i in range (0, len(sep_results)):
		plt.plot(x[i+1], amp_at_threshold[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Amplitude at threshold (mV)")


	plt.subplot(4, 2, 8)

	# plot of time to peak at threshold
	e = np.array([exp_mean_time_to_peak_SD])
	x2 =np.array([1])
	y2 = np.array([exp_mean_time_to_peak])
	for i in range (0, len(sep_results)):
		plt.plot(x[i+1], time_to_peak_at_threshold[i], 'o')

	plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	plt.xticks(x, labels, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("time to peak at threshold (ms)")

	fig = plt.gcf()
	fig.set_size_inches(14, 14)
	plt.savefig(path_figs + 'values_sync' + '.pdf', dpi=600,)


def plot_errors_sync(path_figs, 
	sep_results, dend_loc000, threshold_errors, dist_threshold_errors,
	prox_threshold_errors,
	peak_deriv_errors,
	nonlin_errors,
	suprath_nonlin_errors,
	amplitude_errors,
	time_to_peak_errors):

	plt.close()
	#plt.figure()
	fig2, axes2 = plt.subplots(nrows=3, ncols=2)
	fig2.tight_layout()
	fig2.suptitle(' Errors in units of the experimental SD of the feature (synchronous inputs)')
	plt.subplot(4, 2, 1)

	#threshold error plot
	x = np.array([])
	for i in range (0, len(sep_results)+1):
		x=np.append(x, i+1)

	x_error =np.array([])
	labels_error = []
	for i in range (0, len(sep_results)):
		labels_error.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
		x_error=np.append(x_error, i+1)

		plt.plot(x_error[i], threshold_errors[i], 'o')
	plt.xticks(x_error, labels_error, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Threshold error")

	plt.subplot(4, 2, 2)

	#proximal threshold error plot

	x_prox_err =np.array([])
	labels_prox_err = []
	for i in range (0, len(dend_loc000),2):
		labels_prox_err.append('dend'+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	for i in range (0, len(prox_threshold_errors)):
		x_prox_err=np.append(x, i+1)

		plt.plot(x_prox_err[i], prox_threshold_errors[i], 'o')
	plt.xticks(x_prox_err, labels_prox_err, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Proximal threshold error")

	plt.subplot(4, 2, 3)

	# distal threshold error plot

	x_dist_err =np.array([])
	labels_dist_err = []
	for i in range (1, len(dend_loc000),2):
		labels_dist_err.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	for i in range (0, len(dist_threshold_errors)):
		x_dist_err=np.append(x_dist_err, i+1)

		plt.plot(x_dist_err[i], dist_threshold_errors[i], 'o')
	plt.xticks(x_dist_err, labels_dist_err, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Distal threshold error")

	plt.subplot(4, 2, 4)

	#peak deriv error plot

	for i in range (0, len(sep_results)):

		plt.plot(x_error[i], peak_deriv_errors[i], 'o')
	plt.xticks(x_error, labels_error, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Peak derivative error")

	plt.subplot(4, 2, 5)

	#  degree of nonlin. error plot

	for i in range (0, len(sep_results)):
		plt.plot(x_error[i], nonlin_errors[i], 'o')
	plt.xticks(x_error, labels_error, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Degree of nonlinearity error")


	plt.subplot(4, 2, 6)

	# suprathreshold degree of nonlin. error plot

	for i in range (0, len(sep_results)):
		plt.plot(x_error[i], suprath_nonlin_errors[i], 'o')
	plt.xticks(x_error, labels_error, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Suprath. degree of nonlinearity error")

	plt.subplot(4, 2, 7)

	# amplitude error plot

	for i in range (0, len(sep_results)):
		plt.plot(x_error[i], amplitude_errors[i], 'o')
	plt.xticks(x_error, labels_error, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Amplitude error")

	plt.subplot(4, 2, 8)

	# time to peak error plot

	for i in range (0, len(sep_results)):
		plt.plot(x_error[i], time_to_peak_errors[i], 'o')
	plt.xticks(x_error, labels_error, rotation=20)
	plt.tick_params(labelsize=10)
	plt.margins(0.1)
	plt.ylabel("Time to peak error")

	fig = plt.gcf()
	fig.set_size_inches(14, 18)
	plt.savefig(path_figs + 'errors_sync' + '.pdf', dpi=600,)

def plot_mean_values_sync(path_figs,
	threshold_SD, sd_sep_threshold, threshold_prox_SD,
	sd_prox_thresholds, threshold_dist_SD, sd_dist_thresholds, amp_SD, 
	sd_amp_at_threshold,
	experimental_mean_threshold, mean_sep_threshold, 
	threshold_prox, mean_prox_thresholds, threshold_dist, 
	mean_dist_thresholds, exp_mean_amp, mean_amp_at_threshold,
	deriv_SD, sd_peak_dV_dt_at_threshold,
	exp_mean_peak_deriv, mean_peak_dV_dt_at_threshold,
	nonlin_SD, sd_nonlin, suprath_nonlin_SD, suprath_sd_nonlin,
	exp_mean_time_to_peak_SD, sd_time_to_peak_at_threshold,
	exp_mean_time_to_peak, mean_time_to_peak_at_threshold,
	exp_mean_nonlin, mean_nonlin, suprath_exp_mean_nonlin, suprath_mean_nonlin):

	plt.close()
	#plt.figure()
	# mean values plot
	fig3, axes3 = plt.subplots(nrows=2, ncols=1)
	fig3.tight_layout()
	fig3.suptitle('Synchronous inputs', fontsize=15)
	plt.subplot(2,2,1)

	e_values = np.array([threshold_SD, sd_sep_threshold, threshold_prox_SD, sd_prox_thresholds, threshold_dist_SD, sd_dist_thresholds, amp_SD, sd_amp_at_threshold])
	x_values =np.array([1,2,4,5,7,8,10,11])
	y_values = np.array([experimental_mean_threshold, mean_sep_threshold, threshold_prox, mean_prox_thresholds, threshold_dist, mean_dist_thresholds, exp_mean_amp, mean_amp_at_threshold])
	labels_values=['experimental mean threshold', 'model mean threshold', 'exp. mean proximal threshold', 'model mean proximal threshold', 'exp. mean distal threshold', 'model mean distal threshold','exp. mean amplitude at th.', 'model mean amplitude at th.']
	plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	plt.xticks(x_values, labels_values, rotation=20)
	plt.tick_params(labelsize=15)
	plt.margins(0.1)
	plt.ylabel("experimental and model mean values with SD (mV)", fontsize=15)

	plt.subplot(2,2,2)

	e_values = np.array([deriv_SD, sd_peak_dV_dt_at_threshold])
	x_values =np.array([1,2])
	y_values = np.array([exp_mean_peak_deriv, mean_peak_dV_dt_at_threshold])
	labels_values=['exp. mean peak dV/dt at th.', 'model mean peak dV/dt at th.']
	plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	plt.xticks(x_values, labels_values, rotation=20)
	plt.tick_params(labelsize=15)
	plt.margins(0.1)
	plt.ylabel("experimental and model mean max dV/dt with SD (V/s)", fontsize=15)

	plt.subplot(2,2,3)

	e_values = np.array([nonlin_SD, sd_nonlin, suprath_nonlin_SD, suprath_sd_nonlin])
	x_values = np.array([1,2,4,5])
	y_values = np.array([exp_mean_nonlin, mean_nonlin, suprath_exp_mean_nonlin, suprath_mean_nonlin])
	labels_values=['exp mean degree of nonlinearity at th.', 'model mean degree of nonlinearity at th.', 'exp mean suprath. degree of nonlinearity', 'model mean suprath. degree of nonlinearity']
	plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	plt.xticks(x_values, labels_values, rotation=10)
	plt.tick_params(labelsize=15)
	plt.margins(0.1)
	plt.ylabel("experimental and model mean degree of nonlinearity with SD (%)", fontsize=15)

	plt.subplot(2,2,4)

	e_values = np.array([exp_mean_time_to_peak_SD, sd_time_to_peak_at_threshold])
	x_values =np.array([1,2])
	y_values = np.array([exp_mean_time_to_peak, mean_time_to_peak_at_threshold])
	labels_values=['exp. mean time to peak at th.', 'model mean time to peak at th.']
	plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	plt.xticks(x_values, labels_values, rotation=10)
	plt.tick_params(labelsize=15)
	plt.margins(0.1)
	plt.ylabel("experimental and model mean time to peak with SD (ms)", fontsize=15)

	fig = plt.gcf()
	fig.set_size_inches(22, 18)
	plt.savefig(path_figs + 'mean_values_sync' + '.pdf', dpi=600,)

def plot_mean_errors_sync(path_figs, sd_threshold_errors, sd_prox_threshold_errors, 
	sd_dist_threshold_errors, sd_peak_deriv_errors,	sd_nonlin_errors, suprath_sd_nonlin_errors, 
	sd_amplitude_errors, sd_time_to_peak_errors,
	mean_threshold_errors,  mean_prox_threshold_errors, mean_dist_threshold_errors, 
	mean_peak_deriv_errors, mean_nonlin_errors, suprath_mean_nonlin_errors, mean_amplitude_errors, 
	mean_time_to_peak_errors):

	plt.close()
	#plt.figure()
	# mean errors plot
	plt.title('Synchronous inputs', fontsize=15)
	e_errors = np.array([sd_threshold_errors, sd_prox_threshold_errors, sd_dist_threshold_errors, sd_peak_deriv_errors, sd_nonlin_errors, suprath_sd_nonlin_errors, sd_amplitude_errors, sd_time_to_peak_errors])
	x_errors =np.array([1,2,3,4,5,6,7,8])
	y_errors = np.array([mean_threshold_errors,  mean_prox_threshold_errors, mean_dist_threshold_errors, mean_peak_deriv_errors, mean_nonlin_errors, suprath_mean_nonlin_errors, mean_amplitude_errors, mean_time_to_peak_errors ])
	labels_errors=['mean threshold error', 'mean proximal threshold error', 'mean distal threshold error', 'mean peak dV/dt at th. error', 'mean degree of nonlinearity at th.error', 'mean suprath. degree of nonlinearity error','mean amplitude at th. error', 'mean time to peak at th. error']
	plt.errorbar(x_errors, y_errors, e_errors, linestyle='None', marker='o')
	plt.xticks(x_errors, labels_errors, rotation=20)
	plt.tick_params(labelsize=15)
	plt.margins(0.1)
	plt.ylabel("model mean errors in unit of the experimental SD (with SD)", fontsize=15)

	fig = plt.gcf()
	fig.set_size_inches(16, 18)
	plt.savefig(path_figs + 'mean_errors_sync' + '.pdf', dpi=600,)



###### ObliqueIntegrationTest #########

def plot_traces_async(path_figs, dend_loc000, sep_results, num):

	plt.close()
	#plt.figure()
	fig0, axes0 = plt.subplots(nrows=2, ncols=1)
	fig0.tight_layout()
	fig0.suptitle('Asynchronous inputs (red: dendritic trace, black: somatic trace)')
	for i in range (0,len(dend_loc000)):
		plt.subplot(5,2,i+1)
		plt.subplots_adjust(hspace = 0.5)
		for j, number in enumerate(num):
			plt.plot(sep_results[i][j][0][0]['T'],sep_results[i][j][0][0]['V'], 'k')       # somatic traces
			plt.plot(sep_results[i][j][1][0]['T'],sep_results[i][j][1][0]['V'], 'r')        # dendritic traces
		plt.title('Input in dendrite '+str(dend_loc000[i][0])+ ' at location: ' +str(dend_loc000[i][1]))

		plt.xlabel("time (ms)")
		plt.ylabel("Voltage (mV)")
		plt.xlim(140, 250)
	fig = plt.gcf()
	fig.set_size_inches(12, 18)
	plt.savefig(path_figs + 'traces_async' + '.pdf', dpi=600,)

def plot_somatic_traces_async(path_figs, num, dend_loc000, sep_results):

	plt.close()
	#plt.figure()
	fig0, axes0 = plt.subplots(nrows=2, ncols=1)
	fig0.tight_layout()
	fig0.suptitle('Asynchronous inputs')
	for i in range (0,len(dend_loc000)):
		plt.subplot(5,2,i+1)
		plt.subplots_adjust(hspace = 0.5)
		for j, number in enumerate(num):
			plt.plot(sep_results[i][j][0][0]['T'],sep_results[i][j][0][0]['V'], 'k')       # somatic traces
		plt.title('Input in dendrite '+str(dend_loc000[i][0])+ ' at location: ' +str(dend_loc000[i][1]))

		plt.xlabel("time (ms)")
		plt.ylabel("Somatic voltage (mV)")
		plt.xlim(140, 250)
	fig = plt.gcf()
	fig.set_size_inches(12, 18)
	plt.savefig(path_figs + 'somatic_traces_async' + '.pdf', dpi=600,)


def plot_input_output_curves_async(path_figs, sep_results, sep_soma_expected, sep_soma_max_depols, dend_loc000,
	expected_mean_depol_input, mean_depol_input, SD_depol_input, SEM_depol_input):
		
	plt.close()
	#plt.figure()
	plt.suptitle('Asynchronous inputs')
	plt.subplot(2,1,1)
	# Expected EPSP - Measured EPSP plot
	colormap = plt.cm.spectral      #http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
	plt.gca().set_color_cycle([colormap(j) for j in np.linspace(0, 0.9, len(sep_results))])
	for i in range (0, len(sep_results)):

		plt.plot(sep_soma_expected[i],sep_soma_max_depols[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
		plt.plot(sep_soma_expected[i],sep_soma_expected[i], 'k--')         # this gives the linear line
	plt.xlabel("expected EPSP (mV)")
	plt.ylabel("measured EPSP (mV)")
	plt.legend(loc=2, prop={'size':10})

	plt.subplot(2,1,2)

	plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	plt.plot(expected_mean_depol_input,expected_mean_depol_input, 'k--')         # this gives the linear line
	plt.margins(0.1)
	plt.legend(loc=2)
	plt.title("Summary plot of mean input-output curve")
	plt.xlabel("expected EPSP (mV)")
	plt.ylabel("measured EPSP (mV)")

	fig = plt.gcf()
	fig.set_size_inches(12, 12)
	plt.savefig(path_figs + 'input_output_curves_async' + '.pdf', dpi=600,)

def peak_derivative_plots_async(path_figs, sep_results, dend_loc000, num,
	mean_peak_deriv_input, SD_peak_deriv_input, SEM_peak_deriv_input,
	sep_max_dV_dt):
	
	plt.close()
	#plt.figure()
	plt.subplot(2,1,1)
	plt.title('Asynchronous inputs')
	#Derivative plot
	colormap = plt.cm.spectral
	plt.gca().set_color_cycle([colormap(j) for j in np.linspace(0, 0.9, len(sep_results))])
	for i in range (0, len(sep_results)):

		plt.plot(num,sep_max_dV_dt[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
		plt.xlabel("# of inputs")
		plt.ylabel("dV/dt (V/s)")
		plt.legend(loc=2, prop={'size':10})

	plt.subplot(2,1,2)

	plt.errorbar(num, mean_peak_deriv_input, yerr=SD_peak_deriv_input, linestyle='-', marker='o', color='red', label='SD')
	plt.errorbar(num, mean_peak_deriv_input, yerr=SEM_peak_deriv_input, linestyle='-', marker='o', color='blue', label='SEM')
	plt.margins(0.01)
	plt.legend(loc=2)
	plt.title("Summary plot of mean peak dV/dt amplitude")
	plt.xlabel("# of inputs")
	plt.ylabel("dV/dt (V/s)")

	fig = plt.gcf()
	fig.set_size_inches(12, 12)
	plt.savefig(path_figs + 'peak_derivative_plots_async' + '.pdf', dpi=600,)


def plot_nonlin_values_async(path_figs, dend_loc000, async_nonlin_SD, async_nonlin, sep_nonlin):
	fig0, axes0 = plt.subplots(nrows=5, ncols=2)
	fig0.tight_layout()
	fig0.suptitle('Asynchronous inputs', fontsize=15)
	for j in range (0,len(dend_loc000)):
		plt.subplot(5,2,j+1)
		x =np.array([])
		labels = ['exp. mean\n with SD']
		e = np.array([async_nonlin_SD])
		x2 =np.array([1])
		y2 = np.array([async_nonlin])
		for i in range (0, len(sep_nonlin[0])+1):
			x=np.append(x, i+1)
			labels.append(str(i)+ ' inputs')
		for i in range (0, len(sep_nonlin[j])):
			plt.plot(x[i+1], sep_nonlin[j][i], 'o')

		plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
		plt.xticks(x, labels, rotation=40)
		plt.tick_params(labelsize=15)
		plt.margins(0.1)
		plt.ylabel("Degree of nonlinearity (%)", fontsize=15)
		plt.title('dendrite '+str(dend_loc000[j][0])+ ' location: ' +str(dend_loc000[j][1]))

	fig = plt.gcf()
	fig.set_size_inches(20, 20)
	plt.savefig(path_figs + 'nonlin_values_async' + '.pdf', dpi=600,)

def plot_nonlin_errors_async(path_figs, sep_nonlin, async_nonlin, async_nonlin_SD, n, dend_loc000):
	async_nonlin_errors=[]
	asynch_nonlin_error_at_th=np.array([])

	for i in range (0, len(sep_nonlin)):
		async_nonlin_err = np.array([abs(async_nonlin- async_nonlin_err)/async_nonlin_SD  for async_nonlin_err in sep_nonlin[i]])     # does the same calculation on every element of a list  #x = [1,3,4,5,6,7,8] t = [ t**2 for t in x ]
		async_nonlin_errors.append(async_nonlin_err)
		asynch_nonlin_error_at_th=np.append(asynch_nonlin_error_at_th, async_nonlin_err[4])

	mean_nonlin_error_at_th=np.mean(asynch_nonlin_error_at_th)
	SD_nonlin_error_at_th=np.std(asynch_nonlin_error_at_th)
	SEM_nonlin_error_at_th=SD_nonlin_error_at_th/math.sqrt(n)


	fig0, axes0 = plt.subplots(nrows=5, ncols=2)
	fig0.tight_layout()
	fig0.suptitle('Asynchronous inputs', fontsize=15)
	for j in range (0,len(dend_loc000)):
		plt.subplot(5,2,j+1)
		x =np.array([])
		labels = ['exp. mean\n with SD']

		for i in range (0, len(sep_nonlin[0])+1):
			x=np.append(x, i+1)
			labels.append(str(i)+ ' inputs')

		for i in range (0, len(async_nonlin_errors[j])):
			plt.plot(x[i], async_nonlin_errors[j][i], 'o')

		plt.xticks(x, labels[1:-1], rotation=20)
		plt.tick_params(labelsize=15)
		plt.margins(0.1)
		plt.ylabel("Degree of nonlin. error (%)", fontsize=15)
		plt.title('dendrite '+str(dend_loc000[j][0])+ ' location: ' +str(dend_loc000[j][1]))
	fig = plt.gcf()
	fig.set_size_inches(18, 20)
	plt.savefig(path_figs + 'nonlin_errors_async' + '.pdf', dpi=600,)

def plot_p_values(path_figs, score):
	plt.close()
	x = numpy.arange(1, 10)
	labels=['threshold', 'proximal threshold', 'distal threshold', 'peak dV/dt at th.','degree of nonlinearity at th.', 'suprath. degree of nonlinearity', 'amplitude at th.', 'time to peak at th.', 'asynch. degree of nonlin. at th.']
	plt.plot(x, score, linestyle='None', marker='o')
	plt.xticks(x, labels, rotation=20)
	plt.tick_params(labelsize=11)
	plt.axhline(y=0.05, label='0.05', color='red')
	plt.legend()
	plt.margins(0.1)
	plt.ylabel("p values")
	fig = plt.gcf()
	fig.set_size_inches(12, 10)
	plt.savefig(path_figs + 'p_values' + '.pdf', dpi=600,)


######### SomaticFeaturesTest ############
def plot_traces(path_figs, traces_results):
	plt.close()
	for i in range (0, len(traces_results)):
		for key, value in traces_results[i].items():
			plt.plot(traces_results[i][key][0], traces_results[i][key][1], label=key)
	plt.legend(loc=2)
	plt.savefig(path_figs + 'traces' + '.pdf', dpi=600,)

def plot_traces_subplots(path_figs, traces_results):
	plt.close()
	fig, axes = plt.subplots(nrows=4, ncols=2)
	fig.tight_layout()
	for i in range (0, len(traces_results)):

		for key, value in traces_results[i].items():

			plt.subplot(4,2,i+1)
			plt.plot(traces_results[i][key][0], traces_results[i][key][1])
			plt.title(key)
			plt.xlabel("ms")
			plt.ylabel("mV")
	fig = plt.gcf()
	fig.set_size_inches(12, 10)
	plt.savefig(path_figs + 'traces_subplots' + '.pdf', dpi=600,)

def plot_absolute_features(path_figs, features_names, feature_results_dict, observation):
	axs = plottools.tiled_figure("absolute features", figs={}, frames=1, columns=1, orientation='page',
							height_ratios=[0.9,0.1], top=0.97, bottom=0.05, left=0.25, right=0.97, hspace=0.1, wspace=0.2)

	for i in range (len(features_names)):
		if "Apic" not in features_names[i]:
			feature_name=features_names[i]
			y=i
			axs[0].errorbar(feature_results_dict[feature_name]['feature mean'], y, xerr=feature_results_dict[feature_name]['feature sd'], marker='o', color='blue', clip_on=False)
			axs[0].errorbar(float(observation[feature_name]['Mean']), y, xerr=float(observation[feature_name]['Std']), marker='o', color='red', clip_on=False)
	axs[0].yaxis.set_ticks(list(range(len(features_names))))
	axs[0].set_yticklabels(features_names)
	axs[0].set_ylim(-1, len(features_names))
	axs[0].set_title('Absolute Features')
	plt.savefig(path_figs + 'absolute_features' + '.pdf', dpi=600,)

def plot_feature_errors(path_figs, features_names):

	if not os.path.exists(path_figs):
		os.makedirs(path_figs, exist_ok=True)

	axs2 = plottools.tiled_figure("features", figs={}, frames=1, columns=1, orientation='page',
								  height_ratios=[0.9,0.1], top=0.97, bottom=0.05, left=0.25, right=0.97, hspace=0.1, wspace=0.2)

	for i in range (len(features_names)):
		if "Apic" not in features_names[i]:
			feature_name=features_names[i]
			y=i
			axs2[0].errorbar(feature_results_dict[feature_name]['mean feature error'], y, xerr=feature_results_dict[feature_name]['feature error sd'], marker='o', color='blue', clip_on=False)
	axs2[0].yaxis.set_ticks(list(range(len(features_names))))
	axs2[0].set_yticklabels(features_names)
	axs2[0].set_ylim(-1, len(features_names))
	axs2[0].set_title('Features')
	plt.savefig(path_figs + 'Feature_errors' + '.pdf', dpi=600,)