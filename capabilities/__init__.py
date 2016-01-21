import sciunit

class Hoc(sciunit.Capability):
	"""Indicates the model is stored as a NEURON hoc file."""

	def initialise(self):
		"""Initiailzes the hoc file"""
		raise NotImplementedError()

	def translate(self, sectiontype):
		"""Determines the propery ID for the section"""
		raise NotImplementedError()


class CurrentClamp(sciunit.Capability):
	"""
	Indicates that the model can run a current clamp and
	return the membrane potential time-course.
	"""

	def set_cclamp(self, amp, delay=500, dur=1000, section=None, loc=None):
		"""Set the properities of the current clamp"""
		raise NotImplementedError()
	
	def run_cclamp(self, section=None, loc=None):
		"""Run the current clamp, returns time-course and membrane at section/loc"""
		raise NotImplementedError()

class DendriticSynapse(sciunit.Capability):
	"""
	Indicates that a dendritic synapse can be stimulated, and membrane potential
	is recorded from the synapse and soma.
	"""
	def set_ampa_nmda(self, xloc, ndend, section):
		raise NotImplementedError()

	def set_netstim_netcon(self, interval):
		raise NotImplementedError()

	def set_num_weight(self, number=1, AMPA_weight=0.0004):
		raise NotImplementedError()

	def run_syn(self):
		raise NotImplementedError()

