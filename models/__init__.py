import os
import numpy as np
import sciunit
from sciunit.utils import printd
from neuron import h
from capabilities import Hoc, CurrentClamp, DendriticSynapse


# Possible TODOs:
# - Make NeuronModel a sciUnit.Capability class
# - combine set_cclamp and run_cclamp into one method. 
# - create a single abstract method for setting up and running a dendritic pulse
class NeuronModel(sciunit.Model, Hoc, CurrentClamp, DendriticSynapse):

    def __init__(self, name, modelpath, libfile, hocfile, force_compile=False):
        """        
        Parameters
        ==========
        name : string
               Name of model
        modelpath : string
                    Directory where to compile (i.e. nrnivmodl) the model.
        libfile : string
                  Location where the built model library file is located.
        hocfile : string
                  Location of the hoc file used to run the model.
        force_compile : bool
                        will force nrnivmodl to be ran even if libfile exists
        """
        sciunit.Model.__init__(self, name=name)

        self.name = name
        self.modelpath = modelpath
        self.libfile = libfile
        self.hocfile = hocfile

        # TODO: Allow users to specify build command.
        if (force_compile) or (os.path.isfile(self.libfile) is False):
            os.system("cd " + self.modelpath + "; nrnivmodl")
        h.nrn_load_dll(self.libfile)


        self.threshold = -20
        self.stim = None
        self.soma = None

        self.c_step_start = 0.00004
        self.c_step_stop = 0.000004
        self.c_minmax = np.array([0.00004, 0.004])
        self.dend_loc = []

        self.AMPA_tau1 = 0.1
        self.AMPA_tau2 = 2
        self.start = 150

        self.ns = None
        self.ampa = None
        self.nmda = None
        self.ampa_nc = None
        self.nmda_nc = None
        self.ndend = None
        self.xloc = None
        self.dendrite = None

    def translate(self, sectiontype):
        if "soma" in sectiontype:
            return self.soma
        else:
            # Not fully implemented
            return False

    def initialise(self):
        h.load_file(self.hocfile)

    def set_cclamp(self, amp, delay=500, dur=1000, section=None, loc=None):
        """ Set up a current clamp to be ran on the model.

        Parameters
        ==========
        amp : float
              Aplitude of current.
        delay : Integer
                Time in milliseconds before starting stimulation
        durtion : Integer
                  Time in milliseconds of stimulation
        section : NEURON identifer
                  NEURON section of model to place stimulation electrode
        loc : Float [0, 1] 
              location along section to place stimulation electrode.
        """
        if section == None or loc == None:
            # No section and/or location specified to stimulate, just set current clamp on the soma
            self.stim = h.IClamp(self.get_soma())
        else:
            # Set clamp at a specific location
            exec("self.sect_loc=h." + str(section)+"("+str(loc)+")")
            self.stim = h.IClamp(self.sect_loc)

        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def run_cclamp(self, section=None, loc=None):
        """ Run a current clamp and record the membrane potential.

        Parameters
        ==========
        section : NEURON identifer
                  NEURON section of model to place recording electrode
        loc : Float [0, 1] 
              location along section to place recording electrode.
        """
        printd(("- running model", self.name))

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        if section == None or loc == None:
            # set to record current from the soma
            rec_v.record(self.get_soma()._ref_v)
        else:
            # Set recording electrode at specific location
            exec("self.sect_loc=h." + str(section)+"("+str(loc)+")")
            rec_v.record(self.sect_loc._ref_v)
        
        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = np.array(rec_t)
        v = np.array(rec_v)

        return t, v

    ######### Capablity for AMPA/NMDA synapse ###########
    def set_ampa_nmda(self, section, xloc):
        """Creates a synapse with AMPA and NMDA receptor dynamics at dend_loc0.

        Parameters
        ==========
        section : Identifier (string or int)
                  identification of denrite branch section.
        xloc : float
               loction of synapse along the section.
        """

        self.ampa = h.Exp2Syn(xloc, sec=section)
        self.ampa.tau1 = self.AMPA_tau1
        self.ampa.tau2 = self.AMPA_tau2

        self.nmda = h.NMDA_JS(xloc, sec=section)


    def set_netstim_netcon(self, interval):
        """Initializes the presynaptic synaptic stimulation (netstim) and network connection (netcon) 
        objects for neuron.

        Parameters
        ==========
        interval : float (seconds)
                   the interval of stimulation
        """
        self.ns = h.NetStim()
        self.ns.interval = interval
        self.ns.number = 0
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)
        self.nmda_nc = h.NetCon(self.ns, self.nmda, 0, 0, 0)


    def set_num_weight(self, number=1, AMPA_weight=0.0004):
        """Sets the number of presynaptic spikes and synaptic connection weight.

        Parameters
        ==========
        number : Integer
                 number of spikes generated for the presynaptic stimulation
        AMPA_weight: float
                     A weight to set the synaptic connection
        """

        self.ns.number = number
        self.ampa_nc.weight[0] = AMPA_weight
        self.nmda_nc.weight[0] = AMPA_weight/0.2

    def run_syn(self):
        """Runs a presynaptic stimulation (Netstim) and captures the postsynaptic soma potential 
        and potential at an AMPA/NMDA synapse on the dendrite.

        Returns
        =======
        t : list 
            The runtime span.
        v : list
            The potentials recorded in the center of the soma.
        v_dend : list
                 The potentials recorded at the dendritic synapse (section=self.ndend, location=self.xloc)
        """

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.get_soma()._ref_v)

        rec_v_dend = h.Vector()
        #rec_v_dend.record(self.dendrite)
        rec_v_dend.record(self.get_dendrite()._ref_v)

        printd(("- running model", self.name))
        # initialze and run
        # h.load_file("stdrun.hoc")
        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 300
        h.run()

        # get recordings
        t = np.array(rec_t)
        v = np.array(rec_v)
        v_dend = np.array(rec_v_dend)

        return t, v, v_dend


class KaliFreund(NeuronModel):

    def __init__(self, name="Kali", modelpath=None, libfile=None, hocfile=None, force_compile=False):
        if modelpath == None:
            modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hoc_models/Kali_Freund_modell/scppinput/")
        if libfile == None:
            libfile = os.path.join(modelpath, "x86_64/.libs/libnrnmech.so.0")
        if hocfile == None:
            hocfile = os.path.join(modelpath, "ca1_syn.hoc")

        super(KaliFreund, self).__init__(name, modelpath, libfile, hocfile, force_compile)

        self.soma = "soma"
        self.dend_loc = [[80,0.27],[80,0.83],[54,0.16],[54,0.95],[52,0.38],[52,0.83],[53,0.17],[53,0.7],[28,0.35],[28,0.78]]


    def get_soma(self):
        return h.soma(0.5)

    def get_dendrite(self):
        return h.dendrite[self.ndend](self.xloc)

    def set_ampa_nmda(self, dend_loc0=[80, 0.27]):
        self.ndend, self.xloc = dend_loc0
        super(KaliFreund, self).set_ampa_nmda(h.dendrite[self.ndend], self.xloc)


class Migliore(NeuronModel):

    def __init__(self, name="Migliore", modelpath=None, libfile=None, hocfile=None, force_compile=False):
        if modelpath == None:
            modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hoc_models/Migliore_Schizophr/")
        if libfile == None:
            libfile = os.path.join(modelpath, "x86_64/.libs/libnrnmech.so.0")
        if hocfile == None:
            hocfile = os.path.join(modelpath, "basic_sim_9068802-test.hoc")

        super(Migliore, self).__init__(name, modelpath, libfile, hocfile, force_compile)

        self.soma = "soma[0]"
        self.dend_loc = [[17,0.3],[17,0.9],[24,0.3],[24,0.7],[22,0.3],[22,0.7],[25,0.2],[25,0.5],[30,0.1],[30,0.5]]
        self.trunk_dend_loc_distr = [[10,0.167], [10,0.5], [10,0.83], [11,0.5], [9,0.5], [8,0.5], [7,0.5]]
        self.trunk_dend_loc_clust = [10,0.167]


    def get_soma(self):
        return h.soma[0](0.5)

    def get_dendrite(self):
        return h.apical_dendrite[self.ndend](self.xloc)

    def set_ampa_nmda(self, dend_loc0=[17, 0.3]):
        self.ndend, self.xloc = dend_loc0
        super(Migliore, self).set_ampa_nmda(h.apical_dendrite[self.ndend], self.xloc)


class Bianchi(NeuronModel):

    def __init__(self, name="Bianchi", modelpath=None, libfile=None, hocfile=None, force_compile=False):
        if modelpath == None:
            modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hoc_models/Ca1_Bianchi/experiment/")
        if libfile == None:
            libfile = os.path.join(modelpath, "x86_64/.libs/libnrnmech.so.0")
        if hocfile == None:
            hocfile = os.path.join(modelpath, "basic.hoc")

        super(Bianchi, self).__init__(name, modelpath, libfile, hocfile, force_compile)

        self.soma = "soma[0]"
        self.dend_loc = [[112,0.375], [112,0.875], [118,0.167], [118,0.99], [30,0.167], [30,0.83], [107,0.25],[107,0.75],[114,0.01],[114,0.75]]
        self.trunk_dend_loc_distr = [[65,0.5], [69,0.5], [71,0.5], [64,0.5], [62,0.5], [60,0.5], [81,0.5]]
        self.trunk_dend_loc_clust = [65,0.5]

    def get_soma(self):
        return h.soma[0](0.5)

    def get_dendrite(self):
        return h.apical_dendrite[self.ndend](self.xloc)

    def set_ampa_nmda(self, dend_loc0=[112, 0.375]):
        self.ndend, self.xloc = dend_loc0
        section = h.apical_dendrite[self.ndend]
        super(Bianchi, self).set_ampa_nmda(section, self.xloc)



class Golding(NeuronModel):

    def __init__(self, name="Golding", modelpath=None, libfile=None, hocfile=None, force_compile=False):
        if modelpath == None:
            modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hoc_models/Golding_dichotomy/fig08/")
        if libfile == None:
            libfile = os.path.join(modelpath, "x86_64/.libs/libnrnmech.so.0")
        if hocfile == None:
            hocfile = os.path.join(modelpath, "run_basic.hoc")

        super(Golding, self).__init__(name, modelpath, libfile, hocfile, force_compile)

        self.soma = "somaA"
        self.dend_loc = [["dendA5_00",0.275],["dendA5_00",0.925],["dendA5_01100",0.15],["dendA5_01100",0.76],["dendA5_0111100",0.115],["dendA5_0111100",0.96],["dendA5_01111100",0.38],["dendA5_01111100",0.98],["dendA5_0111101",0.06],["dendA5_0111101",0.937]]
        self.trunk_dend_loc_distr = [["dendA5_01111111111111",0.68], ["dendA5_01111111111111",0.136], ["dendA5_01111111111111",0.864], ["dendA5_011111111111111",0.5], ["dendA5_0111111111111111",0.5], ["dendA5_0111111111111",0.786], ["dendA5_0111111111111",0.5]]
        self.trunk_dend_loc_clust = ["dendA5_01111111111111",0.68]      

    def get_soma(self):
        return h.somaA(0.5)

    def get_dendrite(self):
        return self.dendrite(self.xloc)

    def set_ampa_nmda(self, dend_loc0=["dendA5_01111111100", 0.375]):
        self.ndend, self.xloc = dend_loc0
        exec("self.dendrite=h." + self.ndend)
        section = self.dendrite
        super(Golding, self).set_ampa_nmda(section, self.xloc)


