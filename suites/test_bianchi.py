from configure import *

import unittest
import numpy as np
import matplotlib.pyplot as plt
import models, tests, capabilities


model = models.Bianchi()
show_plot = False
force_run = False
show_summary = False

class TestModels(unittest.TestCase):         

     def test_depolarization_block(self):
          test = tests.DepolarizationBlockTest(depolarization_block(), force_run=force_run, show_plot=show_plot, 
                                               data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.getScore("Ith").score, 0.83333, 4)
          np.testing.assert_almost_equal(score.getScore("Veq").score, -0.94929, 4)
          if show_summary:
               score.summarize()

     def test_oblique_integration(self):
          test = tests.ObliqueIntegrationTest(oblique_integration(), force_run_synapse=force_run, 
                                              force_run_bin_search=force_run, show_plot=show_plot, 
                                              data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.score_l[0], 1.9517183167468579e-14) 
          np.testing.assert_almost_equal(score.score_l[1], 8.5447472746673138e-08) 
          np.testing.assert_almost_equal(score.score_l[2], 7.8909253170837084e-14) 
          np.testing.assert_almost_equal(score.score_l[3], 0.53803962697540586)
          np.testing.assert_almost_equal(score.score_l[4], 0.0027690826314024858) 
          np.testing.assert_almost_equal(score.score_l[5], 0.0034244305344784936) 
          np.testing.assert_almost_equal(score.score_l[6], 0.49924558358277982)
          np.testing.assert_almost_equal(score.score_l[7], 2.0117657348697201e-10) 
          np.testing.assert_almost_equal(score.score_l[8], 0.23341797317947133)
          if show_summary:
               score.summarize()

     def test_somatic_features(self):
          observations, stimuli = somatic_features()
          test = tests.SomaticFeaturesTest(observations, stimuli=stimuli, force_run=force_run, show_plot=show_plot, 
                                           data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.score, 61.804666723)
          if show_summary:
               score.summarize()


if __name__ == '__main__':
     unittest.main()


