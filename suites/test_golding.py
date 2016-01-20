from configure import *

import unittest
import math
import numpy as np
import matplotlib.pyplot as plt
import models, tests, capabilities

model = models.Golding()
show_plot = False
force_run = False
show_summary = True

class TestModels(unittest.TestCase):         

     def test_depolarization_block(self):
          test = tests.DepolarizationBlockTest(depolarization_block(), force_run=force_run, show_plot=show_plot, 
                                               data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          self.assertTrue(math.isnan(score.getScore("Ith").score))
          self.assertTrue(math.isnan(score.getScore("Veq").score))
          if show_summary:
               score.summarize()

     def test_oblique_integration(self):
          test = tests.ObliqueIntegrationTest(oblique_integration(), force_run_synapse=force_run, 
                                              force_run_bin_search=force_run, show_plot=show_plot, 
                                              data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.score_l[0], 4.2386207085898954e-07)
          np.testing.assert_almost_equal(score.score_l[1], 7.9630234104597583e-06) 
          np.testing.assert_almost_equal(score.score_l[2], 2.183855730952043e-14)
          np.testing.assert_almost_equal(score.score_l[3], 0.31437614949977505)
          np.testing.assert_almost_equal(score.score_l[4], 0.43356161861521747) 
          np.testing.assert_almost_equal(score.score_l[5], 0.51942417684905751)
          np.testing.assert_almost_equal(score.score_l[6], 0.00023773718885129706) 
          np.testing.assert_almost_equal(score.score_l[7], 3.4287729915555374e-09) 
          np.testing.assert_almost_equal(score.score_l[8], 0.036830882081799715)
          if show_summary:
               score.summarize()

     def test_somatic_features(self):
          observations, stimuli = somatic_features()
          test = tests.SomaticFeaturesTest(observations, stimuli=stimuli, force_run=force_run, show_plot=show_plot, 
                                           data_directory=DATA_DIR, fig_directory=FIGS_DIR)

          score = test.judge(model)

          np.testing.assert_almost_equal(score.score, 225.819789206)
          if show_summary:
               score.summarize()

if __name__ == '__main__':
     unittest.main()


