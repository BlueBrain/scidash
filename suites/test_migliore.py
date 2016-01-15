from hippounit_setup import *

import unittest
import math
import numpy as np
import matplotlib.pyplot as plt
import models, tests, capabilities

model = models.Migliore()
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

          np.testing.assert_almost_equal(score.score_l[0], 4.0532286169480288e-08)
          np.testing.assert_almost_equal(score.score_l[1], 5.2021270634561647e-06)
          np.testing.assert_almost_equal(score.score_l[2], 1.3297386311246351e-11) 
          np.testing.assert_almost_equal(score.score_l[3], 0.096464685360991523)
          np.testing.assert_almost_equal(score.score_l[4], 0.0016926044408459807)
          np.testing.assert_almost_equal(score.score_l[5], 0.0042065637580864287)
          np.testing.assert_almost_equal(score.score_l[6], 0.090193377683313508)
          np.testing.assert_almost_equal(score.score_l[7], 3.4710455815903194e-11) 
          np.testing.assert_almost_equal(score.score_l[8], 0.30011440172735004)
          if show_summary:
               score.summarize()


     def test_somatic_features(self):
          observations, stimuli = somatic_features()
          test = tests.SomaticFeaturesTest(observations, stimuli=stimuli, force_run=force_run, show_plot=show_plot, 
                                           data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.score, 44.8044558658)
          if show_summary:
               score.summarize()


if __name__ == '__main__':
     unittest.main()


