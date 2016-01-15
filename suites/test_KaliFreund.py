from hippounit_setup import *

import unittest
import numpy as np
import matplotlib.pyplot as plt
import models, tests, capabilities


model = models.KaliFreund()
show_plot = False
force_run = False
show_summary = True

class TestModels(unittest.TestCase):         

     def test_depolarization_block(self):
          test = tests.DepolarizationBlockTest(depolarization_block(), force_run=force_run, show_plot=show_plot, 
                                               data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.getScore("Ith").score, 2.833333333333334)
          np.testing.assert_almost_equal(score.getScore("Veq").score, 4.652044391936047)
          if show_summary:
               score.summarize()

     def test_oblique_integration(self):
          test = tests.ObliqueIntegrationTest(oblique_integration(), force_run_synapse=force_run, 
                                              force_run_bin_search=force_run, show_plot=show_plot, 
                                              data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.score_l[0], 0.0098868006245165369)
          np.testing.assert_almost_equal(score.score_l[1], 0.023209135665234509)
          np.testing.assert_almost_equal(score.score_l[2], 1.8357818838827194e-09)
          np.testing.assert_almost_equal(score.score_l[3], 0.034273409886929292)
          np.testing.assert_almost_equal(score.score_l[4], 0.23243691990166457)
          np.testing.assert_almost_equal(score.score_l[5], 0.68961118007070865)
          np.testing.assert_almost_equal(score.score_l[6], 0.0013060249694657844)
          np.testing.assert_almost_equal(score.score_l[7], 7.0165878421108928e-37)
          np.testing.assert_almost_equal(score.score_l[8], 0.30425297200219531)
          if show_summary:
               score.summarize()

     def test_somatic_features(self):
          observations, stimuli = somatic_features()
          test = tests.SomaticFeaturesTest(observations, stimuli=stimuli, force_run=force_run, show_plot=show_plot, 
                                           data_directory=DATA_DIR, fig_directory=FIGS_DIR)
          score = test.judge(model)

          np.testing.assert_almost_equal(score.score, 266.30191509770617)
          if show_summary:
               score.summarize()

if __name__ == '__main__':
     unittest.main()


