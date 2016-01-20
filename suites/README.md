Hippounit Test Suites
=====================

Runing 
-------
To run the test for the model and see the summary score, run:
`$ python test_<model>.py`

Figures are stored as pdf files, default location records/figures/<model>

Settings
--------
hippounit_setup.py contains information about the directory structure, location of models and data, and empircal observations for tests.

Notes
-----
- Due to naming issues with the NEURON hoc files, only one model can be initalized in memory at once.

