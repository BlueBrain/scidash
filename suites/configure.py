import os
import sys

paths = {}
# Path to this repository's top directory (containing models, tests, etc.)
paths['SUITE_REPO'] = os.path.expand('../..')
# Path to NEURON python libraries.  
paths['NEURON_PYTHON_PATH'] = '/Applications/NEURON/nrn/lib/python'
# Path to development version of NeuronUnit.
paths['NEURONUNIT_DEV_PATH'] = '/Users/rgerkin/Dropbox/dev/scidash/neuronunit'

for path in paths:
    if path not in sys.path:
        sys.path.append(path)

