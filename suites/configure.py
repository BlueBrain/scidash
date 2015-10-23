import os
import sys

# Use *'s in path names to prepend instead of append them to the Python path.  

paths = {}
# Path to this repository's top directory (containing models, tests, etc.)
paths['SUITE_REPO'] = os.path.abspath('../..')
# Path to NEURON python libraries.  
paths['NEURON_PYTHON_PATH'] = '/Applications/NEURON/nrn/lib/python'

# Path to development version of NeuronUnit.
paths['SCIUNIT_DEV_PATH*'] = '/Users/rgerkin/Dropbox/dev/scidash/sciunit'

# Path to development version of NeuronUnit.
paths['NEURONUNIT_DEV_PATH*'] = '/Users/rgerkin/Dropbox/dev/scidash/neuronunit'

for path_name,path in paths.items():
    if path not in sys.path:
        if '*' in path_name:
            sys.path.insert(1,path)
        else:
            sys.path.append(path)

