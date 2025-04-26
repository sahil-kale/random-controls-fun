import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mrac_fun.mrac_mimo_core import MIMOMRACController

import tclab
import numpy as np
# add the .. to path