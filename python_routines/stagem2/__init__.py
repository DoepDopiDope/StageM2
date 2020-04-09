from . import distrib
from . import position
from . import utils
from . import cluster
from os.path import join

import numpy as np


# __data_dir__ = np.char.replace(join(utils.__pkg_dir__, "data"), "\\", "/")
__data_dir__ = join(utils.__pkg_dir__, "data")
