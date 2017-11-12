import argparse
import logging
import numpy as np
import tensorflow as tf
import time

import utils
import dataset
import quality
import scipy
from separated_bit_model import SeparatedBitModel