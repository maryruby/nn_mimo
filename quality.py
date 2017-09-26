#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def bit_error_rate(predicted, real):
    """Returns rate of correctly predicted bits (accuracy) aka BER"""
    total_elems = (real.shape[0] * real.shape[1])
    return np.sum(np.not_equal(predicted, real)) / float(total_elems)


def column_bit_error_rate(predicted, real):
    """Returns numpy array with BER for each column of data"""
    return np.sum(np.not_equal(predicted, real), 1) / real.shape[1]


def row_error_rate(predicted, real):
    """Returns rate of correctly predicted rows"""
    total_elems = (real.shape[0] * real.shape[1])
    return np.sum(np.any(np.not_equal(predicted, real), 1)) / float(total_elems)
