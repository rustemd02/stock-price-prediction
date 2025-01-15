# -*- coding: utf-8 -*-

import os
import logging
from datetime import timedelta

DATE_FORMAT = '%Y-%m-%d'


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def read_last_line(filepath):
    with open(filepath, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        return f.readline().decode().replace('\n', '')