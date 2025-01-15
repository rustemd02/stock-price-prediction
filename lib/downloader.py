# -*- coding: utf-8 -*-

import logging

from lib.helpers import daterange, DATE_FORMAT
from lib.moex_downloader import get_moex_data


def downloader(args):
    if args.dateend:
        dates = daterange(args.date, args.dateend)
    else:
        dates = [args.date]

    for day in dates:
        if day.weekday() >= 5:
            # без выходных
            continue
        day_str = day.strftime(DATE_FORMAT)
        logging.info((args.engine, args.market, day_str))
        get_moex_data(args.engine, args.market, day_str)