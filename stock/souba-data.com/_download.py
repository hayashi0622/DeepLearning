# coding: utf-8

import os
import time
import datetime

d = datetime.datetime.strptime('2018/01/01', '%Y/%m/%d')
while d < datetime.datetime.now():
    file_name = 'T{0:02d}{1:02d}{2:02d}.{3}'.format(d.year % 100, d.month, d.day, 'lzh' if d.year < 2015 else 'zip')
    if not os.path.exists(file_name):
        url = 'http://souba-data.com/k_data/{0}/{1:02d}_{2:02d}/{3}'.format(d.year, d.year % 100, d.month, file_name)
        os.system('wget {0}'.format(url))
        time.sleep(1)
    d = d + datetime.timedelta(days=1)
