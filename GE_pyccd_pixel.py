#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Pixel plotting using pyccd and Google Earth Engine

Usage: GE_pyccd_pixel.py [options]

  --path=PATH   path
  --row=ROW     row
  --band=BAND   band
  --lon=LON     longitude
  --lat=LAT     latitude
  --yl=ylim     y limit


"""

from docopt import docopt
import os,sys
import numpy as np
import datetime
import pandas as pd
import matplotlib.dates as mdates
from IPython.display import Image
from matplotlib import dates
from pylab import *

#Import pycc and earth engine
import ee
import ccd


# Initialize Earth Engine
ee.Initialize()


def pixel(args):
    if args['--path']:
        path = int(args['--path'])
    else:
        print('Calculating path from lon/lat')

    if args['--row']:
        row = int(args['--row'])
    else:
        print('Calculating row from lon/lat')

    lon = float(args['--lon'])

    if np.abs(lon) > 180:
        print('Invalide longitude value')
        sys.exit()

    lat = float(args['--lat'])

    if np.abs(lat) > 90:
        print('Invalide latitude value')
        sys.exit()

    if args['--band']:
        band = args['--band']
    else:
        band = 4
        print('No band specified, defaulting to 4')

    yl = None 
    if args['--yl']:
        yl1 = args['--yl']
        yl = [int(i) for i in (yl1.split(' '))]

    #Location
    point = {'type':'Point', 'coordinates':[lon, lat]};

     #WRS-2 outline
    fc = ee.FeatureCollection('ft:1_RZgjlcqixp-L9hyS6NYGqLaKOlnhSC35AB5M5Ll');

    #Get overlap
    pgeo = ee.Geometry.Point([lon, lat]);
    cur_wrs = fc.filterBounds(pgeo);
    path = cur_wrs.first().get('PATH');
    row = cur_wrs.first().get('ROW');
    print('Path: {}'.format(int(path.getInfo())));
    print('Row: {}'.format(int(row.getInfo())));

    # Create image collection

    #Landsat Collection. TODO: How to reduce line size with API? 
    l8_collection = ee.ImageCollection(
            'LANDSAT/LC8_SR').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            '2012-01-01', '2018-01-01');

    l7_collection = ee.ImageCollection(
            'LANDSAT/LE7_SR').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            '1984-01-01', '2018-01-01');
    
    l5_collection = ee.ImageCollection(
            'LANDSAT/LT5_SR').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            '1984-01-01', '2018-01-01');

    l8_thermal = ee.ImageCollection(
            'LANDSAT/LC08/C01/T1_TOA').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            '2012-01-01', '2018-01-01').select('B10');
        
    l7_thermal = ee.ImageCollection(
            'LANDSAT/LE07/C01/T1_TOA').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            '1984-01-01', '2018-01-01').select('B6_VCID_1');

    l5_thermal = ee.ImageCollection(
            'LANDSAT/LT05/C01/T1_TOA').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            '1984-01-01', '2018-01-01').select('B6');

    #LC8 Band names
    band_list = ['B2','B3','B4','B5','B6','B7','cfmask','cfmask_conf']

    #Names to rename LC8 to / L7L5 band names
    rename_list = ['B1','B2','B3','B4','B5','B7','cfmask','cfmask_conf']

    #L8
    df_sr = make_db(l8_collection, point, band_list, rename_list)

    #L7
    df_sr2 = make_db(l7_collection, point, rename_list, rename_list)
    df_sr = update_df(df_sr, df_sr2)

    #L5
    df_sr2 = make_db(l5_collection, point, rename_list, rename_list)
    df_sr = update_df(df_sr, df_sr2)

    #thermal
    band_list = ['B6']
    rename_list = ['thermal']
    df_thermal = make_db(l5_thermal, point, band_list, rename_list)

    band_list = ['B6_VCID_1']
    df_thermal2 = make_db(l7_thermal, point, band_list, rename_list)
    df_thermal = update_df(df_thermal, df_thermal2)

    band_list = ['B10']
    df_thermal2 = make_db(l8_thermal, point, band_list, rename_list)
    df_thermal = update_df(df_thermal, df_thermal2)

    #Merge the thermal and SR
    df = pd.merge(df_sr, df_thermal, on='time')
    df = df.sort_values('time')

    #Get rid of NaNs
#    df['cfmask'][df['cfmask'].isnull()] = 4
#    df[df.isnull()] = 0

    #Scale brightness temperature by 10 for pyccd
    df['thermal'] = df['thermal'] * 10

    #TODO: Paramaterize everything
    params = {'QA_BITPACKED': False,
              'QA_FILL': 255,
              'QA_CLEAR': 0,
              'QA_WATER': 1,
              'QA_SHADOW': 2,
              'QA_SNOW': 3,
              'QA_CLOUD': 4}

    dates = np.array(df['time'])
    blues = np.array(df['B1'])
    greens = np.array(df['B2'])
    reds = np.array(df['B3'])
    nirs = np.array(df['B4'])
    swir1s = np.array(df['B5'])
    swir2s = np.array(df['B7'])
    thermals = np.array(df['thermal'])
    qas = np.array(df['cfmask'])
    results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, params=params)

    band_names = ['Blue SR', 'Green SR', 'Red SR', 'NIR SR', 'SWIR1 SR', 'SWIR2 SR','Thermal']
    plotlabel = band_names[band] 

    plot_arrays = [blues, greens, reds, nirs, swir1s, swir2s]
    plotband = plot_arrays[band]

    plot_results(results, df, band, plotband, dates, yl, plotlabel)
    

def make_db(collection, point, band_list, rename_list):
    info = collection.getRegion(point, 30).getInfo()
    header = info[0]
    data = array(info[1:])
    iTime = header.index('time')
    time = [datetime.datetime.fromtimestamp(i/1000) for i in (data[0:,iTime].astype(int))]
    time_new = [t.toordinal() for t in (time)]


    iBands = [header.index(b) for b in band_list]
    yData = data[0:,iBands].astype(np.float)

    red = yData[:,0]
    df = pd.DataFrame(data=yData, index=list(range(len(red))), columns=rename_list)
    df['time'] = time_new
    return df

def update_df(df, df2):
    df = df.append(df2)
    return df

def plot_results(results, df, band, plotband, dates, yl, ylabel):
    mask = results['processing_mask']
    predicted_values = []
    prediction_dates = []
    break_dates = []
    start_dates = []


    for num, result in enumerate(results['change_models']):
        print('Result: {}'.format(num))
        print('Start Date: {}'.format(datetime.datetime.fromordinal(result.start_day)))
        print('End Date: {}'.format(datetime.datetime.fromordinal(result.end_day)))
        print(result.break_day)
        print('Break Date: {}'.format(datetime.datetime.fromordinal(result.break_day)))
        print('QA: {}'.format(result.curve_qa))
        print('Norm: {}\n'.format(np.linalg.norm([result.green.magnitude,
                                                 result.red.magnitude,
                                                 result.nir.magnitude,
                                                 result.swir1.magnitude,
                                                 result.swir2.magnitude])))
        print('Change prob: {}'.format(result.change_probability))
    
        days = np.arange(result.start_day, result.end_day + 1)
        prediction_dates.append(days)
        break_dates.append(result.break_day)
        start_dates.append(result.start_day)

        intercept = result[6+band].intercept
        coef = result[6+band].coefficients
    
        predicted_values.append(intercept + coef[0] * days +
                                coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
                                coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
                                coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))
    
    plt.style.use('ggplot')

    fg = plt.figure(figsize=(16,9), dpi=300)

    a1 = fg.add_subplot(2, 1, 1, xlim=(min(dates), max(dates)))

    plot_dates = np.array([datetime.datetime.fromordinal(i) for i in (dates)])

    a1.plot(plot_dates[mask], plotband[mask], 'k*', ms=2, label='Clear observation') # Observed values
    a1.plot(plot_dates[~mask], plotband[~mask], 'r+', ms=1, label='Masked observation') # Observed values masked out

    # Predicted curves
    iter = 0
    for _preddate, _predvalue in zip(prediction_dates, predicted_values):
        if iter == 0:
            a1.plot(_preddate, _predvalue, 'orange', linewidth=1, label='PyCCD Model')
            iter += 1 
        else:
            a1.plot(_preddate, _predvalue, 'orange', linewidth=1)

    for b in break_dates: a1.axvline(b)
    for s in start_dates: a1.axvline(s, color='r')

    if yl:
        a1.set_ylim(yl)

    plt.ylabel(ylabel)
    plt.xlabel('Date')


    a1.legend(loc=2, fontsize=5)
    plt.show()


if __name__ == '__main__':
    args = docopt(__doc__, version='0.6.2')
    pixel(args)

