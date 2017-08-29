#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Return residual from pyccd model using Google Earth Engine

Usage: GE_pyccd_residual.py [options]

  --path=PATH       path
  --row=ROW         row
  --lon=LON         longitude
  --lat=LAT         latitude
  --date=DATE       date to compare (%Y%j)
  --count=COUNT     count of observations after DATE
  --output=OUTPUT   output numpy file
  --start=START     starting year
  --finish=FINISH   finish year
  --expand=EXPAND   expansion distance for thumbnails


"""

from docopt import docopt
import os,sys
import numpy as np
import datetime
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import dates
from pylab import *
import wget
#Import pycc and earth engine
import ee
import ccd


# Initialize Earth Engine
ee.Initialize()


def pixel(args):

    calculate = False

    if args['--path']:
        path = int(args['--path'])
    else:
        print('Calculating path from lon/lat')
        calculate = True

    if args['--row']:
        row = int(args['--row'])
    else:
        print('Calculating row from lon/lat')
        calculate = True

    lon = float(args['--lon'])

    if np.abs(lon) > 180:
        print('Invalide longitude value')
        sys.exit()

    lat = float(args['--lat'])

    if np.abs(lat) > 90:
        print('Invalide latitude value')
        sys.exit()

    if args['--date']:
        _date = args['--date']
        dt = datetime.datetime.strptime(_date, "%Y%j")
        date = dt.toordinal()
    else:
        print('Please specify date')
        sys.exit()

    if args['--count']:
        count = int(args['--count'])
    else:
        count = 1

    if args['--expand']:
        expand = int(args['--expand'])
    else:
        count = 500

    if args['--output']:
        output = args['--output']
        saveout = True
    else:
        saveout = False

    if saveout:
        if not os.path.isdir(output):
            os.mkdir(output)

    if args['--start']:
        start = args['--start']
        start = '{a}-01-01'.format(a=start)
    else:
        start = '1984-01-01'

    if args['--finish']:
        finish = args['--finish']
        finish = '{a}-01-01'.format(a=finish)
    else:
        finish = '2017-01-01'

    #Location
    point = {'type':'Point', 'coordinates':[lon, lat]};

    if calculate:
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
            start, finish);

    l7_collection = ee.ImageCollection(
            'LANDSAT/LE7_SR').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            start, finish);
    
    l5_collection = ee.ImageCollection(
            'LANDSAT/LT5_SR').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            start, finish);

    l8_thermal = ee.ImageCollection(
            'LANDSAT/LC08/C01/T1_TOA').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            start, finish).select('B10');
        
    l7_thermal = ee.ImageCollection(
            'LANDSAT/LE07/C01/T1_TOA').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            start, finish).select('B6_VCID_1');

    l5_thermal = ee.ImageCollection(
            'LANDSAT/LT05/C01/T1_TOA').filter(
            ee.Filter.eq('WRS_PATH', path)).filter(
            ee.Filter.eq('WRS_ROW', row)).filterDate(
            start, finish).select('B6');

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

    #Get the observed values
    observed_values, after_indices = get_observed(df, date, count)

    model, location = get_model(results, date)

    print('Found model {a} date specified'.format(a=location))

    if len(model) == 1:
        model = model[0]

    if location == 'during' or 'before':
        residuals = get_during(model, observed_values)

    normalized_residuals = norm_res(residuals, model)


    if saveout:
        out = output + '/' + output
        np.save(out, np.array(residuals))
        image_count = df['id_x'].iloc[after_indices][0:count]
  
        iter = 0

        #Save image of absolute residuals
        outname = output + '/' 'absolute_residuals.png'
        save_barplot(residuals, outname, 'Absolute Residuals')

        #Save image of normalized residuals
        outname = output + '/' 'normalized_residuals.png'
        save_barplot(normalized_residuals, outname, 'Normalized Residuals')

        #Save thumbnails
        for img in image_count.values:
            save_thumbnail(img, point, output, expand, iter)
            iter += 1

def norm_res(residuals, model):
    normalized_res = {}
    normalized_res['blue'] = residuals['blue'] / model.blue.rmse
    normalized_res['green'] = residuals['green'] / model.green.rmse
    normalized_res['red'] = residuals['red'] / model.red.rmse
    normalized_res['nir'] = residuals['nir'] / model.nir.rmse
    normalized_res['swir1'] = residuals['swir1'] / model.swir1.rmse
    normalized_res['swir2'] = residuals['swir2'] / model.swir2.rmse
    normalized_res['thermal'] = residuals['thermal'] / model.thermal.rmse

    return normalized_res

#plot_results(results, df, band, plotband, dates, yl, plotlabel)
def save_thumbnail(img, point, outdir, expand, iter):
    """Save thumbnails of images after date """
    sensor = img[0:3]
    filename = 'LANDSAT/' + sensor +'_SR/' + img
    image = ee.Image(filename).select(['B3','B2','B1'])
 
    point2plot = ee.Geometry.Point(point['coordinates'])
    bounds2plot = point2plot.buffer(expand).bounds().getInfo()['coordinates']

    thumb = image.getThumbURL({'min': 0, 'max': 1200, 'region': bounds2plot, 'format': 'jpg'})

    out_save = outdir + '/' + str(iter) + '_' + img + '.jpg'
    wget.download(thumb, out_save)

def save_barplot(residuals, output, ylabel):
    keys = residuals.keys()

    means = []
    stds = []

    for i in keys:
        means.append(residuals[i].mean())
        stds.append(residuals[i].std())

    ind = np.arange(len(keys))
    width = 0.35 

    fig, ax = plt.subplots()

    rects = ax.bar(ind, means, width, color='r', yerr=stds)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ind)
    ax.set_xticklabels(keys)

    plt.savefig(output)

def get_during(result, observed_values):
    predicted_values = {}
    residuals = {}
    prediction_dates = []
    #days = np.arange(result.start_day, result.end_day + 1)
    #Add extra data in case it's before #TODO
    days = np.arange(result.start_day, result.end_day + 5000)
    prediction_dates.append(days)

    #Blue
    intercept = result.blue.intercept
    coef = result.blue.coefficients
    predicted_values['B1'] = predict_band(intercept, coef, days, observed_values)
    residuals['blue'] = observed_values['B1'].values - predicted_values['B1'][:,0]

    #Green
    intercept = result.green.intercept
    coef = result.green.coefficients
    predicted_values['B2'] = predict_band(intercept, coef, days, observed_values)
    residuals['green'] = observed_values['B2'].values - predicted_values['B2'][:,0]
    
    #Red
    intercept = result.red.intercept
    coef = result.red.coefficients
    predicted_values['B3'] = predict_band(intercept, coef, days, observed_values)
    residuals['red'] = observed_values['B3'].values - predicted_values['B3'][:,0]
    
    #NIR
    intercept = result.nir.intercept
    coef = result.nir.coefficients
    predicted_values['B4'] = predict_band(intercept, coef, days, observed_values)
    residuals['nir'] = observed_values['B4'].values - predicted_values['B4'][:,0]
    
    #SWIR1
    intercept = result.swir1.intercept
    coef = result.swir1.coefficients
    predicted_values['B5'] = predict_band(intercept, coef, days, observed_values)
    residuals['swir1'] = observed_values['B5'].values - predicted_values['B5'][:,0]
    
    #SWIR2
    intercept = result.swir2.intercept
    coef = result.swir2.coefficients
    predicted_values['B7'] = predict_band(intercept, coef, days, observed_values)
    residuals['swir2'] = observed_values['B7'].values - predicted_values['B7'][:,0]

    #Thermal
    intercept = result.thermal.intercept
    coef = result.thermal.coefficients
    predicted_values['thermal'] = predict_band(intercept, coef, days, observed_values)
    residuals['thermal'] = observed_values['thermal'].values - predicted_values['thermal'][:,0]

    residuals_df = pd.DataFrame(residuals)
    return residuals


def predict_band(intercept, coef, days, observed_values):

    predicted_values = []
    predict_indices = []
    predict = []

    predicted_values.append(intercept + coef[0] * days +
                            coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
                            coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
                            coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))

    for i in observed_values['time'].values:
        ind = np.where(days == i)[0]
        predict.append(predicted_values[0][ind])

    return np.array(predict)

    
def get_model(results, date):
    """ return model to use for comparing to data """

    before_models = []
    during_models = []
    after_models = []
    for i, a in enumerate(results['change_models']):
        if a.end_day < date:
            before_models.append(a)
        elif a.end_day > date and a.start_day < date:
            during_models.append(a)
        elif a.end_day > date and a.start_day > date:
            after_models.append(a)

    if len(during_models) == 1:
        return during_models, 'during'
    elif len(during_models) == 0 and len(before_models) > 0:
        return before_models[-1], 'before'
    elif len(during_models) == 0 and len(before_models) == 0:
        return after_models[0], 'after'
    else:
        print('no models calculated')
        sys.exit()
        

def get_observed(df, date, count):
    """ Return observed values for [count] observations after [date] """


    #Indices after date
    after_indices = np.where(df['time'] > date)[0]

    #Values after date
    after_values = df.iloc[after_indices,:]

    #Clear values after date
    after_clear = after_values[after_values['cfmask'] < 3]
    after_indices = after_indices[after_values['cfmask'] < 3]

    #Clear values for count specified
    after_count = after_clear.iloc[0:count,:]


    return after_count, after_indices


def make_db(collection, point, band_list, rename_list):
    info = collection.getRegion(point, 1).getInfo()
    header = info[0]
    files = array(info[0])
    data = array(info[1:])
    #data = array(info[:])
    iTime = header.index('time')
    time = [datetime.datetime.fromtimestamp(i/1000) for i in (data[0:,iTime].astype(int))]
    time_new = [t.toordinal() for t in (time)]


    iBands = [header.index(b) for b in band_list]
    yData = data[0:,iBands].astype(np.float)

    red = yData[:,0]
    df = pd.DataFrame(data=yData, index=list(range(len(red))), columns=rename_list)
    df['time'] = time_new
    df['id'] = data[0:, 0]
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

