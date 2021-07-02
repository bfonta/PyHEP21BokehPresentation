# Run with `python CustomExample2.py -i SpectraGAN.hdf5 --nepochs 100`

##############################################################################
# # Custom example #2

# ## Astrophysical Spectra Generation

# ### Project: Generate realistic spectra of astrophysics objects (not included)
# - Technique: generative adversarial network with a deep convolutional architecture
# - Output: generated spectra for each iteration of the neural network training (as time progresses the spectra should look more realistic) [data stored in *SpectraGAN.hdf5*]

# ### Visualization:
# - Access all the spectra via a slider that shifts through the iterations (here called "epochs")
# - Display the means and standard deviations for thos spectra
# - Calculate the means of the distributions in the browser
##############################################################################

import os
import numpy as np
import pandas as pd
import h5py
import argparse

from bokeh.models import Text
from bokeh.models import Title
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import Legend
from bokeh.models import Range1d
from bokeh.models import CustomJS
from bokeh.models import HoverTool
from bokeh.models import LinearAxis
from bokeh.models import ColumnDataSource
from bokeh.models import SingleIntervalTicker

from bokeh.palettes import Category10
from bokeh.plotting import figure, show, save
from bokeh.layouts import layout

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    type=str,
    required=True,
    help='Input file in HDF5 format.'
)
parser.add_argument(
    '--nepochs',
    type=int,
    required=True,
    help='Input file in HDF5 format.'
)
parser.add_argument(
    '--no_stats',
    action='store_true',
    help='Consider models which do not generate means and standard deviations.'
)
FLAGS, _ = parser.parse_known_args()

# Define constants
FILENAME = FLAGS.i
RENDERERS = 'source1_init', 'source2_init'
TOTALWIDTH, WIDTHRATIO = 2500, 0.7
WIDTH, HEIGHT = int(TOTALWIDTH*WIDTHRATIO), 550
WIDTH2, HEIGHT2 = int(TOTALWIDTH*(1-WIDTHRATIO)), HEIGHT/2
NPOINTS = 3500
NEPOCHS = FLAGS.nepochs
NSTATS = 2000
NBINS = 100
NSPECTRA = 8
NDIGITS = int(np.floor(np.log10(np.abs(NEPOCHS))) + 1) #number of digits in a integer
YSHIFT = 0.15
LOGAXIS = False

# Define plotting help functions
def add_legend(fig, obj, label, **kwargs):
    leg = Legend(
        items=[(label, [obj])],
        **kwargs)
    fig.add_layout(leg, 'center')
    
def add_axis(fig, xlabel=None, ylabel=None, xinterval=500, yinterval=None, xbounds=None, ybounds=None):
    color = '#000066'
    AXIS_FORMATS = dict(
        minor_tick_in=None,
        minor_tick_out=None,
        major_tick_in=None,
        major_label_text_font_size="11pt",
        major_label_text_font_style="normal",
        axis_label_text_font_size="11pt",
        axis_label_text_font_style="normal",
        axis_line_color=color,
        major_tick_line_color=color,
        major_label_text_color=color,
        major_tick_line_cap="round",
        axis_line_cap="round",
        axis_line_width=1.5,
        major_tick_line_width=1.5,
    )

    if xlabel:
        xaxis = LinearAxis(
            ticker     = SingleIntervalTicker(interval=xinterval),
            axis_label = xlabel,
            **AXIS_FORMATS
        )
        if xbounds:
            xaxis.bounds = xbounds
        fig.add_layout(xaxis, 'below')

    if ylabel:
        yaxis = LinearAxis(
            ticker     = SingleIntervalTicker(interval=yinterval),
            axis_label = ylabel,
            **AXIS_FORMATS
        )
        if ybounds:
            yaxis.bounds = ybounds
        fig.add_layout(yaxis, 'left')

def create_figure(w, h, title, xlims=None, ylims=None, xaxislog=False):
    TOOLTIPS = [('wavelength', '$x ' + u'\u212b')]
    p = figure(plot_width=w, plot_height=h, x_axis_type='log' if xaxislog else 'linear',
               tools='hover,pan,wheel_zoom,box_zoom,reset', toolbar_location='above',
               tooltips=TOOLTIPS)
    p.toolbar.logo = None
    #p.toolbar_location = None
    p.axis.visible = False
    p.title = Title(text=title, text_font_size='12pt')
    if ylims:
        p.y_range = Range1d(ylims[0],ylims[1])
    if xlims:
        p.x_range = Range1d(xlims[0],xlims[1])
    return p

def add_graph(fig, xname, yname, renderer_source, color):
    fig.circle(x=xname, y=yname, size=1, color=color, source=renderer_source)
    fig.line(x=xname, y=yname, color=color, source=renderer_source)

def add_bars(fig, xname, yname, width, renderer_source, color):
    fig.vbar(x=xname, top=yname, bottom=0, width=width, fill_color=color, line_color='black', source=renderer_source)

def add_vline(fig, xname, yname, width, renderer_source, color):
    return fig.line(x=xname, y=yname, line_width=width, color=color, source=renderer_source)

def add_vline_static(fig, x, y, width, color):
    return fig.line(x=x, y=y, line_width=width, color=color)

##############################################################################
# Read and store the data. 
# - The wavelength values are encoded using an initial value $w_i$ and a fixed interval $w_d$ between data points. If total number of wavelength data points (```npoints```) is known, the full range can be recovered.
##############################################################################

with h5py.File(FILENAME, 'r') as f1:
    y = []
    if not FLAGS.no_stats:
        means, stds = ([] for _ in range(2))
    #print(f1['flux'].keys())

    npoints = len(f1['flux']['flux_{}'.format('0'.zfill(NDIGITS))][0,:])
    w_i = f1['meta']['wavelength'][0]
    w_d = f1['meta']['wavelength'][1]
    arr_x = np.arange(w_i, w_i+w_d*npoints, w_d)
    for e in range(NEPOCHS):
        estr = str(e).zfill(NDIGITS)
        y.append( np.expand_dims(f1['fluxsamp']['flux_samp_'+estr][:NSPECTRA,:], axis=0) )
        if not FLAGS.no_stats:
            means.append( f1['meanssamp']['means_samp_'+estr][:] )
            stds.append( f1['stdssamp']['stds_samp_'+estr][:] )
    yreal = f1['flux']['flux_{}'.format('0'.zfill(NDIGITS))][:NSPECTRA,:]
    if not FLAGS.no_stats:
        meansreal = f1['means']['means_{}'.format('0'.zfill(NDIGITS))][:]
        stdsreal = f1['stds']['stds_{}'.format('0'.zfill(NDIGITS))][:]
arr_y = np.vstack(tuple(x for x in y))
#print(arr_y.shape) NEPOCHS*NSPECTRA
if not FLAGS.no_stats:
    arr_mean = np.vstack(tuple(x for x in means))
    arr_std = np.vstack(tuple(x for x in stds))

##############################################################################
# Define further constants:
# - Strings to be used with *ColumnDataSource*s;
# - The minimum and maximum values of the histograms will be later useful for plotting.
##############################################################################

epochs = [x for x in range(NEPOCHS)]
s1, s2 = dict(), dict()
xs1, ys1 = 'wavelength', 'flux'
if not FLAGS.no_stats:
    xs2, ys2 = ('means', 'stds'), ('meanscounts', 'stdscounts')
    xse, yse = ('mean1x', 'mean2x'), ('mean1y', 'mean2y')
    wstr = 'width1', 'width2'

def handle_histogram(arr):
    h, edg = np.histogram(arr, NBINS)
    c = (edg[:-1] + edg[1:])/2
    return h, c

if not FLAGS.no_stats:
    hstdmax, hmeanmax, cmeanmax, cstdmax = (-999 for _ in range(4))
    hstdmin, hmeanmin, cmeanmin, cstdmin = (999 for _ in range(4))
    hstdmaxreal, hmeanmaxreal, cmeanmaxreal, cstdmaxreal = (-999 for _ in range(4))
    hstdminreal, hmeanminreal, cmeanminreal, cstdminreal = (999 for _ in range(4))

def get_min_max(data, dmax, dmin):
    if np.max(data) > dmax:
        dmax = np.max(data)
    if np.min(data) < dmin:
        dmin = np.min(data)
    return dmax, dmin

for epoch in range(NEPOCHS):
    s1name, s2name = 'source1_{}'.format(epoch), 'source2_{}'.format(epoch)
    s1[s1name] = ColumnDataSource(data={xs1: arr_x.tolist()})
    for i in range(NSPECTRA):
        s1[s1name].data[ys1+str(i)] = (arr_y[epoch][i]+i*YSHIFT).tolist()

    if not FLAGS.no_stats:
        hmean, cmean = handle_histogram(arr_mean[epoch])
        hstd, cstd = handle_histogram(arr_std[epoch])
        s2[s2name] = ColumnDataSource(data={xs2[0]: cmean.tolist(),
                                            ys2[0]: hmean.tolist(),
                                            xs2[1]: cstd.tolist(),
                                            ys2[1]: hstd.tolist(),
                                            wstr[0]: [cmean[1]-cmean[0] for _ in range(len(cmean))],
                                            wstr[1]: [cstd[1]-cstd[0] for _ in range(len(cstd))]})

    if epoch==0:
        s1[RENDERERS[0]] = ColumnDataSource(data={xs1: arr_x.tolist()})
        for i in range(NSPECTRA):
            s1[RENDERERS[0]].data[ys1+str(i)] = (arr_y[epoch][i]+i*YSHIFT).tolist()
        if not FLAGS.no_stats:
            initmeanx, initstdx = np.mean(cmean), np.mean(cstd)
            s2[RENDERERS[1]] = ColumnDataSource(data={xs2[0]: cmean.tolist(),
                                                      ys2[0]: hmean.tolist(),
                                                      xs2[1]: cstd.tolist(),
                                                      ys2[1]: hstd.tolist(),
                                                      wstr[0]: [cmean[1]-cmean[0] for _ in range(len(cmean))],
                                                      wstr[1]: [cstd[1]-cstd[0] for _ in range(len(cstd))]})

    if not FLAGS.no_stats:
        hmeanmax, hmeanmin = get_min_max(hmean, hmeanmax, hmeanmin)
        hstdmax, hstdmin = get_min_max(hstd, hstdmax, hstdmin)
        cmeanmax, cmeanmin = get_min_max(cmean, cmeanmax, cmeanmin)
        cstdmax, cstdmin = get_min_max(cstd, cstdmax, cstdmin)

if not FLAGS.no_stats:
    hmeanreal, cmeanreal = handle_histogram(meansreal)
    hstdreal, cstdreal = handle_histogram(stdsreal)
    hmeanmaxreal, hmeanminreal = get_min_max(hmeanreal, hstdmaxreal, hstdminreal)
    hstdmaxreal, hstdminreal = get_min_max(hstdreal, hstdmaxreal, hstdminreal)
    cmeanmaxreal, cmeanminreal = get_min_max(cmeanreal, cmeanmaxreal, cmeanminreal)
    cstdmaxreal, cstdminreal = get_min_max(cstdreal, cstdmaxreal, cstdminreal)
    
sreal1 = ColumnDataSource(data={xs1: arr_x.tolist()})
for i in range(NSPECTRA):
    sreal1.data[ys1+str(i)] = (yreal[i]+i*YSHIFT).tolist()

if not FLAGS.no_stats:
    sreal2 = ColumnDataSource(data={xs2[0]: cmeanreal.tolist(),
                                    ys2[0]: hmeanreal.tolist(),
                                    xs2[1]: cstdreal.tolist(),
                                    ys2[1]: hstdreal.tolist(),
                                    wstr[0]: [cmeanreal[1]-cmeanreal[0] for _ in range(len(cmeanreal))],
                                    wstr[1]: [cstdreal[1]-cstdreal[0] for _ in range(len(cstdreal))]})
    sreal2mean_mean = np.dot(cmeanreal,hstdreal)/np.sum(hstdreal)
    sreal2mean_std = np.dot(cstdreal,hstdreal)/np.sum(hstdreal)

    sextraname = 'sextra'
    sextra = ColumnDataSource(data={xse[0]: [initmeanx,initmeanx], yse[0]: [hmeanmin, hmeanmax],
                                    xse[1]: [initstdx,initstdx], yse[1]: [hstdmin, hstdmax]})
    
    sextra_real = ColumnDataSource(data={xse[0]: [initmeanx,initmeanx], yse[0]: [hmeanmin, hmeanmax],
                                         xse[1]: [initstdx,initstdx], yse[1]: [hstdmin, hstdmax]})

dict_sources_1 = dict(zip(epochs, ['source1_{}'.format(x) for x in epochs]))
js_sources_1 = str(dict_sources_1).replace("'", "")
if not FLAGS.no_stats:
    dict_sources_2 = dict(zip(epochs, ['source2_{}'.format(x) for x in epochs]))
    js_sources_2 = str(dict_sources_2).replace("'", "")
    js_sources_e = sextraname

################################################################################################################
# Draw figures. The lines (```add_vline*```) start with dummy values so that they can be updated in the browser.
################################################################################################################

spectra_ylims = [-0.15,1.4]
#spectra_ylims = [-0.15,100.4]
spectra_yint = (spectra_ylims[0]-spectra_ylims[1])/15
spectraxlabel, spectraylabel = 'Wavelength [' + u'\u212b' + ']', 'Flux'
pfake = create_figure(WIDTH, HEIGHT, 'Generated QSO Spectra', ylims=spectra_ylims)
add_axis(pfake, xlabel=spectraxlabel, ylabel=spectraylabel,
         ybounds=spectra_ylims, xinterval=w_d*npoints/5, yinterval=spectra_yint)
for i in range(NSPECTRA):
    add_graph(pfake, xs1, ys1+str(i), s1[RENDERERS[0]], color=Category10[NSPECTRA][i])

preal = create_figure(WIDTH, HEIGHT, 'Real Spectra', ylims=spectra_ylims)
add_axis(preal, xlabel=spectraxlabel, ylabel=spectraylabel,
         ybounds=spectra_ylims, xinterval=w_d*npoints/5, yinterval=spectra_yint)
for i in range(NSPECTRA):
    add_graph(preal, xs1, ys1+str(i), sreal1, color=Category10[NSPECTRA][i])

if not FLAGS.no_stats:
    p_mean = create_figure(int(WIDTH2), int(HEIGHT2), 'Generated Means',
                           xlims=[cmeanmin,cmeanmax], ylims=[hmeanmin,hmeanmax],
                           xaxislog=LOGAXIS)
    add_axis(p_mean, xlabel=' ', ylabel='Counts',
             xinterval=(cmeanmax-cmeanmin)/10, yinterval=hmeanmax/5)
    add_bars(p_mean, xs2[0], ys2[0], wstr[0], s2[RENDERERS[1]], '#660044')
    line = add_vline(p_mean, xse[0], yse[0], 3., sextra, '#008000')
    add_legend(fig=p_mean, obj=line, label='Mean')
    
    p_mean_real = create_figure(int(WIDTH2), int(HEIGHT2), 'Real Means',
                                xlims=[cmeanminreal,40],#xlims=[cmeanminreal,cmeanmaxreal],
                                ylims=[hmeanminreal,hmeanmaxreal],
                                xaxislog=LOGAXIS)
    add_axis(p_mean_real, xlabel=' ', ylabel='Counts',
         xinterval=(cmeanmaxreal-cmeanminreal)/40, yinterval=hmeanmaxreal/5)
    add_bars(p_mean_real, xs2[0], ys2[0], wstr[0], sreal2, '#660044')
    line = add_vline_static(p_mean_real, x=sreal2mean_mean, y=[hmeanminreal,hmeanmaxreal], width=3., color='#008000')
    add_legend(fig=p_mean_real, obj=line, label='Mean')
    
    p_std = create_figure(int(WIDTH2), int(HEIGHT2), 'Generated Standard Deviations',
                          xlims=[cstdmin,cstdmax], ylims=[hstdmin,hstdmax],
                          xaxislog=LOGAXIS)
    add_axis(p_std, xlabel=' ', ylabel='Counts',
             xinterval=(cstdmax-cstdmin)/10, yinterval=hstdmaxreal/5)
    add_bars(p_std, xs2[1], ys2[1], wstr[1], s2[RENDERERS[1]], '#660044')
    line = add_vline(p_std, xse[1], yse[1], 3., sextra, '#008000')
    add_legend(fig=p_std, obj=line, label='Mean')
    
    p_std_real = create_figure(int(WIDTH2), int(HEIGHT2), 'Real Standard Deviations',
                               xlims=[cstdminreal,20],#xlims=[cstdminreal,cstdmaxreal],
                               ylims=[hstdminreal,hstdmaxreal],
                               xaxislog=LOGAXIS)
    add_axis(p_std_real, xlabel=' ', ylabel='Counts',
             xinterval=(cstdmaxreal-cstdminreal)/40, yinterval=hstdmaxreal/5)
    add_bars(p_std_real, xs2[1], ys2[1], wstr[1], sreal2, '#660044')
    line_mean = add_vline_static(p_std_real, x=sreal2mean_std, y=[hstdminreal,hstdmaxreal], width=3., color='#008000')
    add_legend(fig=p_std_real, obj=line_mean, label='Mean')

if FLAGS.no_stats:
    code = """
    var epoch = slider.value;
    var s1 = {js_sources_1};
    
    var new_s1 = s1[epoch].data;
    s1_update.data = new_s1;
    s1_update.change.emit();
    """.format(js_sources_1=js_sources_1)
else:
    code = """
    var epoch = slider.value;
    var s1 = {js_sources_1};
    var s2 = {js_sources_2};
    var se = {js_sources_e};
    
    var new_s1 = s1[epoch].data;
    s1_update.data = new_s1;
    s1_update.change.emit();

    var new_s2 = s2[epoch].data;
    s2_update.data = new_s2;
    s2_update.change.emit();

    var d = s2_update.data;

    var ncounts1_update = 0;
    var mean1_update = 0.;
    for (var i = 0; i<d['{means1}'].length; i++) {{
        mean1_update += d['{meanscounts1}'][i] * d['{means1}'][i];
        ncounts1_update += d['{meanscounts1}'][i];
    }}
    mean1_update /= ncounts1_update;
    se.data['mean1x'][0] = mean1_update;
    se.data['mean1x'][1] = mean1_update;
    se.data['mean1y'][0] = 0;
    se.data['mean1y'][1] = {vlinemax1};

    var ncounts2_update = 0;
    var mean2_update = 0.;
    for (var i = 0; i<d['{means2}'].length; i++) {{
        mean2_update += d['{meanscounts2}'][i] * d['{means2}'][i];
        ncounts2_update += d['{meanscounts2}'][i];
    }}
    mean2_update /= ncounts2_update;
    se.data['mean2x'][0] = mean2_update;
    se.data['mean2x'][1] = mean2_update;
    se.data['mean2y'][0] = 0;
    se.data['mean2y'][1] = {vlinemax2};
    se.change.emit();
    """.format(js_sources_1=js_sources_1, js_sources_2=js_sources_2,
               js_sources_e=js_sources_e,
               vlinemax1=hmeanmax, vlinemax2=hstdmax,
               wstr1=wstr[0],wstr2=wstr[1],
               means1='means', means2='stds',
               meanscounts1='meanscounts', meanscounts2='stdscounts')

callback_args = s1
if not FLAGS.no_stats:
    callback_args.update(s2)
    callback_args.update({sextraname: sextra})
callback = CustomJS(args=callback_args, code=code)

slider = Slider(start=0, end=NEPOCHS-1, value=0, step=1,
                title='Epoch', width=int(WIDTH/2))
slider.js_on_change('value_throttled', callback)

callback.args['s1_update'] = s1[RENDERERS[0]]
if not FLAGS.no_stats:
    callback.args['s2_update'] = s2[RENDERERS[1]]
callback.args['slider'] = slider

################################################################################################################
# Display
################################################################################################################
if FLAGS.no_stats:
    display = layout([pfake, preal, slider])
else:
    display = layout([[pfake, [p_mean, p_std]], [preal, [p_mean_real, p_std_real]], [slider]])
show(display)
#save(display) saves an independent html page
