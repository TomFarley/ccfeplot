#!/usr/bin/env python
"""
Author: Tom Farley, Sudeep Mandal
Forked from ccfeplot by Sudeep Mandal: https://github.com/HamsterHuey/ccfeplot
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import inspect
import warnings
from collections import OrderedDict

string23 = str if sys.version_info > (3, 0) else basestring

if not plt.isinteractive():
    print("\nMatplotlib interactive mode is currently OFF. It is "
          "recommended to use a suitable matplotlib backend and turn it "
          "on by calling matplotlib.pyplot.ion()\n")

class CcfePlot(object):
    """
    Class that implements thin matplotlib wrapper for easy, reusable plotting
    """

    _figure_defaults = {'figsize': None,
                        'dpi': mpl.rcParams['figure.dpi'],
                        'sharex': False,
                        'sharey': False
                        }

    _legend_defaults = OrderedDict(loc='best',
                                   fancybox=True,
                                   framealpha=0.8,
                                   numpoints=1,
                                   ncol=1
                                   )

    _plot_defaults = OrderedDict(linestyle='-'
                                 )

    _other_defaults = OrderedDict(showlegend=True
                                 )

    # Dictionary of plot parameter aliases
    _alias_dict = {'lw': 'linewidth', 'ls': 'linestyle',
                       'mfc': 'markerfacecolor', 'mew': 'markeredgewidth',
                       'mec': 'markeredgecolor', 'ms': 'markersize',
                       'mev': 'markevery', 'c': 'color', 'fs': 'fontsize'}

    # List of all named plot parameters passable to plot method
    _line_attributes = ['label', 'linewidth', 'linestyle', 'marker',
                             'markerfacecolor', 'markeredgewidth', 'markersize',
                             'markeredgecolor', 'markevery', 'alpha', 'color']

    _legend_attributes = ['fancybox', 'loc', 'framealpha', 'numpoints',
                               'ncol', 'markerscale', 'mode', 'bbox_to_anchor']

    # Parameters that should only be passed to the plot once, then reset
    _uniqueparams = ['color', 'label', 'marker', 'linestyle',
                          'colorcycle']

    # Mapping between plot parameter and corresponding axes function to call
    _ax_methods = {'xlabel': 'set_xlabel',
                    'ylabel': 'set_ylabel',
                    'xlim': 'set_xlim',
                    'ylim': 'set_ylim',
                    'title': 'set_title',
                    'colorcycle': 'set_color_cycle',
                    'grid': 'grid',
                    'xscale': 'set_xscale',
                    'yscale': 'set_yscale'}

    def __init__(self, nrows=1, ncolumns=1, sharex=False, sharey=False, *args, **kwargs):
        """
        Arguments
        =========
        *args : Support for plot(y), plot(x, y), plot(x, y, 'b-o'). x, y and
                format string are passed through for plotting
        
        **kwargs: All kwargs are optional
          Plot Parameters:
          ----------------
            fig : figure instance for drawing plots
            ax : axes instance for drawing plots (If user wants to supply axes,
                 figure externally, both ax and fig must be supplied together)
            figSize : tuple of integers ~ width & height in inches
            dpi : dots per inch setting for figure
            label : Label for line plot as determined by *args, string
            color / c : Color of line plot, overrides format string in *args if
                        supplied. Accepts any valid matplotlib color
            linewidth / lw : Plot linewidth
            linestyle / ls : Plot linestyle ['-','--','-.',':','None',' ','']
            marker : '+', 'o', '*', 's', 'D', ',', '.', '<', '>', '^', '1', '2'
            markerfacecolor / mfc : Face color of marker
            markeredgewidth / mew :
            markeredgecolor / mec : 
            markersize / ms : Size of markers
            markevery / mev : Mark every Nth marker 
                              [None|integer|(startind, stride)]
            alpha : Opacity of line plot (0 - 1.0), default = 1.0
            title : Plot title, string
            xlabel : X-axis label, string
            ylabel : Y-axis label, string
            xlim : X-axis limits - tuple. eg: xlim=(0,10). Set to None for auto
            ylim : Y-axis limits - tuple. eg: ylim=(0,10). Set to None for auto
            xscale : Set x axis scale ['linear'|'log'|'symlog']
            yscale : Set y axis scale ['linear'|'log'|'symlog']
                Only supports basic xscale/yscale functionality. Use 
                get_axes().set_xscale() if further customization is required
            grid : Display axes grid. ['on'|'off']. See grid() for more options
            colorcycle / cs: Set plot colorcycle to list of valid matplotlib
                             colors
            fontsize : Global fontsize for all plots

          Legend Parameters:
          ------------------
            showlegend : set to True to display legend
            fancybox : True by default. Enables rounded corners for legend box
            framealpha : Legend box opacity (0 - 1.0), default = 1.0
            loc : Location of legend box in plot, default = 'best'
            numpoints : number of markers in legend, default = 1.0
            ncol : number of columns for legend. default is 1
            markerscale : The relative size of legend markers vs. original. 
                          If None, use rc settings.
            mode : if mode is “expand”, the legend will be horizontally 
                   expanded to fill the axes area (or bbox_to_anchor)
            bbox_to_anchor : The bbox that the legend will be anchored. Tuple of
                             2 or 4 floats
        """
        self._1d_artists = []
        self._2d_artists = []
        self._3d_artists = []
        self._legends = OrderedDict()
        self._figure_settings = CcfePlot._figure_defaults.copy()
        self._legend_settings = CcfePlot._legend_defaults.copy()
        self._plot_settings_default = CcfePlot._plot_defaults.copy()
        self._other_defaults = CcfePlot._other_defaults.copy()

        for arg in args:
            raise NotImplementedError

        for key, value in kwargs.items():
            if hasattr(plt.subplots, key):
                self._figure_settings[key] = value
            elif hasattr(plt.legend, key):
                self._figure_settings[key] = value
            elif hasattr(plt.plot, key):
                self._plot_settings_default[key] = value
            elif key in self._other_defaults:
                self._other_defaults[key] = value


        self._fig, self._axes = plt.subplots(nrows, ncolumns, squeeze=False, **self._figure_settings)

        self._figure_settings['subplots'] = (nrows, ncolumns)

        self._colorcycle = []

        self.set_default_axis([0, 0])  # Default axis for plot calls

        # self.kwargs = CcfePlot._figure_defaults.copy()  # Prevent mutating dictionary
        # self.args = []
        # self._line_list = []  # List of all Line2D items that are plotted
        # self.add_line(*args, **kwargs)

    def get_axis(self, ax):
        if isinstance(ax, mpl.axes.Axes):
            return ax
        elif isinstance(ax, (list, tuple)):
            return self._axes[ax[0]][ax[1]]
        elif isinstance(ax, int):
            raise NotImplementedError
        else:
            raise ValueError

    def get_axis_index(self, ax):
        for i in np.arange(self._figure_settings['subplots'][0]):
            for j in np.arange(self._figure_settings['subplots'][1]):
                if ax == self._axes[i,j]:
                    return ax
        return None

    def set_default_axis(self, indices):
        """Set default axis for plot calls

        :indices: Tuple of axes indices in format (row, column)
        """
        if isinstance(indices, int):
            raise NotImplementedError
        ax = self.get_axis(indices)
        self.default_ax = ax
        plt.sca(ax)
        return ax

    def _update_legend(self, indices):
        ax = self.get_axis(indices)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Silence warning about no existing labels
            leg = ax.legend(**self._legend_settings) if self._other_defaults['showlegend'] else None
        if leg is not None:
            leg.draggable(state=True)
        self._legends[indices] = leg
        return leg

    def update_legends(self):
        """Update legends on all axes"""
        for i in np.arange(self._figure_settings['subplots'][0]):
            for j in np.arange(self._figure_settings['subplots'][1]):
                self._update_legend((i,j))
        return self._legends

    def plot(self, x, *args, ax='default', **kwargs):
        """
        Add plot using supplied parameters and existing instance parameters
        
        Creates new Figure and Axes object if 'fig' and 'ax' parameters not
        supplied. Stores references to all Line2D objects plotted in 
        self.line_list.
        Previously called add_plot

        Arguments
        =========
            *args : Supports format plot(y), plot(x, y), plot(x, y, 'b-'). x, y 
                    and format string are passed through for plotting
            **kwargs : Plot parameters. Refer to __init__ docstring for details
        """
        if ax == 'default':
            ax = self.default_ax
        elif isinstance(ax, (tuple, list, int)):
            ax = self.set_default_axis(ax)
        ax_index = self.get_axis_index(ax)

        ax.ticklabel_format(useOffset=False)  # Prevent offset notation in plots

        plot_settings = self._plot_settings_default.copy()

        for key, value in kwargs.items():
            if hasattr(plt.plot, key):
                plot_settings[key] = value

        ax.plot(x, *args, **plot_settings)

        self._update_legend(ax)

        if plt.isinteractive(): # Only redraw canvas in interactive mode
            self.redraw()
          
    def update_plot(self, **kwargs):
        """"Update plot parameters (keyword arguments) and replot figure
        
        Usage:
            a = EasyPlot([1,2,3], [2,4,8], 'r-o', label='label 1')
            # Update title and xlabel string and redraw plot
            a.update_plot(title='Title', xlabel='xlabel')
        """
        self.plot(**kwargs)
        
    def new_plot(self, *args, **kwargs):
        """
        Plot new plot using EasyPlot object and default plot parameters
        
        Pass a named argument reset=True if all plotting parameters should
        be reset to original defaults
        """
        reset = kwargs['reset'] if 'reset' in kwargs else False
        self._reset(reset=reset)
        if self._colorcycle:
            self.kwargs['colorcycle'] = self._colorcycle
        self.plot(*args, **kwargs)
    
    def add_lines(self, x, y, mode='dict', **kwargs):
        """
        Plot multiple plots by iterating through x, y and parameter lists

        Previously called iter_plot

        Arguments:
        ==========
          x : x values. 1D List/Array, Dictionary or Numpy 2D Array
          y : y values. Dictionary or 2D Python array (List of Lists where each
              sub-list is one set of y-data) or Numpy 2D Array
          mode : y, labels and other parameters should either be a Dictionary
                 or a 2D Numpy array/2D List where each row corresponds to a 
                 single plot ['dict'|'array']
          **kwargs : Plot params as defined in __init__ documentation.
             Params can either be:
               scalars (same value applied to all plots),
               dictionaries (mode='dict', key[val] value applies to each plot)
               1D Lists/Numpy Arrays (mode='array', param[index] applies to each
               plot)
        """
        if mode.lower() == 'dict':
            for key in y:
                loop_kwargs={}
                for kwarg in kwargs:
                    try: # Check if parameter is a dictionary
                        loop_kwargs[kwarg] = kwargs[kwarg][key]
                    except:
                        loop_kwargs[kwarg] = kwargs[kwarg]
                try:
                    x_loop = x[key]
                except:
                    x_loop = x
                self.plot(x_loop, y[key], **loop_kwargs)

        elif mode.lower() == 'array':
            for ind in range(len(y)):
                loop_kwargs={}
                for kwarg in kwargs:
                    # Do not iterate through tuple/string plot parameters
                    if isinstance(kwargs[kwarg], (string23, tuple)):
                        loop_kwargs[kwarg] = kwargs[kwarg]
                    else:
                        try: # Check if parameter is a 1-D List/Array
                            loop_kwargs[kwarg] = kwargs[kwarg][ind]
                        except:
                            loop_kwargs[kwarg] = kwargs[kwarg]
                try:
                    x_loop = x[ind][:]
                except:
                    x_loop = x
                self.plot(x_loop, y[ind], **loop_kwargs)
        else:
            print('Error! Incorrect mode specification. Ignoring method call')

    def autoscale(self, enable=True, axis='both', tight=None):
        """Autoscale the axis view to the data (toggle).
        
        Convenience method for simple axis view autoscaling. It turns 
        autoscaling on or off, and then, if autoscaling for either axis is on,
        it performs the autoscaling on the specified axis or axes.
        
        Arguments
        =========
        enable: [True | False | None]
        axis: ['x' | 'y' | 'both']
        tight: [True | False | None]
        """
        ax = self.get_axes()
        ax.autoscale(enable=enable, axis=axis, tight=tight)
        # Reset xlim and ylim parameters to None if previously set to some value
        if 'xlim' in self.kwargs and (axis=='x' or axis=='both'):
            self.kwargs.pop('xlim') 
        if 'ylim' in self.kwargs and (axis=='y' or axis=='both'):
            self.kwargs.pop('ylim')
        self.redraw()

    def grid(self, **kwargs):
        """Turn axes grid on or off

        Call signature: grid(self, b=None, which='major', axis='both', **kwargs)
        **kwargs are passed to linespec of grid lines (eg: linewidth=2)
        """
        self.get_axes().grid(**kwargs)
        self.redraw()

    def get_figure(self):
        """Returns figure instance of current plot"""
        return self.kwargs['fig']
        
    def get_axes(self):
        """Returns axes instance for current plot"""
        return self.kwargs['ax']
        
    def redraw(self):
        """
        Redraw plot. Use after custom user modifications of axes & fig objects
        """
        if plt.isinteractive():
            fig = self._fig
            #Redraw figure if it was previously closed prior to updating it
            if not plt.fignum_exists(fig.number):
                fig.show()
            fig.canvas.draw()
        else:
            print('redraw() is unsupported in non-interactive plotting mode!')
    
    def set_fontsize(self, font_size):
        """ Updates global font size for all plot elements"""
        mpl.rcParams['font.size'] = font_size
        self.redraw()
        #TODO: Implement individual font size setting
#        params = {'font.family': 'serif',
#          'font.size': 16,
#          'axes.labelsize': 18,
#          'text.fontsize': 18,
#          'legend.fontsize': 18,
#          'xtick.labelsize': 18,
#          'ytick.labelsize': 18,
#          'text.usetex': True}
#        mpl.rcParams.update(params)
    
#    def set_font(self, family=None, weight=None, size=None):
#        """ Updates global font properties for all plot elements
#        
#        TODO: Font family and weight don't update dynamically"""
#        if family is None:
#            family = mpl.rcParams['font.family']
#        if weight is None:
#            weight = mpl.rcParams['font.weight']
#        if size is None:
#            size = mpl.rcParams['font.size']
#        mpl.rc('font', family=family, weight=weight, size=size)
#        self.redraw()
        
    def _delete_uniqueparams(self):
        """Delete plot parameters that are unique per plot
        
        Prevents unique parameters (eg: label) carrying over to future plots"""
        # Store colorcycle list prior to deleting from this instance
        if 'colorcycle' in self.kwargs:
            self._colorcycle = self.kwargs['colorcycle']

        for param in self._uniqueparams:
            self.kwargs.pop(param, None)
        
    def _update(self, *args, **kwargs):
        """Update instance variables args and kwargs with supplied values """
        if args:
            self.args = args # Args to be directly passed to plot command
            self.isnewargs = True
        else:
            self.isnewargs = False

        # Update self.kwargs with full parameter name of aliased plot parameter
        for alias in self.alias_dict:
            if alias in kwargs:
                self.kwargs[self.alias_dict[alias]] = kwargs.pop(alias)
            
        # Update kwargs dictionary
        for key in kwargs:
            self.kwargs[key] = kwargs[key]
           
    def _reset(self, reset=False):
        """Reset instance variables in preparation for new plots
        reset: True if current instance defaults for plotting parameters should
               be reset to Class defaults"""
        self.args = []
        self._line_list = []
        self.kwargs['fig'] = None
        self.kwargs['ax'] = None
        if reset:
            self.kwargs = CcfePlot._figure_defaults.copy()

def kws_filter(kwargs, func):
    kws = {k: v for k, v in kwargs.items() if k in inspect.getargspec(func)[0]}
    return kws