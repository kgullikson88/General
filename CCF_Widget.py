"""
Bokeh widget for analyzing CCF data.
"""

import os
import time
from collections import OrderedDict

from bokeh.models import ColumnDataSource, Plot, HoverTool
from bokeh.plotting import figure, curdoc
from bokeh.properties import String, Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, VBox, VBoxForm, Select

from Analyze_CCF import CCF_Interface


CCF_FILE = '{}/School/Research/McDonaldData/Cross_correlations/CCF.hdf5'.format(os.environ['HOME'])
ADDMODE = 'simple'


class BokehApp(VBox):
    extra_generated_classes = [["BokehApp", "BokehApp", "VBox"]]
    jsmodel = "VBox"

    # data source
    #source = Instance(ColumnDataSource)
    T_run = Instance(ColumnDataSource)

    # layout boxes
    mainrow = Instance(HBox)
    ccfrow = Instance(HBox)

    # plots
    mainplot = Instance(Plot)
    ccf_plot = Instance(Plot)

    # inputs
    star = String(default=u"HIP 92855")
    date = String(default=u"20141015")
    star_select = Instance(Select)
    date_select = Instance(Select)
    # star_input_box = Instance(VBoxForm)
    #date_input_box = Instance(VBoxForm)
    input_box = Instance(VBoxForm)

    # ccf_interface = CCF_Interface(CCF_FILE)
    #T_run = pd.DataFrame()


    def __init__(self, *args, **kwargs):
        super(BokehApp, self).__init__(*args, **kwargs)


    @classmethod
    def create(cls, ccf_filename):
        """
        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        # create layout widgets
        obj = cls()
        cls._ccf_interface = CCF_Interface(ccf_filename)
        obj.mainrow = HBox()
        obj.ccfrow = HBox()
        #obj.star_input_box = VBoxForm()
        #obj.date_input_box = VBoxForm()
        obj.input_box = VBoxForm()

        # create input widgets
        obj.set_defaults()
        obj.make_star_input()
        obj.make_date_input()

        # outputs
        #obj.pretext = PreText(text="", width=500)
        obj.make_source()
        obj.make_plots()

        # layout
        obj.set_children()
        return obj

    def set_defaults(self):
        starnames = self._ccf_interface.list_stars()
        dates = self._ccf_interface.list_dates(starnames[0])
        self.star = starnames[0]
        self.date = dates[0]


    def make_star_input(self):
        starnames = self._ccf_interface.list_stars()
        self.star_select = Select(
            name='Star',
            value=starnames[0],
            options=starnames,
        )

    def make_date_input(self):
        dates = self._ccf_interface.list_dates(self.star)
        self.date = dates[0]
        # TODO: Figure out how to make a new date list more than once!
        if isinstance(self.date_select, Select):
            self.date_select.update(value=dates[0], options=dates)
        else:
            self.date_select = Select.create(
                name='Date',
                value=dates[0],
                options=dates,
            )

    def make_source(self):
        self._source = self.df
        self.T_run = ColumnDataSource(self._ccf_interface.get_temperature_run(df=self._source))

    def plot_ccf(self, T, x_range=None):
        # First, find the best values where temperature = T
        T_run = self.T_run.to_df()
        good = T_run.loc[T_run['T'] == T]
        pars = {'vsini': good.vsini.item(), '[Fe/H]': good['[Fe/H]'].item(), 'T': T,
                'logg': good.logg.item(), 'addmode': 'simple'}
        t1 = time.time()
        ccf = self._ccf_interface.get_ccf(pars, df=self.df)
        t2 = time.time()
        print('Time to retrieve ccf with requested parameters: {}'.format(t2 - t1))

        # Now, plot
        p = figure(
            title='{} K'.format(T),
            x_range=x_range,
            plot_width=1000, plot_height=500,
            title_text_font_size="10pt",
            tools="pan,wheel_zoom,box_select,reset,save"
        )
        p.line(ccf.velocity, ccf.CCF, size=2,
               xlabel='Velocity', ylabel='CCF')
        return p

    def make_plots(self):
        star = self.star
        date = self.date
        T_run = self.T_run.to_df()
        p = figure(
            title="{} / {}".format(star, date),
            plot_width=1000, plot_height=400,
            tools="pan,wheel_zoom,tap,hover,reset",
            title_text_font_size="10pt",
        )
        p.circle("T", "ccf_value",
                 size=8,
                 nonselection_alpha=1.0,
                 source=self.T_run
        )
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("Temperature", "@T"),
            ("vsini", "@vsini"),
            ("[Fe/H]", "@metal"),
            ("log(g)", "@logg"),
            ("Radial Velocity (km/s)", "@rv"),
            ("ccf peak height", "@ccf_value"),
        ])
        self.mainplot = p

        # T = T_run['T'].max()
        T = T_run.loc[T_run['ccf_value'] == T_run['ccf_value'].max()]['T'].item()
        #self._ccf_plot = self.plot_ccf(T)
        self.ccf_plot = self.plot_ccf(T)


    def set_children(self):
        self.children = [self.mainrow, self.ccfrow]
        # self.mainrow.children = [self.input_box, self._plot]
        self.mainrow.children = [self.input_box, self.mainplot]
        self.input_box.children = [self.star_select, self.date_select]
        #self.ccfrow.children = [self._ccf_plot]
        self.ccfrow.children = [self.ccf_plot]

    def star_change(self, obj, attrname, old, new):
        print 'Star change!'
        self.star = new
        self.make_date_input()
        self.make_source()
        self.make_plots()
        self.set_children()
        curdoc().add(self)

    def date_change(self, obj, attrname, old, new):
        print 'Date change!'
        self.date = new
        self.make_source()
        self.make_plots()
        self.set_children()
        curdoc().add(self)

    def setup_events(self):
        super(BokehApp, self).setup_events()
        #if self.source:
        #    self.source.on_change('selected', self, 'selection_change')
        if self.T_run:
            self.T_run.on_change('selected', self, 'Trun_change')
        if self.star_select:
            self.star_select.on_change('value', self, 'star_change')
        if self.date_select:
            self.date_select.on_change('value', self, 'date_change')


    def Trun_change(self, obj, attrname, old, new):
        t1 = time.time()
        T = self.T_run.to_df().ix[new]['T'].item()
        t2 = time.time()
        print('Time to convert T_run to dataframe: {}'.format(t2 - t1))
        t1 = time.time()
        self.ccf_plot = self.plot_ccf(T)
        t2 = time.time()
        print('Time to make ccf plot: {}'.format(t2 - t1))
        self.set_children()
        curdoc().add(self)


    @property
    def df(self):
        return self._ccf_interface._compile_data(self.star, self.date, addmode=ADDMODE)


# The following code adds a "/bokeh/stocks/" url to the bokeh-server. This URL
# will render this StockApp. If you don't want serve this applet from a Bokeh
# server (for instance if you are embedding in a separate Flask application),
# then just remove this block of code.
@bokeh_app.route("/bokeh/ccf/")
@object_page("ccf")
def make_ccf_app():
    app = BokehApp.create(CCF_FILE)
    return app