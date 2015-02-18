"""
Bokeh widget for analyzing CCF data.
"""

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.properties import String, Instance
from bokeh.models.widgets import HBox, VBox, VBoxForm, PreText, Select

from Analyze_CCF import CCF_Interface


class BokehApp(VBox):
    # data source
    source = Instance(ColumnDataSource)

    # layout boxes
    mainrow = Instance(HBox)
    ccfrow = Instance(HBox)

    # inputs
    star = String(default="HIP 92855")
    date = String(default="20141015")
    star_select = Instance(Select)
    date_select = Instance(Select)
    # star_input_box = Instance(VBoxForm)
    #date_input_box = Instance(VBoxForm)
    input_box = Instance(VBoxForm)

    def __init__(self, ccf_filename, *args, **kwargs):
        super(BokehApp, self).__init__(*args, **kwargs)
        self.ccf_interface = CCF_Interface(ccf_filename)

    @classmethod
    def create(cls):
        """
        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        # create layout widgets
        obj = cls()
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
        obj.pretext = PreText(text="", width=500)
        obj.make_source()
        obj.make_plots()

        # layout
        obj.set_children()
        return obj

    def set_defaults(self):
        starnames = self.ccf_interface.list_stars()
        dates = self.ccf_interface.list_dates(starnames[0])
        self.star = String(default=starnames[0])
        self.date = String(default=dates[0])


    def make_star_input(self):
        starnames = self.ccf_interface.list_stars()
        self.star_select = Select(
            name='Star',
            value=starnames[0],
            options=starnames,
        )

    def make_date_input(self):
        dates = self.ccf_interface.list_dates(self.star)
        self.ticker2_select = Select(
            name='Date',
            value=dates[0],
            options=dates,
        )

    def make_source(self):
        self.source = ColumnDataSource(data=self.df)

    def plot_ccf(self, T, x_range=None):
        # First, find the best values where temperature = T
        good = self.T_run.loc[self.T_run.T == T]
        pars = {'vsini': good.vsini, '[Fe/H]': good['[Fe/H]'], 'T': T, 'logg': good.logg, 'addmode': 'simple'}
        ccf = self.ccf_interface.get_ccf(pars, df=self.source.to_df())

        # Now, plot
        p = figure(
            title='{} K'.format(T),
            x_range=x_range,
            plot_width=1000, plot_height=400,
            title_text_font_size="10pt",
            tools="pan,wheel_zoom,box_select,reset"
        )
        p.line(
            'velocity', 'CCF',
            size=2,
            source=ccf,
        )
        return p

    def make_plots(self):
        star = self.star
        date = self.date
        self.T_run = self.ccf_interface.get_temperature_run(df=self.source.to_df())
        p = figure(
            title="{} / {}".format(star, date),
            plot_width=1000, plot_height=400,
            tools="pan,wheel_zoom,tap_select,reset",
            title_text_font_size="10pt",
        )
        p.circle("T", "ccf_value",
                 size=2,
                 nonselection_alpha=0.02,
                 source=ColumnDataSource(self.T_run)
        )
        self.plot = p

        T = self.T_run.T.max()
        self.ccf_plot = self.plot_ccf(T)


    def set_children(self):
        self.children = [self.mainrow, self.ccfrow, self.ccfplot]
        self.mainrow.children = [self.input_box, self.plot]
        self.input_box.children = [self.star_select, self.date_select]
        self.ccfrow.children = [self.ccf_plot]

    def star_change(self, obj, attrname, old, new):
        self.star = new
        self.make_date_input()
        self.make_source()
        self.make_plots()
        self.set_children()
        curdoc().add(self)

    def date_change(self, obj, attrname, old, new):
        self.date = new
        self.make_source()
        self.make_plots()
        self.set_children()
        curdoc().add(self)

    def setup_events(self):
        super(BokehApp, self).setup_events()
        if self.source:
            self.source.on_change('selected', self, 'selection_change')
        if self.star_select:
            self.star_select.on_change('value', self, 'star_change')
        if self.date_select:
            self.date_select.on_change('value', self, 'date_change')

    @property
    def selected_T(self):
        pandas_df = self.df
        selected = self.T_run.selected
        if selected:
            pandas_df = pandas_df.iloc[selected, :]

        #TODO: make this work to give either the maximum T if there is no selection, or the selected temperature!
        return pandas_df

    def selection_change(self, obj, attrname, old, new):
        self.make_stats()
        self.hist_plots()
        self.set_children()
        curdoc().add(self)


    @property
    def df(self):
        return self.ccf_interface._compile_data(self.star, self.date)