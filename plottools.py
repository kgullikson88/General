import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_style('ticks')

def add_spt_axis(axis, spt_values=('M5', 'M0', 'K5', 'K0', 'G5', 'G0')):
    import SpectralTypeRelations
    MS = SpectralTypeRelations.MainSequence()
    # Find the temperatures at each spectral type
    temp_values = MS.Interpolate('Temperature', spt_values)
    
    # make the axis
    top = axis.twiny()
    
    # Set the full range to be the same as the data axis
    xlim = axis.get_xlim()
    top.set_xlim(xlim)
    
    # Set the ticks at the temperatures corresponding to the right spectral type
    top.set_xticks(temp_values)
    top.set_xticklabels(spt_values)
    top.set_xlabel('Spectral Type')
    return top