import numpy as np
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go

# Filter requirements.
T = 5.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

t = np.linspace(0, T, n)
# sin wave
sig = np.sin(1.2*2*np.pi*t)
# Lets add some noise
noise = 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
data = sig + noise

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq  # Nyquist Frequency
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=data,
    line=dict(shape='spline'),
    name='signal with noise'
))
fig.add_trace(go.Scatter(
    y=y,
    line=dict(shape='spline'),
    name='filtered signal'
))
fig.show()