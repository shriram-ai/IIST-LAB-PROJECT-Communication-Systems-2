from bdb import effective
from cmath import pi
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp


def bitgen(N):
    random_numbers = np.random.rand(N, 1)
    bitstr = np.zeros([N, 1])
    bitstr[np.where(random_numbers > 0.5)] = 1
    return bitstr


def bits_to_impulse(bitstr, Tb, Fs, amplitudes=[-1, 1]):
    assert 1 / Fs < Tb, "Pulse duration should be less than sampling time"

    num_bits = len(bitstr)
    num_samples_per_bit = int(Tb * Fs)
    t = np.arange(0, Tb * num_bits, 1 / Fs)
    num_samples = num_samples_per_bit * num_bits

    impulse_train = np.zeros([num_samples, 1])
    n = np.int0(np.arange(0, num_samples, num_samples_per_bit))
    k = np.int0(n / num_samples_per_bit)
    impulse_train[n] = np.array(
        [amplitudes[0] * (1 - bitstr[k]) + amplitudes[1] * (bitstr[k])]
    )
    return impulse_train


def pam_code(bitstr, Tb, Fs, amplitudes=[-1, 1]):
    impulse_train = bits_to_impulse(bitstr, Tb, Fs, amplitudes)
    impulse_train = np.ndarray.flatten(impulse_train)
    pulse_shape = np.ones([int(Tb * Fs), 1])
    pulse_shape = np.ndarray.flatten(pulse_shape)
    line_code = np.convolve(impulse_train, pulse_shape)
    return line_code


def sinc_code(bitstr, Tb, Fs, amplitudes=[-1, 1]):
    impulse_train = bits_to_impulse(bitstr, Tb, Fs, amplitudes)
    impulse_train = np.ndarray.flatten(impulse_train)
    t = np.arange(-2.5 * Tb, 2.5 * Tb + 1 / Fs, 1 / Fs)
    pulse_shape = np.pi * np.sinc(t / Tb)
    pulse_shape = np.ndarray.flatten(pulse_shape)
    line_code = np.convolve(impulse_train, pulse_shape)
    return line_code


def raised_cos_code(bitstr, Tb, Fs, alpha, amplitudes=[-1, 1]):
    impulse_train = bits_to_impulse(bitstr, Tb, Fs)
    impulse_train = np.ndarray.flatten(impulse_train)
    t = np.arange(-2.5 * Tb, 2.5 * Tb + 1 / Fs, 1 / Fs)
    assert np.count_nonzero((1 - (2 * alpha * t / Tb) ** 2) == 0) < len(
        t
    ), "ZeroDivisionError: Try changing Tb "
    pulse_shape = (
        np.pi
        * np.sinc(t / Tb)
        * np.cos(np.pi * alpha * t / Tb)
        / (1 - (2 * alpha * t / Tb) ** 2)
    )
    pulse_shape = np.ndarray.flatten(pulse_shape)
    line_code = np.convolve(impulse_train, pulse_shape)
    return line_code


def eye_diagram(signal, Tb, Fs, interval, title_text, ax=None):
    interval_length = interval[1] - interval[0]
    assert (
        interval_length > 1 / Fs
    ), "Sampling time must be smaller than length of time interval"

    if ax is None:
        ax = plt.gca()

    # Number of samples in the input sequence
    num_samples = len(signal)

    # Number of samples in each interval
    num_samples_interval = int(interval_length * Fs)

    for i in range(1, int(num_samples / num_samples_interval)):
        ax.plot(
            np.arange(interval[0], interval[1] + 1 / Fs, 1 / Fs),
            signal[(i - 1) * num_samples_interval : i * num_samples_interval + 1],
            "r-",
        )

    ax.set_title(title_text)
    ax.set_xlabel("Time (in seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(interval)
    return ax


def PSD(signal, Fs, total_time, title_text, to_plot, ax=None):

    dft = np.fft.fftshift(np.fft.fft(signal))
    # Converting Discrete Fourier Transform to Continuous
    # ctft = dft/Fs

    energy = (np.abs(dft) / Fs) ** 2
    psd = energy / (Fs * total_time)

    num_samples = len(signal)

    freq_range = np.arange(-num_samples / 2, num_samples / 2) * Fs / num_samples

    if to_plot:
        if ax is None:
            ax = plt.gca()
        ax.plot(freq_range, psd, "k")
        ax.set_title(title_text)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
    return ax


def channel(signal, dc_gain, f_pass, f_stop, Fs, filter_length, to_plot, ax=None):

    assert f_pass < Fs / 2 and f_stop < Fs / 2, "Increase Sampling frequency"

    # Filter impulse response
    impulse_response = dc_gain * sp.firwin(
        filter_length, f_pass, f_stop - f_pass, pass_zero="lowpass", scale=True, fs=Fs
    )

    if to_plot:
        if ax is None:
            ax = plt.gca()
        freq_array, freq_response = sp.freqz(impulse_response, 1, len(impulse_response))
        ax.plot(freq_array / math.pi, np.abs(freq_response), "m")
        ax.set_title("Magnitude Response of Channel")
        ax.set_xlabel("Normalised frequency (in $\pi$ radians/sample)")
        ax.set_ylabel("Magnitude")

    # Filtering signal with channel
    channel_output = np.convolve(signal, impulse_response)

    return ax, channel_output


def anychannel(signal, gain_array, freq_array, Fs, filter_length, to_plot, ax=None):

    assert (
        freq_array[-1] < Fs
    ), "Sampling frequency must be more than twice as large as last frequency component in sorted freq_array"

    freq_array = np.concatenate((freq_array, [Fs / 2]), axis=0)
    gain_array = np.concatenate((gain_array, [0]), axis=0)

    # Channel impulse response
    impulse_response = sp.firwin2(filter_length, freq_array, gain_array, 1025, fs=Fs)

    if to_plot:
        if ax is None:
            ax = plt.gca()
        freq, freq_response = sp.freqz(impulse_response, 1, len(impulse_response))
        ax.plot(freq / math.pi, np.abs(freq_response), "m")
        ax.set_title("Magnitude Response of Channel")
        ax.set_xlabel("Normalised frequency (in $\pi$ radians/sample)")
        ax.set_ylabel("Magnitude")

    # Filtering signal with channel
    channel_output = np.convolve(signal, impulse_response)

    return ax, impulse_response, channel_output


def zero_forcing_eq(signal, impulse_response, Fs, to_plot, ax):
    # Computing frequency response of the channel
    freq_array, freq_response = sp.freqz(
        impulse_response, 1, len(impulse_response), include_nyquist=True
    )

    # Channel Inversion
    inv_freq_response = np.zeros(len(freq_array))
    mag_response = np.abs(freq_response)
    for i in range(len(inv_freq_response)):
        if mag_response[i] < 10 ** (-2):
            inv_freq_response[i] = 0
        else:
            inv_freq_response[i] = 1 / mag_response[i]

    # Inverted Channel impulse response
    inv_impulse_response = sp.firwin2(
        len(impulse_response),
        freq_array * Fs / (2 * math.pi),
        inv_freq_response,
        1025,
        fs=Fs,
    )

    if to_plot:
        if ax is None:
            ax = plt.gca()
        ax.plot(freq_array / math.pi, np.abs(inv_freq_response), "m")
        ax.set_title("Magnitude Response of Inverted Channel")
        ax.set_xlabel("Normalised frequency (in $\pi$ radians/sample)")
        ax.set_ylabel("Magnitude")

    # Filtering through inverted channel
    equaliser_output = np.convolve(signal, inv_impulse_response)

    return ax, inv_impulse_response, equaliser_output


def convolution_matrix(f, g):
    # Generates convolution matrix of f of order len(f) x len(g) 
    # which when matrix multiplied with g produces 'f * g' where '*' is convolution
    conv_matrix = np.zeros((len(g), len(f)))
    for i in range(len(g)):
        for j in range(len(f)):
            conv_matrix[i, j] = f[(i + j) % len(f)]

    return conv_matrix


def tapped_delay_eq(signal, order, Tb, Fs, pulse_shape, channel_impulse_response, to_plot, ax):
    filter_length = np.uint16(2 * order + 1)

    # Effective Channel = Channel impulse response convolved with pulse shape
    eff_channel = np.convolve(channel_impulse_response, pulse_shape)

    # Sampling at middle of bit times
    sample_instances = np.arange(Tb*Fs/2, len(eff_channel), Tb * Fs, dtype="uint16")

    # Obtaining samples of Effective Channel
    eff_channel_samples = eff_channel[sample_instances]

    # Ideal Effective pulse shape for 0 ISI
    g = np.zeros(filter_length)
    g[np.uint16(filter_length / 2)] = 1

    # Computing the convolution matrix of Effective channel
    conv_matrix = convolution_matrix(eff_channel_samples, g)
    tapped_filter_weights = np.matmul(np.linalg.pinv(conv_matrix), g)
    

    # Upsampling the obtained weight matrix with constraints 
    tapped_delay_filter = np.zeros(np.uint16(len(tapped_filter_weights)*Tb*Fs))
    tapped_delay_filter[np.uint16(sample_instances)] = tapped_filter_weights

    # Evaluating the actual 'g' obtained
    g = np.convolve(tapped_delay_filter, eff_channel)
    g = g[(np.arange(0, len(g), Tb * Fs, dtype="uint16"))]

    if to_plot:
        if ax is None:
            ax = plt.gca()
        ax.plot(tapped_delay_filter)
        ax.set_title("Impulse Response of Tapped Delay Equaliser")
        ax.set_xlabel("Index")
        ax.set_ylabel("Amplitude")

    # Filtering the input signal through the tapped delay equaliser (filter)
    eq_output = np.convolve(signal, tapped_delay_filter)

    return ax, tapped_delay_filter, eq_output, g

