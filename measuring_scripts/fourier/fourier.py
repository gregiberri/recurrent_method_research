import matplotlib.pyplot as plt

import numpy as np

z = 0.03
N_div = 7
N = int(1.2 * N_div / z)
delta_t0 = 1 / 48000

# sin function
frequency = 600
A = 5
fi = np.pi / 4
times = np.arange(0, 100, delta_t0)
y_t = A * np.sin(2 * np.pi * frequency * times + fi)

# plot input
# plt.plot(y_t)
# plt.show()

# noise
noise_amplitude = 0
noise = noise_amplitude * np.random.random(len(times))
y_t = y_t + noise

# fourier transform for different omegas

frequencies = []
freq = 100
fs = []
while freq < 1000:
    tau = 1 / (delta_t0 * N_div * freq)
    f_cos = 1 / N * np.sum([y_t[-int(n * tau)] * np.cos(2 * np.pi * n / N_div * freq) for n in range(N)])
    f_sin = 1 / N * np.sum([y_t[-int(n * tau)] * np.sin(2 * np.pi * n / N_div * freq) for n in range(N)])
    f = np.sqrt(f_cos ** 2 + f_sin ** 2)
    frequencies.append(freq)
    fs.append(f)
    freq = (1 + z) * freq

max_index = np.argmax(fs)
max_frequency = frequencies[max_index]
max_value = fs[max_index]

plt.plot(frequencies, fs)
# plt.xscale('log')
plt.vlines(max_frequency, 0, max_value, colors='r')
plt.text(max_frequency, max_value / 2, s=f'freq: {max_frequency:.3f}', color='r')
plt.show()
