import numpy as np
import matplotlib.pyplot as plt


def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])


def tophat(input_array):
    output = []
    for x in input_array:
        if x < 5 and x > -5:
            output.append(1)
        else:
            output.append(0)
    return output


if __name__ == "__main__":
    t_128 = np.linspace(0, 40, 128)
    spacing_128 = t_128[1]-t_128[0]
    omega_128 = np.linspace(0, 1/(2*spacing_128), 64)
    t_512 = np.linspace(0, 40, 512)
    spacing_512 = t_512[1]-t_512[0]
    omega_512 = np.linspace(0, 1/(2*spacing_512), 256)
    t_1024 = np.linspace(0, 40, 1024)
    spacing_1024 = t_1024[1]-t_1024[0]
    omega_1024 = np.linspace(0, 1/(2*spacing_1024), 512)
    tophat_128 = tophat(t_128)
    tophat_512 = tophat(t_512)
    tophat_1024 = tophat(t_1024)
    gauss_128 = np.exp(-t_128**2/5)
    gauss_512 = np.exp(-t_512**2/5)
    gauss_1024 = np.exp(-t_1024**2/5)
    gausscos_128 = np.exp(-t_128**2/10)*np.cos(t_128)
    gausscos_512 = np.exp(-t_512**2/10)*np.cos(t_512)
    gausscos_1024 = np.exp(-t_1024**2/10)*np.cos(t_1024)
    dft_tophat_128 = dft(tophat_128)
    dft_tophat_512 = dft(tophat_512)
    dft_tophat_1024 = dft(tophat_1024)
    dft_gauss_128 = dft(gauss_128)
    dft_gauss_512 = dft(gauss_512)
    dft_gauss_1024 = dft(gauss_1024)
    dft_gausscos_128 = dft(gausscos_128)
    dft_gausscos_512 = dft(gausscos_512)
    dft_gausscos_1024 = dft(gausscos_1024)
    fft_tophat_128 = fft(tophat_128)
    fft_tophat_512 = fft(tophat_512)
    fft_tophat_1024 = fft(tophat_1024)
    fft_gauss_128 = fft(gauss_128)
    fft_gauss_512 = fft(gauss_512)
    fft_gauss_1024 = fft(gauss_1024)
    fft_gausscos_128 = fft(gausscos_128)
    fft_gausscos_512 = fft(gausscos_512)
    fft_gausscos_1024 = fft(gausscos_1024)
    dft_tophat_fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(omega_128, 2/spacing_128 *
             np.abs(dft_tophat_128[:64]), label="N=128")
    plt.plot(omega_512, 2/spacing_512 *
             np.abs(dft_tophat_512[:256]), label="N=512")
    plt.plot(omega_1024, 2/spacing_1024 *
             np.abs(dft_tophat_1024[:512]), label="N=1024")
    ax.set_title("Discrete Fourier Transform of Tophat Function")
    ax.legend(loc="upper right")
    ax.set_xlabel("Frequency ($\omega$)")
    ax.set_ylabel("Amplitude (a.u.)")
    plt.savefig('dft_tophat.png', format='png')
    dft_gauss_fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(omega_128, 2/spacing_128 *
             np.abs(dft_gauss_128[:64]), label="N=128")
    plt.plot(omega_512, 2/spacing_512 *
             np.abs(dft_gauss_512[:256]), label="N=512")
    plt.plot(omega_1024, 2/spacing_1024 *
             np.abs(dft_gauss_1024[:512]), label="N=1024")
    ax.set_title("Discrete Fourier Transform of Gaussian Function")
    ax.legend(loc="upper right")
    ax.set_xlabel("Frequency ($\omega$)")
    ax.set_ylabel("Amplitude (a.u.)")
    plt.savefig('dft_gauss.png', format='png')
    dft_gausscos_fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(omega_128, 2/spacing_128 *
             np.abs(dft_gausscos_128[:64]), label="N=128")
    plt.plot(omega_512, 2/spacing_512 *
             np.abs(dft_gausscos_512[:256]), label="N=512")
    plt.plot(omega_1024, 2/spacing_1024 *
             np.abs(dft_gausscos_1024[:512]), label="N=1024")
    ax.set_title("Discrete Fourier Transform of GaussCos Function")
    ax.legend(loc="upper right")
    ax.set_xlabel("Frequency ($\omega$)")
    ax.set_ylabel("Amplitude (a.u.)")
    plt.savefig('dft_gausscos.png', format='png')
    fft_tophat_fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(omega_128, 2/spacing_128 *
             np.abs(fft_tophat_128[:64]), label="N=128")
    plt.plot(omega_512, 2/spacing_512 *
             np.abs(fft_tophat_512[:256]), label="N=512")
    plt.plot(omega_1024, 2/spacing_1024 *
             np.abs(fft_tophat_1024[:512]), label="N=1024")
    ax.set_title("Fast Fourier Transform of Tophat Function")
    ax.legend(loc="upper right")
    ax.set_xlabel("Frequency ($\omega$)")
    ax.set_ylabel("Amplitude (a.u.)")
    plt.savefig('fft_tophat.png', format='png')
    fft_gauss_fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(omega_128, 2/spacing_128 *
             np.abs(fft_gauss_128[:64]), label="N=128")
    plt.plot(omega_512, 2/spacing_512 *
             np.abs(fft_gauss_512[:256]), label="N=512")
    plt.plot(omega_1024, 2/spacing_1024 *
             np.abs(fft_gauss_1024[:512]), label="N=1024")
    ax.set_title("Fast Fourier Transform of Gaussian Function")
    ax.legend(loc="upper right")
    ax.set_xlabel("Frequency ($\omega$)")
    ax.set_ylabel("Amplitude (a.u.)")
    plt.savefig('fft_gauss.png', format='png')
    fft_gausscos_fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(omega_128, 2/spacing_128 *
             np.abs(fft_gausscos_128[:64]), label="N=128")
    plt.plot(omega_512, 2/spacing_512 *
             np.abs(fft_gausscos_512[:256]), label="N=512")
    plt.plot(omega_1024, 2/spacing_1024 *
             np.abs(fft_gausscos_1024[:512]), label="N=1024")
    ax.set_title("Fast Fourier Transform of GaussCos Function")
    ax.set_xlabel("Frequency ($\omega$)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(loc="upper right")
    plt.savefig('fft_gausscos.png', format='png')
    plt.show()
