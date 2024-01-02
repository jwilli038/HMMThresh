import pywt
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from skimage.metrics import structural_similarity as ssim
import copy
from skimage.metrics import mean_squared_error
from skimage.restoration import denoise_wavelet

def piecewise_constant_signal(n_samples=1024, n_jumps=5):
    jump_indices = np.sort(np.random.randint(0, n_samples, n_jumps))
    signal = np.zeros(n_samples)
    for idx in jump_indices:
        signal[idx:] = np.random.randn()  # Assign random values after each jump
    return signal

def calculate_MSE(original_signal, noisy_signal, denoised_signal):
    universal_thresh_signal = denoise_wavelet(noisy_signal, method='VisuShrink', wavelet='db6')
    universal_thresh_signal = universal_thresh_signal-np.mean(universal_thresh_signal-original_signal)
    denoised_signal = denoised_signal-np.mean(denoised_signal-original_signal)
    mse_universal = mean_squared_error(original_signal, universal_thresh_signal)
    mse_denoised = mean_squared_error(original_signal, denoised_signal)
    
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, label='Original Signal', alpha=0.5)
    plt.plot(denoised_signal, label='Reconstructed Signal', color = 'r', alpha=0.8)
    plt.plot(universal_thresh_signal, label='Universal Threshold', alpha=0.8)
    
    plt.legend()
    plt.title('Original Signal vs Reconstructed Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    print(f"MSE between Original and Universal Threshold Signal: {mse_universal}")
    print(f"MSE between Original and Denoised Signal: {mse_denoised}")


def objective_function(original_signal, reconstructed_signal, thresholded_wavelets):
    # Calculate SSIM between original and reconstructed signals
    ssim_value = ssim(original_signal, reconstructed_signal, data_range=original_signal.max() - original_signal.min())

    # Define a penalty term based on the number of thresholded wavelets
    num_nonzero_coeffs = thresholded_wavelets  # Count non-zero coefficients
    penalty = num_nonzero_coeffs   # Adjust the penalty factor as needed

    # Combine SSIM and penalty into the objective function
    objective_value = ssim_value + np.log(penalty/910)+3
    return objective_value

def denoise_signal(noisy_signal):
    coeffs_blocks = pywt.wavedec(noisy_signal, 'db6', level=3)
    wavelet_levels = len(coeffs_blocks) - 1
    coeff_tot = np.size(np.hstack(coeffs_blocks[1:]))
    previous_thresh = 0
    objective = 0
    fig, axs = plt.subplots(wavelet_levels, 1, figsize=(10, 6 * wavelet_levels))

    for i in range(wavelet_levels, 0, -1):
        level_coeff = coeffs_blocks[i]
        block_size = int(np.log(len(level_coeff)))

        blocks = [level_coeff[j:j + block_size] for j in range(0, len(level_coeff), block_size)
                  if j + block_size <= len(level_coeff)]
        n_states = block_size

        model = hmm.GaussianHMM(n_components=n_states)
        model.fit(blocks)

        labels = model.predict(blocks)

        for label in range(n_states):
            thresholded_coeffs = copy.deepcopy(coeffs_blocks)
            thresh_size = 0
            for j, block in enumerate(blocks):
                if labels[j] == label:
                    thresholded_coeffs[i][j * block_size:(j + 1) * block_size] = 0
                    thresh_size = thresh_size + np.size(thresholded_coeffs[i][j * block_size:(j + 1) * block_size])
            thresholded_coeffs[i][-block_size:] = 0
            reconstructed_signal = pywt.waverec(thresholded_coeffs, 'db6')
            thresholded_wavelets = thresh_size + previous_thresh
            current_objective = objective_function(noisy_signal, reconstructed_signal, thresholded_wavelets)
            if current_objective >= objective:
                objective = current_objective
                previous_thresh = thresholded_wavelets
                coeffs_blocks[i] = thresholded_coeffs[i]

        axs[wavelet_levels - i - 1].plot(coeffs_blocks[i])
        axs[wavelet_levels - i - 1].set_title(f'Level {i}')
        axs[wavelet_levels - i - 1].set_xlabel('Sample')
        axs[wavelet_levels - i - 1].set_ylabel('Coefficient')
        axs[wavelet_levels - i - 1].grid()

    reconstructed = pywt.waverec(coeffs_blocks, 'db6')
    plt.figure()
    plt.plot(reconstructed, label='Reconstructed Signal', alpha=0.8)
    plt.legend()
    plt.title('Reconstructed Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

    print(previous_thresh)
    return reconstructed

# Generate noisy signal
#signal = pywt.data.demo_signal('doppler', 1024)
#signal = pywt.data.demo_signal('blocks',1024)
#signal = pywt.data.demo_signal('bumps',1024)
#signal = pywt.data.demo_signal('heavisine',1024)
#signal = pywt.data.demo_signal('Piece-Polynomial',1024)
#signal = pywt.data.demo_signal('LinChirps',1024)
#signal = pywt.data.demo_signal('TwoChirp',1024)
#signal = pywt.data.demo_signal('QuadChirp',1024)
#signal = pywt.data.demo_signal('MishMash',1024)
#signal = pywt.data.demo_signal('HypChirps',1024)
noise_power = np.var(signal) / 10**(3 / 10)
noisy_signal = signal + np.random.lognormal(mean=0, sigma=np.sqrt(noise_power), size=len(signal))
reconstructed = denoise_signal(noisy_signal)
calculate_MSE(signal, noisy_signal, reconstructed)