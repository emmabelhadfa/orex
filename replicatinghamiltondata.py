import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

# Directory containing HDF5 files
data_directory = "/Users/emmabelhadfa/Documents/Oxford/OTES/data/locations/"
save_dir = '/Users/emmabelhadfa/Documents/Oxford/OTES/results'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get all HDF5 file paths in the directory
file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".hdf5")]

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.01)  # Ensure lowcut is above 0
    high = min(highcut / nyquist, 0.99)  # Ensure highcut is below Nyquist
    if low >= high:
        raise ValueError("Lowcut frequency must be less than highcut frequency.")
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Process each file separately
for file_path in file_paths:
    site_name = os.path.basename(file_path).split('.')[0]
    
    with h5py.File(file_path, "r") as file:
        mt_emissivity = file["mt_emissivity"][:, :, :]
        xaxis_L3 = file["xaxis_L3"][:, 0, 0]

    mt_emissivity_clipped = np.clip(mt_emissivity, -1.2, 1.2)
    
    # Filter non-outliers
    means = np.mean(mt_emissivity_clipped, axis=(0, 1))
    z_scores = (means - np.mean(means)) / np.std(means)
    non_outliers = np.where(np.abs(z_scores) <= 3)[0]
    
    non_outlier_emissivity = mt_emissivity_clipped[:, :, non_outliers[non_outliers < mt_emissivity_clipped.shape[2]]]
    site_average = np.mean(non_outlier_emissivity, axis=2)[:, 0]

    # Split the spectrum at wavenumber 600
    high_freq_mask = xaxis_L3 >= 600
    low_freq_mask = xaxis_L3 < 600

    # Find periodicity in the high frequency region
    high_freq_data = site_average[high_freq_mask]
    peaks, _ = find_peaks(high_freq_data, distance=5)
    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        avg_period = np.mean(peak_distances)
        print(f"{site_name} periodicity: {avg_period:.2f} samples")
        
        # Apply bandpass filter based on found periodicity
        fs = 1  # Sample rate (1 sample per index)
        lowcut = 1 / (avg_period + 5)  # Adjust window as needed
        highcut = 1 / (avg_period - 5)
        
        # Ensure valid frequency range
        if lowcut < highcut:
            smoothed_bandpass = site_average.copy()
            smoothed_bandpass[high_freq_mask] = bandpass_filter(site_average[high_freq_mask], 
                                                              lowcut, highcut, fs)
        else:
            print(f"Invalid frequency range for {site_name}, skipping bandpass filter.")
            smoothed_bandpass = site_average.copy()
    else:
        print(f"No clear periodicity found for {site_name}")
        smoothed_bandpass = site_average.copy()

    # Apply existing filters
    window_length = 21  # must be odd
    polyorder = 3
    smoothed_sg = site_average.copy()
    smoothed_sg[high_freq_mask] = savgol_filter(site_average[high_freq_mask], 
                                               window_length, polyorder)
    
    sigma = 3
    smoothed_gaussian = site_average.copy()
    smoothed_gaussian[high_freq_mask] = gaussian_filter1d(site_average[high_freq_mask], sigma)

    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'OTES Emissivity Spectrum Analysis - {site_name}', fontsize=16)

    # Plot 1: Original Data
    ax1.plot(xaxis_L3, site_average, 'b-', label='Original', alpha=0.7)
    ax1.axvline(x=600, color='r', linestyle='--', alpha=0.5, label='Filter boundary')
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("Emissivity")
    ax1.set_title("Original Emissivity Spectrum")
    ax1.set_xlim(1500, 300)
    ax1.set_ylim(0.95, 0.99)
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Smoothed Data
    ax2.plot(xaxis_L3, site_average, 'b-', label='Original', alpha=0.4)
    ax2.plot(xaxis_L3, smoothed_sg, 'r-', label='Savitzky-Golay', alpha=0.7)
    ax2.plot(xaxis_L3, smoothed_gaussian, 'g-', label='Gaussian', alpha=0.7)
    ax2.plot(xaxis_L3, smoothed_bandpass, 'm-', label='Bandpass', alpha=0.7)
    ax2.axvline(x=600, color='r', linestyle='--', alpha=0.5, label='Filter boundary')
    ax2.set_xlabel("Wavenumber (cm⁻¹)")
    ax2.set_ylabel("Emissivity")
    ax2.set_title("Smoothed Emissivity Spectrum (Filtered ≥600 cm⁻¹)")
    ax2.set_xlim(1500, 300)
    ax2.set_ylim(0.95, 0.99)
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Channeling
    channeling_sg = np.zeros_like(site_average)
    channeling_gaussian = np.zeros_like(site_average)
    channeling_bandpass = np.zeros_like(site_average)
    
    channeling_sg[high_freq_mask] = site_average[high_freq_mask] - smoothed_sg[high_freq_mask]
    channeling_gaussian[high_freq_mask] = site_average[high_freq_mask] - smoothed_gaussian[high_freq_mask]
    channeling_bandpass[high_freq_mask] = site_average[high_freq_mask] - smoothed_bandpass[high_freq_mask]

    ax3.plot(xaxis_L3, channeling_sg, 'r-', label='Channeling (SG)', alpha=0.7)
    ax3.plot(xaxis_L3, channeling_gaussian, 'g-', label='Channeling (Gaussian)', alpha=0.7)
    ax3.plot(xaxis_L3, channeling_bandpass, 'm-', label='Channeling (Bandpass)', alpha=0.7)
    ax3.axvline(x=600, color='r', linestyle='--', alpha=0.5, label='Filter boundary')
    ax3.set_xlabel("Wavenumber (cm⁻¹)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Extracted Channeling Components (≥600 cm⁻¹)")
    ax3.set_xlim(1500, 300)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()

    # Save the analysis figure
    filename = os.path.join(save_dir, f'{site_name}_600emissivityanalysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved analysis for {site_name} to {filename}")
    plt.close()

    # Create comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(xaxis_L3, site_average, 'b-', label='Original', alpha=0.4)
    plt.plot(xaxis_L3, smoothed_gaussian, 'g-', label='Smoothed (Gaussian)', alpha=0.7)
    plt.axvline(x=600, color='r', linestyle='--', alpha=0.5, label='Filter boundary')
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Emissivity")
    plt.title(f"OTES Emissivity Spectrum - {site_name}")
    plt.xlim(1500, 300)
    plt.ylim(0.95, 0.99)
    plt.grid(True)
    plt.legend()

    comparison_filename = os.path.join(save_dir, f'{site_name}_600emissivity_comparison.png')
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot for {site_name} to {comparison_filename}")
    plt.close()

# Final all-sites comparison of bandpass-smoothed data
plt.figure(figsize=(12, 6))
for file_path in file_paths:
    site_name = os.path.basename(file_path).split('.')[0]
    with h5py.File(file_path, "r") as file:
        mt_emissivity = file["mt_emissivity"][:, :, :]
        xaxis_L3 = file["xaxis_L3"][:, 0, 0]

    mt_emissivity_clipped = np.clip(mt_emissivity, -1.2, 1.2)
    means = np.mean(mt_emissivity_clipped, axis=(0, 1))
    z_scores = (means - np.mean(means)) / np.std(means)
    non_outliers = np.where(np.abs(z_scores) <= 3)[0]
    non_outlier_emissivity = mt_emissivity_clipped[:, :, non_outliers[non_outliers < mt_emissivity_clipped.shape[2]]]
    site_average = np.mean(non_outlier_emissivity, axis=2)[:, 0]
    
    # Apply bandpass filter to high frequencies
    high_freq_mask = xaxis_L3 >= 600
    peaks, _ = find_peaks(site_average[high_freq_mask], distance=5)
    
    smoothed = site_average.copy()
    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        avg_period = np.mean(peak_distances)
        fs = 1
        lowcut = 1 / (avg_period + 5)
        highcut = 1 / (avg_period - 5)
        
        if lowcut < highcut:
            smoothed[high_freq_mask] = bandpass_filter(site_average[high_freq_mask], 
                                                     lowcut, highcut, fs)
    
    plt.plot(xaxis_L3, smoothed, label=site_name, alpha=0.7)

plt.axvline(x=600, color='r', linestyle='--', alpha=0.5, label='Filter boundary')
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Emissivity")
plt.title("OTES Emissivity Spectrum - All Sites Comparison (Bandpass Filtered ≥600 cm⁻¹)")
plt.xlim(1500, 300)
plt.ylim(0.95, 0.99)
plt.grid(True)
plt.legend()

bandpass_comparison_filename = os.path.join(save_dir, 'all_sites_bandpass_comparison.png')
plt.savefig(bandpass_comparison_filename, dpi=300, bbox_inches='tight')
print(f"Saved bandpass-filtered comparison to {bandpass_comparison_filename}")
plt.close()
