import numpy as np
import glob
import os
import struct
import xml.etree.ElementTree as ET
import struct
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt


data_type_mapping = {
    "UnsignedMSB2": ">H",  # 2-byte unsigned int (big-endian)
    "UnsignedMSB4": ">I",  # 4-byte unsigned int (big-endian)
    "UnsignedByte": ">B",  # 1-byte unsigned int
    "IEEE754MSBSingle": ">f",  # 4-byte single-precision float (big-endian)
    "IEEE754MSBDouble": ">d",  # 8-byte double-precision float (big-endian)

}


class Record:
    def __init__(self, raw_data, fpath, number, fields):
        self.fpath = fpath
        self.number = number
        self.raw_data = raw_data
        self.fields = fields
        self.parse_record()

    def __repr__(self):
        out = ''
        out = out + f"Record {self.number}:\n"
        for field in self.fields:
            value = getattr(self, field["name"])
            out = out + f"  {field['name']}: {value}\n"
        out = out + ("-" * 40)
        return out
    
    def parse_record(self):
        for field in self.fields:
            if field["name"] == "ifgm":
                start = field["group_location"]-1
                num_ifgm_values = field["repetitions"]
                ifgm_values = []
                for i in range(num_ifgm_values):
                    # Extract each `ifgm` value
                    s = start + i * field["length"]
                    e = start + (i + 1) * field["length"]
                    ifgm_data = self.raw_data[s : e]
                    ifgm_value = struct.unpack(field["type"], ifgm_data)[0]
                    ifgm_values.append(ifgm_value)
                setattr(self, "ifgm", ifgm_values)
            else:
                start = field["offset"]
                end = start + field["length"]
                value = struct.unpack(field["type"], self.raw_data[start:end])[0]
                setattr(self, field["name"], value)



def parse_XML(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    namespace = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
    fields = []
    records_element = root.find(".//pds:Table_Binary/pds:records", namespace)

    # Extract the number of records
    num_records = int(records_element.text) if records_element is not None else None
    for field in root.findall(".//pds:Field_Binary", namespace):

        name = field.find("pds:name", namespace).text
        if name == 'ifgm':
            continue
        offset = int(field.find("pds:field_location", namespace).get("unit") == "byte" and
                    field.find("pds:field_location", namespace).text or 0)
        length = int(field.find("pds:field_length", namespace).get("unit") == "byte" and
                    field.find("pds:field_length", namespace).text or 0)
        data_type = field.find("pds:data_type", namespace).text


        struct_format = data_type_mapping.get(data_type, None)

        # Append to fields if valid type found
        if struct_format:
            fields.append({"name": name, "offset": offset-1, "type": struct_format, "length": length})
    
    for field in root.findall(".//pds:Group_Field_Binary", namespace):
        name = field.findall(".//pds:name", namespace)[0].text
        offset_element = field.findall(".//pds:field_location", namespace)[0]
        offset = int(offset_element.text) if offset_element is not None else 0
        g_loc = int(field.find("pds:group_location", namespace).get("unit") == "byte" and
                    field.find("pds:group_location", namespace).text or 0)
        length = int(field.findall(".//pds:field_length", namespace)[0].get("unit") == "byte" and
                    field.findall(".//pds:field_length", namespace)[0].text or 0)
        g_length = int(field.find("pds:group_length", namespace).get("unit") == "byte" and
                    field.find("pds:group_length", namespace).text or 0)
        data_type = field.findall(".//pds:data_type", namespace)[0].text
        n_reps_element = field.find("pds:repetitions", namespace)
        n_reps = int(n_reps_element.text) if n_reps_element is not None else 0



        struct_format = data_type_mapping.get(data_type, None)

        # Append to fields if valid type found
        if struct_format:
            fields.append({"name": name, "offset": offset-1, "type": struct_format, "length": length,
                           "repetitions": n_reps, "group_length":g_length, "group_location": g_loc})
    return fields, num_records

def sinusoidal_fit(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

def extract_voltage_spectrum(datafile):
    # Datafile is a list of record objects
    spectrum = []
    for record in datafile:
        rr = record.ifgm
        spectrum.append(rr)

    return spectrum #spectrum is n_records x 1414


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y 
def analyze_landing_site(folder, folder_name, periodicity):
    save_dir = '/Users/emmabelhadfa/Documents/Oxford/OTES/results'
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'Analysis for {folder_name}', fontsize=16)
    
    # Get average spectrum
    all_spectra = []
    for records in folder:
        voltages = [record.ifgm for record in records][0]
        spectrum_slice = voltages[600:800]
        all_spectra.append(spectrum_slice)
    
    avg_spectrum = np.mean(all_spectra, axis=0)
    x = np.arange(600, 800)
    
    # Apply all smoothing methods first
    # 1. Savitzky-Golay filter
    window_length = 21  # must be odd
    polyorder = 3
    smoothed_sg = savgol_filter(avg_spectrum, window_length, polyorder)
    
    # 2. Gaussian filter
    sigma = 3
    smoothed_gaussian = gaussian_filter1d(avg_spectrum, sigma)
    
    # 3. Bandpass filter
    fs = 1  # Sample rate
    lowcut = 1 / (periodicity + 5)
    highcut = 1 / (periodicity - 5)
    bandpassed = bandpass_filter(avg_spectrum, lowcut, highcut, fs)
    
    # Now plot everything
    # Plot 1: Original and smoothed signals
    ax1.plot(x, avg_spectrum, 'b-', label='Original Signal', alpha=0.7)
    ax1.plot(x, smoothed_sg, 'r-', label='Savitzky-Golay filter', alpha=0.7)
    ax1.plot(x, smoothed_gaussian, 'g-', label='Gaussian filter', alpha=0.7)
    ax1.plot(x, bandpassed, 'm-', label='Bandpass filter', alpha=0.7)
    ax1.set_title('Original Signal with Different Smoothing Methods')
    ax1.set_xlabel('Spectrum Index')
    ax1.set_ylabel('Voltage')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Channeling components
    channeling_sg = avg_spectrum - smoothed_sg
    channeling_gaussian = avg_spectrum - smoothed_gaussian
    channeling_bandpass = avg_spectrum - bandpassed
    
    ax2.plot(x, channeling_sg, 'r-', label='Channeling (SG)', alpha=0.7)
    ax2.plot(x, channeling_gaussian, 'g-', label='Channeling (Gaussian)', alpha=0.7)
    ax2.plot(x, channeling_bandpass, 'm-', label='Channeling (Bandpass)', alpha=0.7)
    ax2.set_title('Extracted Channeling Components')
    ax2.set_xlabel('Spectrum Index')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: FFT Analysis
    fft_result = np.fft.fft(channeling_sg)
    freqs = np.fft.fftfreq(len(channeling_sg), 1)
    
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    power = np.abs(fft_result)[pos_mask]
    
    ax3.semilogy(freqs, power, 'b-', label='FFT of Channeling')
    
    top_indices = np.argsort(power)[-3:][::-1]
    print(f"\n{folder_name} channeling frequency analysis:")
    for idx, ind in enumerate(top_indices):
        period = 1/freqs[ind]
        print(f"Peak {idx+1}: frequency = {freqs[ind]:.4f}, period = {period:.2f} samples")
        ax3.axvline(x=freqs[ind], color=f'C{idx+1}', linestyle='--', 
                   label=f'Peak {idx+1}: period={period:.2f}')
    
    ax3.set_title('FFT of Channeling (log scale)')
    ax3.set_xlabel('Frequency (cycles/sample)')
    ax3.set_ylabel('Power')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f'{folder_name}_channellinganalysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved analysis for {folder_name} to {filename}")
    plt.close(fig)
    
    return smoothed_sg, smoothed_gaussian, bandpassed

def display_voltage_plot(folders, folder_names):
    datasets = {}
    periodicities = []  # Store periodicities for each folder

    for i, dataset in enumerate(folders):
        specs = []
        for datafile in dataset:
            # Each file is a list of record objects
            spec = extract_voltage_spectrum(datafile)
            spec_mean = np.mean(spec, axis=0)  # Compute mean along the correct axis
            specs.append(spec_mean)  # Append the mean spectrum to the list

        specs = np.mean(specs, 0)
        datasets[folder_names[i]] = specs

        # Compute periodicity for each folder
        peaks, _ = find_peaks(specs, distance=5)  # Adjust distance as needed
        if len(peaks) > 1:
            peak_distances = np.diff(peaks)
            avg_period = np.mean(peak_distances)
            periodicities.append(avg_period)
        else:
            periodicities.append(None)  # Handle case with no peaks

    plot_data(datasets)

    # Plot 1: Voltage Spectrum
    plt.subplot(2, 1, 1)
    for i, folder in enumerate(folders):
        all_spectra = []
        for records in folder:
            voltages = [record.ifgm for record in records][0]
            spectrum_slice = voltages[600:800]
            all_spectra.append(spectrum_slice)
        
        avg_spectrum = np.mean(all_spectra, axis=0)
        plt.plot(range(600, 800), avg_spectrum, label=folder_names[i])
    
    plt.title('Average Voltage Spectrum (600-800)')
    plt.xlabel('Spectrum Index')
    plt.ylabel('Voltage')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: FFT
    plt.subplot(2, 1, 2)
    for i, folder in enumerate(folders):
        all_spectra = []
        for records in folder:
            voltages = [record.ifgm for record in records][0]
            spectrum_slice = voltages[600:800]
            all_spectra.append(spectrum_slice)
        
        avg_spectrum = np.mean(all_spectra, axis=0)
        
        # Remove DC component (mean)
        avg_spectrum = avg_spectrum - np.mean(avg_spectrum)
        
        # Compute FFT
        fft_result = np.fft.fft(avg_spectrum)
        # Calculate frequency axis properly
        sample_spacing = 1  # assuming uniform spacing of 1
        freqs = np.fft.fftfreq(len(avg_spectrum), sample_spacing)
        
        # Get positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        power = np.abs(fft_result[:len(freqs)//2])
        
        # Plot on log scale for better visibility
        plt.semilogy(pos_freqs, power, label=folder_names[i])
    
    plt.title('FFT Power Spectrum of Average Signal')
    plt.xlabel('Frequency (cycles per sample)')
    plt.ylabel('Power (log scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Analyze each landing site separately
    smoothed_data = {}
    for folder, folder_name, periodicity in zip(folders, folder_names, periodicities):
        if periodicity is not None:
            smoothed_sg, smoothed_gaussian, bandpassed = analyze_landing_site(folder, folder_name, periodicity)
            smoothed_data[folder_name] = {
                'savgol': smoothed_sg,
                'gaussian': smoothed_gaussian,
                'bandpassed': bandpassed
            }
    
    # Comparison of Bandpass Filters
    plt.figure(figsize=(12, 6))
    for folder_name in folder_names:
        if folder_name in smoothed_data:
            plt.plot(range(600, 800), smoothed_data[folder_name]['bandpassed'], 
                     label=f'{folder_name} Bandpass', alpha=0.7)
    plt.title('Comparison of Bandpass Filters')
    plt.xlabel('Spectrum Index')
    plt.ylabel('Voltage')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Comparison of Savitzky-Golay Filters
    plt.figure(figsize=(12, 6))
    for folder_name in folder_names:
        if folder_name in smoothed_data:
            plt.plot(range(600, 800), smoothed_data[folder_name]['savgol'], 
                     label=f'{folder_name} Savitzky-Golay', alpha=0.7)
    plt.title('Comparison of Savitzky-Golay Filters')
    plt.xlabel('Spectrum Index')
    plt.ylabel('Voltage')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data(datasets):
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset
    for dataset_name, values in datasets.items():
        plt.plot(values, label=dataset_name)
    
    # Add grid, labels, title, and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Original Voltage Spectrums")
    plt.xlabel("Spectrum Index")
    plt.ylabel("Voltage (V)")
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":

    fpath = '/Users/emmabelhadfa/Documents/Oxford/OTES/data/locations/level0/**/*.dat' # replace this with the path to your folder containing all your dat files
    folders = []
    fprev = None
    k = -1
    folder_names = []
    for filepath in glob.glob(fpath, recursive=True):
        path = os.path.normpath(filepath)
        fcurr = path.split(os.sep)[-2]
        if fcurr != fprev:
            k += 1
            folders.append([])
            folder_names.append(fcurr)
        xmlp = filepath[:-4] + '.xml'
        fields, num_records = parse_XML(xmlp)

        record_length = int(os.path.getsize(filepath)/num_records)


        with open(filepath, 'rb') as file:
            parsed_records = []
            for i in range(num_records):
                record = file.read(record_length)
                record_obj = Record(record, filepath, i, fields)
                parsed_records.append(record_obj)
        folders[k].append(parsed_records)
        fprev = fcurr
    display_voltage_plot(folders, folder_names)
