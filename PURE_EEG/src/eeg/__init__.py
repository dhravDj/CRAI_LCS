"""
Pure EEG Module.
"""
from src.eeg.preprocessing import preprocess_eeg_window, butterworth_bandpass_filter, iir_notch_filter, common_average_reference
