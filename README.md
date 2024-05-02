# Naive GPU
This implementation can be run by first cloning the repository on a slurm capable system, and running `sbatch slurm.sh`, which will read in the audio file `white_noise.wav` and generate an `output.wav`. The program can be modified to read in a different file, but that file must be an uncompressed wav file with a 48khz sample rate, and a single channel of 16 bit PCM samples.

## Frequency Plotter
`frequenct_plotter.m` is a matlab program that provides an easy way to visualize the frequency components of an audio file in order to compare filtered and unfiltered waveforms. Simply provide the program with a path to a file, and it will plot its frequency componenents.

## Audio Comparison
`audio_compare.m` is a matlab program that will compare two audio files of equal length using Euclidean distance and cosine similarity. Because of the slight differences in floating point math between different systems, it becomes necessary to quantify exactly how much two sampled waveforms deviate in order to detect any unextpected deviation in expected filter output.
