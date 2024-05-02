# CPU implementation
This implementation can be run by first cloning the repository on a slurm capable system, running `git checkout cpu` and running `sbatch slurm.sh`, which will read in the audio file `white_noise.wav` and generate an `output.wav`. The program can be modified to read in a different file, but that file must be an uncompressed wav file with a 48khz sample rate, and a single channel of 16 bit PCM samples.

## Frequency Plotter
`frequenct_plotter.m` is a matlab program that provides an easy way to visualize the frequency components of an audio file in order to compare filtered and unfiltered waveforms. Simply provide the program with a path to a file, and it will plot its frequency componenents.
