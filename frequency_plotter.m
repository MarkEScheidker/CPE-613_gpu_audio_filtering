filename = ['output.wav'];

%read the audio data from the file
[data, fs] = audioread(filename);

%compute the fft
fftData = fft(data);

%shift zero frequency components to the center of the array
fftDataShifted = fftshift(fftData);

%compute the magnitude spectrum and normalize it
magnitude = abs(fftDataShifted) / length(fftDataShifted);

%frequency vector for plotting
f = (-length(data)/2:length(data)/2-1) * (fs/length(data));

%plot the frequency spectrum
figure;
plot(f, magnitude);
title('Frequency Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
axis tight;

grid on;
