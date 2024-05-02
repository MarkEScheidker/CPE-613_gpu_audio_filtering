[file1, fs1] = audioread('output.wav');
[file2, fs2] = audioread('output2.wav');

fft1 = fft(file1);
fft2 = fft(file2);

mag1 = abs(fft1);
mag2 = abs(fft2);

%normalize the magnitude spectra
norm_mag1 = mag1 / norm(mag1);
norm_mag2 = mag2 / norm(mag2);

%calculate Euclidean distance
euclidean_distance = norm(norm_mag1 - norm_mag2);

%calculate Cosine similarity
cosine_similarity = dot(norm_mag1, norm_mag2) / (norm(norm_mag1) * norm(norm_mag2));

fprintf('Euclidean Distance: %f\n', euclidean_distance);
fprintf('Cosine Similarity: %f\n', cosine_similarity);
