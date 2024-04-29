#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <stdlib.h>
#include "helper_cuda.h"
#include <cufft.h>

struct header {
    unsigned int ChunkID;
    unsigned int ChunkSize;
    unsigned int Format;
    unsigned int Subchunk1ID;
    unsigned int Subchunk1Size;
    unsigned short AudioFormat;
    unsigned short NumChannels;
    unsigned int SampleRate;
    unsigned int ByteRate;
    unsigned short BlockAlign;
    unsigned short BitsPerSample;
    unsigned int Subchunk2ID;
    unsigned int Subchunk2Size;
};

__global__ void applyHighpass(cufftComplex *freqData, int numSamples, int cutoffIndex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cutoffIndex) {
        freqData[idx].x = 0;
        freqData[idx].y = 0;
    }
}

__global__ void windowAndCopy(float *windowedData, const int16_t *audioData, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float window = 0.5 * (1 - cos(2 * 3.14159 * idx / (N - 1)));
        windowedData[idx] = window * audioData[idx];
    }
}

__global__ void accumulate(float *outputData, const float *inputData, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&outputData[idx], inputData[idx] / N);
    }
}


int main() {
    std::string infile_name = "white_noise.wav";
    std::string outfile_name = "output.wav";

    //open both files
    std::ifstream infile(infile_name, std::ios::binary);
    std::ofstream outfile(outfile_name, std::ios::binary);

    //quit if files cannot be opened
    if (!infile) {
      std::cout << "Error: Could not open input file." << std::endl;
      return 1;
    }
    if (!outfile) {
      std::cout << "Error: Could not open output file." << std::endl;
      return 1;
    }

    //create header struct and fill with data from input file, checking for supported sample rate before continuing
    header my_header;
    infile.read(reinterpret_cast<char*>(&my_header), sizeof(header));

    //set the parameters
    float cutoff_frequency = 10000.0;
    int N = 2048;
    int cutoffIndex = (int)(cutoff_frequency / my_header.SampleRate * N);

    //get number of samples
    int num_samples = my_header.Subchunk2Size/2;

    //allocate the area required to store the data
    int16_t* input_audio_data = (int16_t*) malloc(num_samples * sizeof(int16_t));
    float *output_audio_data = (float *)malloc(num_samples * sizeof(float));

    //get data from the file and store it in memory
    infile.read(reinterpret_cast<char*>(input_audio_data), num_samples * sizeof(int16_t));

    //allocate gpu resources
    int16_t *gpu_audio_data;
    float *gpu_windowed_data, *gpu_output_data;
    cufftComplex *gpu_freq_data;
    cufftHandle plan_fwd, plan_inv;
    checkCudaErrors(cudaMalloc(&gpu_audio_data, num_samples * sizeof(int16_t)));
    checkCudaErrors(cudaMalloc(&gpu_windowed_data, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&gpu_output_data, num_samples * sizeof(float)));
    checkCudaErrors(cudaMalloc(&gpu_freq_data, (N/2 + 1) * sizeof(cufftComplex)));
    

    //copy data to gpu memory
    checkCudaErrors(
        cudaMemcpy(
            gpu_audio_data, 
            input_audio_data, 
            num_samples*sizeof(int16_t), 
            cudaMemcpyHostToDevice
        )
    );

    //setup fft on gpu
    cufftPlan1d(&plan_fwd, N, CUFFT_R2C, 1);
    cufftPlan1d(&plan_inv, N, CUFFT_C2R, 1);
    
    float duration_ms = 0.0f;
    cudaEvent_t start, stop;

    // timing code instantiation
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));


    int hopSize = N / 2;
    for (int start = 0; start < num_samples - N; start += hopSize) {

        dim3 blocks((N + 1023) / 1024);
        dim3 threads(1024);

        windowAndCopy<<<blocks, threads>>>(gpu_windowed_data, gpu_audio_data + start, N);
        cufftExecR2C(plan_fwd, gpu_windowed_data, gpu_freq_data);

        applyHighpass<<<blocks, threads>>>(gpu_freq_data, N/2 + 1, cutoffIndex);

        cufftExecC2R(plan_inv, gpu_freq_data, gpu_windowed_data);
        accumulate<<<blocks, threads>>>(gpu_output_data + start, gpu_windowed_data, N);
    }

    // get the time
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&duration_ms, start, stop));

    //copy results back
    checkCudaErrors(cudaMemcpy(output_audio_data, gpu_output_data, num_samples * sizeof(float), cudaMemcpyDeviceToHost));

    printf("naive time: %.10fms\n", duration_ms);

    //convert back to int16
    for (int i = 0; i < num_samples; i++) {
        output_audio_data[i] /= N; // Normalize
        input_audio_data[i] = static_cast<int16_t>(output_audio_data[i]);
    }

    //write the header to the output file
    outfile.write(reinterpret_cast<const char*>(&my_header), sizeof(header));
    //write audio data
    outfile.write(reinterpret_cast<const char*>(input_audio_data), num_samples * sizeof(int16_t));

    //free memory
    free(input_audio_data);
    free(output_audio_data);
    checkCudaErrors(cudaFree(gpu_audio_data));
    checkCudaErrors(cudaFree(gpu_windowed_data));
    checkCudaErrors(cudaFree(gpu_output_data));
    checkCudaErrors(cudaFree(gpu_freq_data));
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    infile.close();
    outfile.close();
    return 0;

}