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

__global__ void applyHighpassKernel(cufftComplex *freq_data, int n, float sample_rate, float cutoff_frequency) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int cutoff_index = (int)((cutoff_frequency / sample_rate) * n);

    if (index < cutoff_index) {
        //zero out frequencies below the cutoff
        freq_data[index].x = 0;
        freq_data[index].y = 0;
    }
}

__global__ void cudaKernelScale(float* data, float factor, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        data[index] *= factor;
    }
}

__global__ void convertFloatToInt16(float* input, int16_t* output, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        output[index] = __float2int_rn(input[index]);
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

    //set the cutoff frequency
    float cutoff_frequency = 1000.0;

    //get number of samples
    int num_samples = my_header.Subchunk2Size/2;

    //allocate the area required to store the data
    int16_t* input_audio_data = (int16_t*) malloc(num_samples * sizeof(int16_t));
    int16_t *output_audio_data = (int16_t *)malloc(num_samples * sizeof(int16_t));

    //get data from the file and store it in memory
    infile.read(reinterpret_cast<char*>(input_audio_data), num_samples * sizeof(int16_t));

    //allocate gpu resources
    int16_t *gpu_audio_data, *gpu_result;
    cufftHandle plan_fwd, plan_inv;
    cufftComplex *gpu_freq_data;
    checkCudaErrors(cudaMalloc(&gpu_audio_data, num_samples*sizeof(int16_t)));
    checkCudaErrors(cudaMalloc(&gpu_result, num_samples*sizeof(int16_t)));
    checkCudaErrors(cudaMalloc(&gpu_freq_data, sizeof(cufftComplex) * (num_samples/2 + 1)));
    

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
    cufftPlan1d(&plan_fwd, num_samples, CUFFT_R2C, 1);
    cufftPlan1d(&plan_inv, num_samples, CUFFT_C2R, 1);
    
    float duration_ms = 0.0f;
    cudaEvent_t start, stop;

    // timing code instantiation
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // execute fft
    cufftExecR2C(plan_fwd, (cufftReal*)gpu_audio_data, gpu_freq_data);
    cudaDeviceSynchronize();

    //launch the kernel
    int blockSize = 256;
    int numBlocks = (num_samples / 2 + 1 + blockSize - 1) / blockSize;
    applyHighpassKernel<<<numBlocks, blockSize>>>(gpu_freq_data, num_samples, my_header.SampleRate, cutoff_frequency);
    cudaDeviceSynchronize();

    //inverse fft
    cufftExecC2R(plan_inv, gpu_freq_data, (cufftReal*)gpu_result);
    cudaDeviceSynchronize();

    //normalize the fft output
    cudaKernelScale<<<numBlocks, blockSize>>>((float*)gpu_result, 1.0f / num_samples, num_samples);
    cudaDeviceSynchronize();

    //convert float to int
    convertFloatToInt16<<<numBlocks, blockSize>>>((float*)gpu_result, (int16_t*)gpu_result, num_samples);
    cudaDeviceSynchronize();

    // get the time
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&duration_ms, start, stop));

    //copy results back
    checkCudaErrors(cudaMemcpy(output_audio_data, gpu_result, num_samples * sizeof(int16_t), cudaMemcpyDeviceToHost));

    printf("naive time: %.10fms\n", duration_ms);

    //write the header to the output file
    outfile.write(reinterpret_cast<const char*>(&my_header), sizeof(header));
    //write the audio data
    outfile.write(reinterpret_cast<const char*>(output_audio_data), num_samples * sizeof(int16_t));

    //free memory
    free(input_audio_data);
    free(output_audio_data);
    checkCudaErrors(cudaFree(gpu_audio_data));
    checkCudaErrors(cudaFree(gpu_result));
    checkCudaErrors(cudaFree(gpu_freq_data));
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    infile.close();
    outfile.close();
    return 0;

}