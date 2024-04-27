#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <stdlib.h>
#include "fdacoefs.h"
#include "helper_cuda.h"
#include <stdio.h>
#include <cuda_runtime.h>

__constant__ float const_conv_kernel[BL];

__global__ void tiled_convolution_1D_kernel(int16_t *result, const int16_t *audio_data, int data_size, int kernel_size) {
    extern __shared__ int16_t shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx = threadIdx.x;
    int half_kernel = kernel_size / 2;

    if (t_idx < half_kernel) {
        int left_index = idx - half_kernel;
        shared_data[t_idx] = (left_index < 0) ? 0 : audio_data[left_index];
    }

    shared_data[t_idx + half_kernel] = (idx < data_size) ? audio_data[idx] : 0;

    if (t_idx < half_kernel) {
        int right_index = idx + blockDim.x;
        shared_data[t_idx + blockDim.x + half_kernel] = (right_index >= data_size) ? 0 : audio_data[right_index];
    }
    __syncthreads();

    float value = 0.0f;
    for (int j = 0; j < kernel_size; ++j) {
        int shared_index = t_idx + j;
        value += (shared_data[shared_index] / 32768.0f) * const_conv_kernel[j];
    }

    if (idx < data_size) {
        value = value * 32768.0f;
        value = fmaxf(-32768.0f, fminf(32767.0f, roundf(value)));
        result[idx] = static_cast<int16_t>(value);
    }
}


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

    //get number of samples
    int num_samples = my_header.Subchunk2Size/2;

    //allocate the area required to store the data
    int16_t* input_audio_data = (int16_t*) malloc(num_samples * sizeof(int16_t));
    int16_t *output_audio_data = (int16_t *)malloc(num_samples * sizeof(int16_t));

    //get data from the file and store it in memory
    infile.read(reinterpret_cast<char*>(input_audio_data), num_samples * sizeof(int16_t));

    //allocate gpu memory
    int16_t *gpu_audio_data, *gpu_result;
    float *gpu_convol_kernel;
    checkCudaErrors(cudaMalloc(&gpu_audio_data, num_samples*sizeof(int16_t)));
    checkCudaErrors(cudaMalloc(&gpu_convol_kernel, BL*sizeof(float)));
    checkCudaErrors(cudaMalloc(&gpu_result, num_samples*sizeof(int16_t)));

    //copy data to gpu memory
    checkCudaErrors(
        cudaMemcpy(
            gpu_audio_data, 
            input_audio_data, 
            num_samples*sizeof(int16_t), 
            cudaMemcpyHostToDevice
        )
    );

    checkCudaErrors(
        cudaMemcpy(
            gpu_convol_kernel, 
            B, 
            BL*sizeof(float), 
            cudaMemcpyHostToDevice
        )
    );

    // copy filter to global memory
    cudaMemcpyToSymbol(const_conv_kernel, B, BL * sizeof(float));
    
    float duration_ms = 0.0f;
    cudaEvent_t start, stop;

    // timing code instantiation
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //kernel launch parameters
    int kernel_size = BL;
    int blockSize = 256;
    int numBlocks = (num_samples + blockSize - 1) / blockSize;
    int sharedMemSize = (blockSize + 2 * (kernel_size / 2)) * sizeof(int16_t);

    //launch the kernel
    checkCudaErrors(cudaEventRecord(start));

    // Kernel launch
    tiled_convolution_1D_kernel<<<numBlocks, blockSize, sharedMemSize>>>(gpu_result, gpu_audio_data, num_samples, kernel_size);

    // get the time
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&duration_ms, start, stop));

    //copy result back to host
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
    checkCudaErrors(cudaFree(gpu_convol_kernel));
    checkCudaErrors(cudaFree(gpu_result));
    infile.close();
    outfile.close();
    return 0;

}