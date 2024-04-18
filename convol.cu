#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <stdlib.h>
#include "fdacoefs.h"
#include "helper_cuda.h"

__global__ void convolution_1D_kernel(uint16_t *result, const uint16_t *audio_data, const float *conv_kernel, int data_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    int start = idx - kernel_size / 2;

    for (int j = 0; j < kernel_size; ++j) {
        if (start + j >= 0 && start + j < data_size) {
            float sample = static_cast<float>(audio_data[start + j]) - 32768.0f;
            sample /= 32768.0f; //scale to -1.0 to 1.0
            value += sample * conv_kernel[j];
        }
    }

    //convert result back to uint16_t
    if (idx < data_size) {
        value *= 32768.0f; //scale back to 16-bit range
        value += 32768.0f;
        result[idx] = static_cast<uint16_t>(fmaxf(0.0f, fminf(65535.0f, roundf(value))));
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
    std::string infile_name = "input.wav";
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
    uint16_t* input_audio_data = (uint16_t*) malloc(num_samples * sizeof(uint16_t));
    uint16_t *output_audio_data = (uint16_t *)malloc(num_samples * sizeof(uint16_t));

    //get data from the file and store it in memory
    infile.read(reinterpret_cast<char*>(input_audio_data), num_samples * sizeof(uint16_t));

    //allocate gpu memory
    uint16_t *gpu_audio_data, *gpu_result;
    float *gpu_convol_kernel;
    checkCudaErrors(cudaMalloc(&gpu_audio_data, num_samples*sizeof(uint16_t)));
    checkCudaErrors(cudaMalloc(&gpu_convol_kernel, BL*sizeof(float)));
    checkCudaErrors(cudaMalloc(&gpu_result, num_samples*sizeof(uint16_t)));

    //copy data to gpu memory
    checkCudaErrors(
        cudaMemcpy(
            gpu_audio_data, 
            input_audio_data, 
            num_samples*sizeof(uint16_t), 
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
    
    float duration_ms = 0.0f;
    cudaEvent_t start, stop;

    // timing code instantiation
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));


// naive -------------------------------------------------------------------------------------------------

    //kernel launch parameters
    int blockSize = 256;
    int numBlocks = (num_samples + blockSize - 1) / blockSize;

    //launch the kernel
    checkCudaErrors(cudaEventRecord(start));
    convolution_1D_kernel<<<numBlocks, blockSize>>>(gpu_result, gpu_audio_data, gpu_convol_kernel, num_samples, BL);

    // get the time
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&duration_ms, start, stop));

    //copy result back to host
    checkCudaErrors(cudaMemcpy(output_audio_data, gpu_result, num_samples * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    printf("naive time: %.10fms\n", duration_ms);

    //write the header to the output file
    outfile.write(reinterpret_cast<const char*>(&my_header), sizeof(header));
    //write the audio data
    outfile.write(reinterpret_cast<const char*>(output_audio_data), num_samples * sizeof(uint16_t));

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