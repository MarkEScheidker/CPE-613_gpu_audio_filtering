#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <stdlib.h>
#include "B_48000.h"

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
    infile_name = "input.wav";
    outfile_name = "output.wav";

    infile = fopen(infile_name.c_str(), "r");   //open both files
    outfile = fopen(outfile_name.c_str(), "w+");
    
    //quit if files cannot be opened
    if (infile == NULL) {
      cout << "Error: Could not open input file." << endl;
      return 1;
    }
    if (outfile == NULL) {
      cout << "Error: Could not open output file." << endl;
      return 1;
    }

    //create header struct and fill with data from input file, checking for supported sample rate before continuing
    struct header my_header;
    fread(&my_header, sizeof(my_header),1, infile);

    //get number of samples
    num_samples = my_header.Subchunk2Size/2;

    //allocate the area required to store the data
    uint16_t *input_audio_data = (uint16_t *)malloc(num_samples*sizeof(uint16_t));
    //uint16_t *output_audio_data = (uint16_t *)malloc();

    //get data from the file and store it in memory
    uint16_t audio_sample;
    for (int i = 0; i < num_samples; i++) {
            fread(&audio_sample, sizeof(audio_sample), 1, infile);
            input_audio_data[i] = audio_sample;
    }

    //allocate gpu memory
    uint16_t *gpu_audio_data, *gpu_result;
    float *gpu_convol_kernel;
    checkCudaErrors(cudaMalloc(&gpu_audio_data, num_samples*sizeof(uint16_t)es));
    checkCudaErrors(cudaMalloc(&gpu_convol_kernel, BL*sizeof(float)));
    checkCudaErrors(cudaMalloc(&gpu_result, num_samples*sizeof(uint16_t));

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
    checkCudaErrors(cudaMemcpy(output_audio_data, gpu_result, grayscale_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    printf("naive time: %.10fms\n", duration_ms);

    //write the header to the output file
    fwrite(&my_header, sizeof(my_header), 1, outfile);
    //write the audio data
    fwrite(output_audio_data, sizeof(uint16_t), num_samples, outfile);

    //free memory
    free(input_audio_data);
    free(output_audio_data);
    checkCudaErrors(cudaFree(gpu_audio_data));
    checkCudaErrors(cudaFree(gpu_convol_kernel));
    checkCudaErrors(cudaFree(gpu_result));
    return 0;

}