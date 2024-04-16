

int main() {


    std::ifstream file("input.wav");
    if (!file.is_open()) {
        printf("Failed to open file\n");
        return 1;
    }

    file >> pixel_w >> pixel_h;

    //get the size for the input and output
    long int grayscale_size = pixel_h*pixel_w;

    //allocate the area required to store the data
    uint8_t *input_image_data = (uint8_t *)malloc(grayscale_size);
    uint8_t *output_image_data = (uint8_t *)malloc(grayscale_size);

    //get data from the file and store it in memory
    for (int i = 0; i < pixel_h; i++) {
        for (int j = 0; j < pixel_w; j++) {
            std::string value_str;
            file >> value_str;
            int value = std::stoi(value_str);
            input_image_data[(i * pixel_w) + j] = static_cast<uint8_t>(value);
        }
    }

    std::ifstream file2("input_kernel.txt");
    if (!file2.is_open()) {
        printf("Failed to open kernel file\n");
        return 1;
    }

    kernel_dim = ((FILTER_RADIUS * 2) + 1);

    //get the size for the kernel
    int kernel_size = kernel_dim*kernel_dim;

    //allocate the area required to store the data
    float *kernel_data = (float *)malloc(kernel_size * sizeof(float));

    //get data from the file and store it in memory
    for (int i = 0; i < kernel_dim; i++) {
        for (int j = 0; j < kernel_dim; j++) {
            std::string value_str;
            file2 >> value_str;
            float value = std::stof(value_str);
            kernel_data[i * kernel_dim + j] = value;
        }
    }

    for (int i = 0; i < kernel_dim; i++) {
        for (int j = 0; j < kernel_dim; j++) {
            printf("%.4f    ", kernel_data[i * kernel_dim + j]);
        }
        printf("\n");
    } 


    //allocate gpu memory
    uint8_t *gpu_image_data, *gpu_result;
    float *gpu_convol_kernel;
    checkCudaErrors(cudaMalloc(&gpu_image_data, grayscale_size));
    checkCudaErrors(cudaMalloc(&gpu_convol_kernel, kernel_size*sizeof(float)));
    checkCudaErrors(cudaMalloc(&gpu_result, grayscale_size));

    //copy data to gpu memory
    checkCudaErrors(
        cudaMemcpy(
            gpu_image_data, 
            input_image_data, 
            grayscale_size, 
            cudaMemcpyHostToDevice
        )
    );

    checkCudaErrors(
        cudaMemcpy(
            gpu_convol_kernel, 
            kernel_data, 
            kernel_size, 
            cudaMemcpyHostToDevice
        )
    );
    
    float duration_ms = 0.0f;
    cudaEvent_t start, stop;

    // timing code instantiation
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

/*
// naive -------------------------------------------------------------------------------------------------

    //kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((pixel_w + threadsPerBlock.x - 1) / threadsPerBlock.x, (pixel_h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    checkCudaErrors(cudaEventRecord(start));

    //launch the kernel
    convolution_2D_kernel<<<numBlocks, threadsPerBlock>>>(gpu_result, gpu_image_data, gpu_convol_kernel, pixel_w, pixel_h, kernel_dim);

    // get the time
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&duration_ms, start, stop));

    //copy result back to host
    checkCudaErrors(cudaMemcpy(output_image_data, gpu_result, grayscale_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    writePGM("naive_image.pgm", output_image_data, pixel_w, pixel_h);
    printf("naive time: %.10fms\n", duration_ms);
*/

    //free memory
    free(input_image_data);
    free(kernel_data);
    free(output_image_data);
    checkCudaErrors(cudaFree(gpu_image_data));
    checkCudaErrors(cudaFree(gpu_convol_kernel));
    checkCudaErrors(cudaFree(gpu_result));
    return 0;