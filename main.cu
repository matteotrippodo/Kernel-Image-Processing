#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

// Define the file size (height=width case) and the tile size
#define FILTER_SIZE 3 //or 5
#define TILE_WIDTH 16
#define SHAREDMEM_DIM (TILE_WIDTH + FILTER_SIZE -1 )

// Define the constant memory for the filter
__constant__ float constant_filter[FILTER_SIZE * FILTER_SIZE];

__global__ void KernelProcessingTiledConstant(uchar* input_image, uchar* output_image, int width, int height, int channels, int filter_size){
    extern __shared__ float image_tile[];

    int bx = blockIdx.x;
    int by = blockIdx.y ;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int channel = blockIdx.z * blockDim.z + tz;

    // Upload the first TILE_WIDTH*TILE_WIDTH pixel in shared memory
    int dest = (ty * TILE_WIDTH) + tx;
    int dX = dest % (SHAREDMEM_DIM);
    int dY = dest / (SHAREDMEM_DIM);

    int inputY =  by * TILE_WIDTH + dY - filter_size/2;
    int inputX = bx * TILE_WIDTH + dX - filter_size/2;
    int input = (inputY * width + inputX) * channels + channel;

    if(inputY >= 0 && inputY < height && inputX >= 0 && inputX < width){
        image_tile[dY*SHAREDMEM_DIM +dX] = input_image[input];
    }
    else{
        image_tile[dY*SHAREDMEM_DIM + dX] = 0;
    }

    // Check that all the threads have completed the upload
    __syncthreads();

    // Upload the remaining pixels in shared memory
    dest = (ty * TILE_WIDTH) + tx + TILE_WIDTH * TILE_WIDTH;
    dX = dest % (TILE_WIDTH + filter_size -1 );
    dY = dest/ (TILE_WIDTH + filter_size -1 );;

    inputY = dY + (by * TILE_WIDTH) - filter_size/2;
    inputX = dX + (bx * TILE_WIDTH) - filter_size/2;
    input = (inputY * width + inputX) * channels + channel;

    if(dY < (TILE_WIDTH + filter_size -1 )) {
        if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width) {
            image_tile[dY*SHAREDMEM_DIM + dX] = input_image[input];
        } else {
            image_tile[dY*SHAREDMEM_DIM + dX] = 0;
        }
    }

    // Check that all the threads have completed the upload
    __syncthreads();

    if (col < width && row < height && channel < channels) {
        float pixel_value = 0.0f;

        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {

                // Compute the convolution using the shared tile and the constant filter
                pixel_value += image_tile[((ty + i)*SHAREDMEM_DIM + (tx + j))] * constant_filter[i * filter_size + j];
            }
        }

        // Check if the value is in the uchar range and update the output image
        pixel_value = fmaxf(0.0f, fminf(255.0f, pixel_value));
        output_image[(row * width + col) * channels + channel] = (uchar) pixel_value;
    }

}

__global__ void KernelProcessingTiled(uchar* input_image, uchar* output_image, int width, int height, int channels, float* filter, int filter_size){
    extern __shared__ float image_tile[];

    int bx = blockIdx.x;
    int by = blockIdx.y ;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int channel = blockIdx.z * blockDim.z + tz;

    // Upload the first TILE_WIDTH*TILE_WIDTH pixel in shared memory
    int dest = (ty * TILE_WIDTH) + tx;
    int dX = dest % (SHAREDMEM_DIM);
    int dY = dest / (SHAREDMEM_DIM);

    int inputY =  by * TILE_WIDTH + dY - filter_size/2;
    int inputX = bx * TILE_WIDTH + dX - filter_size/2;
    int input = (inputY * width + inputX) * channels + channel;

    if(inputY >= 0 && inputY < height && inputX >= 0 && inputX < width){
        image_tile[dY*SHAREDMEM_DIM +dX] = input_image[input];
    }
    else{
        image_tile[dY*SHAREDMEM_DIM + dX] = 0;
    }
    // Check that all the threads have completed the upload
    __syncthreads();

    // Upload the remaining pixels in shared memory
    dest = (ty * TILE_WIDTH) + tx + TILE_WIDTH * TILE_WIDTH;
    dX = dest % (TILE_WIDTH + filter_size -1 );
    dY = dest/ (TILE_WIDTH + filter_size -1 );;

    inputY = dY + (by * TILE_WIDTH) - filter_size/2;
    inputX = dX + (bx * TILE_WIDTH) - filter_size/2;
    input = (inputY * width + inputX) * channels + channel;

    if(dY < (TILE_WIDTH + filter_size -1 )) {
        if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width) {
            image_tile[dY*SHAREDMEM_DIM + dX] = input_image[input];
        } else {
            image_tile[dY*SHAREDMEM_DIM + dX] = 0;
        }
    }

    // Check that all the threads have completed the upload
    __syncthreads();

    if (col < width && row < height && channel < channels) {
        float pixel_value = 0.0f;

        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                int index = ((ty + i) * SHAREDMEM_DIM + (tx + j)) ;//* channels + channel;

                // Update the pixel value using the shared tile and the global filter
                pixel_value += image_tile[index] * filter[i * filter_size + j];
            }
        }

        // Check if the pixel value is in the right range and update the output
        pixel_value = fmaxf(0.0f, fminf(255.0f, pixel_value));
        output_image[(row * width + col) * channels + channel] = (uchar) pixel_value;
    }

}

__global__ void KernelProcessingGlobal(uchar* input_image, uchar* output_image, int width, int height, int channels, float* filter, int filter_size) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < width && row < height && channel < channels) {
        float pixel_value = 0.0f;

        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                int neighbor_row = row + i - filter_size / 2;
                int neighbor_col = col + j - filter_size / 2;

                // Check if one of the neighbor pixel used for convolution is on or over the border
                bool is_border = neighbor_row < 0 || neighbor_row >= height || neighbor_col < 0 || neighbor_col >= width;


                if (!is_border) {
                    // Update the pixel value using the global input image and the global filter
                    pixel_value += input_image[(neighbor_row * width + neighbor_col) * channels + channel] *
                                   filter[i * filter_size + j];
                }
            }
        }

        // Check if the pixel value is in the (0,255) range and update the output pixel
        pixel_value = fmaxf(0.0f, fminf(255.0f, pixel_value));
        output_image[(row * width + col) * channels + channel] = (uchar) pixel_value;
    }
}

void KernelProcessingSequential(uchar* input_image, uchar* output_image, int width, int height, int channels, const float* filter, int filter_size) {

    // Use three nested loop to iterate sequentially over height, width and channels of the image
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int ch = 0; ch < channels; ch++) {
                float pixel_value = 0.0f;

                for (int i = 0; i < filter_size; ++i) {
                    for (int j = 0; j < filter_size; ++j) {
                        int neighbor_row = row + i - filter_size / 2;
                        int neighbor_col = col + j - filter_size / 2;

                        // Check if one of the neighbor pixel used for convolution is on or over the border
                        bool is_border = neighbor_row < 0 || neighbor_row >= height || neighbor_col < 0 || neighbor_col >= width;

                        if (!is_border) {
                            // Update the pixel value using the global input image and the global filter
                            pixel_value += input_image[(neighbor_row * width + neighbor_col) * channels + ch] *
                                           filter[i * filter_size + j];
                        }

                    }
                }

                // Check if the pixel value is in the right range and update the output pixel
                pixel_value = fmaxf(0.0f, fminf(255.0f, pixel_value));
                output_image[(row * width + col) * channels + ch] = (uchar) pixel_value;
            }
        }
    }
}


int main() {

    // Read the input image
    cv::Mat input_image = cv::imread(".\\images\\565.jpg", cv::IMREAD_COLOR);//cv::IMREAD_GRAYSCALE
    if (input_image.empty()) {
        std::cerr << "Error: Impossible load the image " << std::endl;
        return -1;
    }

    int width = input_image.cols;
    int height = input_image.rows;
    int channels = input_image.channels();

    // Compute the size of the input/output image
    int size = width * height * channels * sizeof(uchar);//width * height * sizeof(uchar);

    // Define some useful filters
    const float identity_filter[3][3] = {
            {0, 0, 0},
            {0, 1, 0},
            {0, 0, 0}
    };
    const float sharpenFilter[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
    };
    const float  edgeDetectorFilter[3][3] = {
            {-1.0f, -1.0f, -1.0f},
            {-1.0f, 8.0f, -1.0f},
            {-1.0f, -1.0f, -1.0f}
    };
    const float gaussian_filter[3][3] = {
            {1/16.0f, 2/16.0f, 1/16.0f},
            {2/16.0f, 4/16.0f, 2/16.0f},
            {1.0f/16.0f, 2/16.0f, 1/16.0f}
    };
    const float box_blur_filter[3][3] = {
            {1/9.0f, 1/9.0f, 1/9.0f},
            {1/9.0f, 1/9.0f, 1/9.0f},
            {1/9.0f, 1/9.0f, 1/9.0f}
    };
    const float unsharpKernel[5][5] = {
            {-1/256.0f, -4/256.0f, -6/256.0f, -4/256.0f, -1/256.0f},
            {-4/256.0f, -16/256.0f, -24/256.0f, -16/256.0f, -4/256.0f},
            {-6/256.0f, -24/256.0f, 476/256.0f, -24/256.0f, -6/256.0f},
            {-4/256.0f, -16/256.0f, -24/256.0f, -16/256.0f, -4/256.0f},
            {-1/256.0f, -4/256.0f, -6/256.0f, -4/256.0f, -1/256.0f}
    };

    int choosen_filter = std::stoi(std::getenv("FILTER"));

    const float * filter;
    int filter_size;
    switch(choosen_filter){
        case 0:
            printf("Identity filter\n");
            filter_size = 3;
            filter = (const float*) identity_filter;
            break;
        case 1:
            printf("Sharpen filter\n");
            filter_size = 3;
            filter = (const float*) sharpenFilter;
            break;
        case 2:
            printf("Edge detector filter\n");
            filter_size = 3;
            filter = (const float*) edgeDetectorFilter;
            break;
        case 3:
            printf("Gaussian filter\n");
            filter_size = 3;
            filter = (const float*) gaussian_filter;
            break;
        case 4:
            printf("Box blur filter\n");
            filter_size = 3;
            filter = (const float*) gaussian_filter;
            break;
        case 5:
            printf("Unsharp Mask filter\n");
            filter_size = 5;
            filter = (const float*) unsharpKernel;
            break;
    }

    uchar* d_input_image, * d_output_image;
    float* d_filter;

    uchar* seq_output_image = new uchar[size];
    auto seq_start_time = std::chrono::high_resolution_clock::now();

    // Call the sequential kernel image processing function
    KernelProcessingSequential(input_image.data, seq_output_image, width, height, channels, filter,filter_size);

    auto seq_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = seq_end_time - seq_start_time ;
    std::cout << "Sequential execution time: " << elapsed_seconds.count() << " secs" << std::endl;

    // Read the output image
    cv::Mat seq_output_Mat(height, width, CV_8UC(channels), seq_output_image);

    cv::imwrite("seq_output_image.jpg", seq_output_Mat);
    cv::imshow("Output Image", seq_output_Mat);
    cv::waitKey(0);

    delete[] seq_output_image;

    //////// CUDA PART ////////

    // Copy the filter in the constant memory
    cudaMemcpyToSymbol(constant_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // Reserve global memory in the gpu for input/output image and the filter
    cudaMalloc((void**)&d_filter, filter_size * filter_size * sizeof(float));
    cudaMalloc((void**)&d_input_image, size);
    cudaMalloc((void**)&d_output_image, size);

    // Copy the input image and the filter in the
    cudaMemcpy(d_input_image, input_image.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define the block and the grid size
    //dim3 blockSize(16, 16);
    dim3 blockSize(TILE_WIDTH ,TILE_WIDTH,1);
    //dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, (channels + blockSize.z - 1) / blockSize.z);  // Calcola le dimensioni del grid

    auto par_start_time = std::chrono::high_resolution_clock::now();

    // Call the cuda kernel for kernel image processing
    //KernelProcessingGlobal<<<gridSize, blockSize>>>(d_input_image, d_output_image, width, height, channels,d_filter,filter_size);
    //KernelProcessingTiled<<<gridSize, blockSize, (SHAREDMEM_DIM) * (SHAREDMEM_DIM) * sizeof (float)>>>(d_input_image, d_output_image, width, height, channels,d_filter,filter_size);
    KernelProcessingTiledConstant<<<gridSize, blockSize, SHAREDMEM_DIM * SHAREDMEM_DIM * sizeof (float)>>>(d_input_image, d_output_image, width, height, channels,filter_size);

    auto par_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> par_elapsed_seconds = par_end_time - par_start_time;
    std::cout << "Tempo di esecuzione: " << par_elapsed_seconds.count() << " secondi" << std::endl;

    uchar* par_output_image = new uchar[size];

    // Copy the output image from the gpu and read it
    cudaMemcpy(par_output_image, d_output_image, size, cudaMemcpyDeviceToHost);
    cv::Mat par_output_Mat(height, width, CV_8UC(channels), par_output_image);

    cv::imshow("Output Image", par_output_Mat);
    cv::waitKey(0);

    cudaFree(d_input_image);
    cudaFree(d_output_image);
    cudaFree(d_filter);

    delete[] par_output_image;

    return 0;
}



