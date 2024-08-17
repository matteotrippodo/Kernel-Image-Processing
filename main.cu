#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>
#include <vector>
#include <numeric>
#include <filesystem>

// Define some useful constants for the code
// Choose the filter, the size of the blocks and the number of tests
#define FILTER_CHOICE 2
#if FILTER_CHOICE < 5
    #define FILTER_SIZE 3
#else
    #define FILTER_SIZE 5
#endif
#define BLOCK_WIDTH 16
#define TILE_WIDTH (BLOCK_WIDTH + FILTER_SIZE -1 )
#define TEST_NUM 200

// Define the constant memory for the filter
__constant__ float constant_filter[FILTER_SIZE * FILTER_SIZE];

// CUDA kernel that uses tiling and constant memory for the filter
__global__ void KernelProcessingTiledConstant(const uchar* input_image, uchar* output_image, int width, int height, int channels){
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
    int dest = (ty * BLOCK_WIDTH) + tx;
    int dX = dest % (TILE_WIDTH);
    int dY = dest / (TILE_WIDTH);

    int inputY =  by * BLOCK_WIDTH + dY - FILTER_SIZE/2;
    int inputX = bx * BLOCK_WIDTH + dX - FILTER_SIZE/2;
    int input = (inputY * width + inputX) * channels + channel;

    if(inputY >= 0 && inputY < height && inputX >= 0 && inputX < width){
        image_tile[dY*TILE_WIDTH +dX] = input_image[input];
    }
    else{
        image_tile[dY*TILE_WIDTH + dX] = 0;
    }

    // Check that all the threads have completed the upload
    __syncthreads();

    // Upload the remaining pixels in shared memory
    dest = (ty * BLOCK_WIDTH) + tx + BLOCK_WIDTH * BLOCK_WIDTH;
    dX = dest % (BLOCK_WIDTH + FILTER_SIZE -1 );
    dY = dest/ (BLOCK_WIDTH + FILTER_SIZE -1 );;

    inputY = dY + (by * BLOCK_WIDTH) - FILTER_SIZE/2;
    inputX = dX + (bx * BLOCK_WIDTH) - FILTER_SIZE/2;
    input = (inputY * width + inputX) * channels + channel;

    if(dY < (BLOCK_WIDTH + FILTER_SIZE -1 )) {
        if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width) {
            image_tile[dY*TILE_WIDTH + dX] = input_image[input];
        } else {
            image_tile[dY*TILE_WIDTH + dX] = 0;
        }
    }

    // Check that all the threads have completed the upload
    __syncthreads();

    if (col < width && row < height && channel < channels) {
        float pixel_value = 0.0f;

        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                int index = ((ty + i)*TILE_WIDTH + (tx + j));

                // Compute the convolution using the shared tile and the constant filter
                pixel_value += image_tile[index] * constant_filter[i * FILTER_SIZE + j];
            }
        }

        // Check if the value is in the uchar range and update the output image
        pixel_value = fmaxf(0.0f, fminf(255.0f, pixel_value));
        output_image[(row * width + col) * channels + channel] = (uchar) pixel_value;
    }

}

// CUDA kernel that uploads tiles of image in shared memory
__global__ void KernelProcessingTiled(const uchar* input_image, uchar* output_image, int width, int height, int channels, float* filter){
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
    int dest = (ty * BLOCK_WIDTH) + tx;
    int dX = dest % (TILE_WIDTH);
    int dY = dest / (TILE_WIDTH);

    int inputY =  by * BLOCK_WIDTH + dY - FILTER_SIZE/2;
    int inputX = bx * BLOCK_WIDTH + dX - FILTER_SIZE/2;
    int input = (inputY * width + inputX) * channels + channel;

    if(inputY >= 0 && inputY < height && inputX >= 0 && inputX < width){
        image_tile[dY*TILE_WIDTH +dX] = input_image[input];
    }
    else{
        image_tile[dY*TILE_WIDTH + dX] = 0;
    }
    // Check that all the threads have completed the upload
    __syncthreads();

    // Upload the remaining pixels in shared memory
    dest = (ty * BLOCK_WIDTH) + tx + BLOCK_WIDTH * BLOCK_WIDTH;
    dX = dest % (BLOCK_WIDTH + FILTER_SIZE -1 );
    dY = dest/ (BLOCK_WIDTH + FILTER_SIZE -1 );;

    inputY = dY + (by * BLOCK_WIDTH) - FILTER_SIZE/2;
    inputX = dX + (bx * BLOCK_WIDTH) - FILTER_SIZE/2;
    input = (inputY * width + inputX) * channels + channel;

    if(dY < (BLOCK_WIDTH + FILTER_SIZE -1 )) {
        if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width) {
            image_tile[dY*TILE_WIDTH + dX] = input_image[input];
        } else {
            image_tile[dY*TILE_WIDTH + dX] = 0;
        }
    }

    // Check that all the threads have completed the upload
    __syncthreads();

    if (col < width && row < height && channel < channels) {
        float pixel_value = 0.0f;

        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                int index = ((ty + i) * TILE_WIDTH + (tx + j)) ;//* channels + channel;

                // Update the pixel value using the shared tile and the global filter
                pixel_value += image_tile[index] * filter[i * FILTER_SIZE + j];
            }
        }

        // Check if the pixel value is in the right range and update the output
        pixel_value = fmaxf(0.0f, fminf(255.0f, pixel_value));
        output_image[(row * width + col) * channels + channel] = (uchar) pixel_value;
    }

}

// CUDA kernel that uses only global memory
__global__ void KernelProcessingGlobal(const uchar* input_image, uchar* output_image, int width, int height, int channels, float* filter) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < width && row < height && channel < channels) {
        float pixel_value = 0.0f;

        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                int neighbor_row = row + i - FILTER_SIZE / 2;
                int neighbor_col = col + j - FILTER_SIZE / 2;

                // Check if one of the neighbor pixel used for convolution is on or over the border
                bool is_border = neighbor_row < 0 || neighbor_row >= height || neighbor_col < 0 || neighbor_col >= width;

                if (!is_border) { //
                    // Update the pixel value using the global input image and the global filter
                    pixel_value += input_image[(neighbor_row * width + neighbor_col) * channels + channel] *
                                   filter[i * FILTER_SIZE + j];
                }
            }
        }
        // Check if the pixel value is in the (0,255) range and update the output pixel
        pixel_value = fmaxf(0.0f, fminf(255.0f, pixel_value));
        output_image[(row * width + col) * channels + channel] = (uchar) pixel_value;
    }
}

// Function that sequentially compute the convolution between an image and a filter
void KernelProcessingSequential(uchar* input_image, uchar* output_image, int width, int height, int channels, const float* filter) {

    // Use three nested loop to iterate sequentially over height, width and channels of the image
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int ch = 0; ch < channels; ch++) {
                float pixel_value = 0.0f;

                for (int i = 0; i < FILTER_SIZE; ++i) {
                    for (int j = 0; j < FILTER_SIZE; ++j) {
                        int neighbor_row = row + i - FILTER_SIZE / 2;
                        int neighbor_col = col + j - FILTER_SIZE / 2;

                        // Check if one of the neighbor pixel used for convolution is on or over the border
                        bool is_border = neighbor_row < 0 || neighbor_row >= height || neighbor_col < 0 || neighbor_col >= width;

                        if (!is_border) {
                            // Update the pixel value
                            pixel_value += input_image[(neighbor_row * width + neighbor_col) * channels + ch] *
                                           filter[i * FILTER_SIZE + j];
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

int main()  {

    std::cout << "Filter size: " << FILTER_SIZE << " block size ("<< BLOCK_WIDTH <<"," << BLOCK_WIDTH << ") " << std::endl;

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Define some useful filters
    const float identity_filter[3][3] = {
            {0, 0, 0},
            {0, 1, 0},
            {0, 0, 0}
    };
    const float sharpen_filter[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}
    };
    const float  edge_detector_filter[3][3] = {
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
    const float unsharp_kernel[5][5] = {
            {-1/256.0f, -4/256.0f, -6/256.0f, -4/256.0f, -1/256.0f},
            {-4/256.0f, -16/256.0f, -24/256.0f, -16/256.0f, -4/256.0f},
            {-6/256.0f, -24/256.0f, 476/256.0f, -24/256.0f, -6/256.0f},
            {-4/256.0f, -16/256.0f, -24/256.0f, -16/256.0f, -4/256.0f},
            {-1/256.0f, -4/256.0f, -6/256.0f, -4/256.0f, -1/256.0f}
    };

    int choosen_filter = FILTER_CHOICE;
    const float * filter;

    switch(choosen_filter){
        case 0:
            filter = (const float*) identity_filter;
            break;
        case 1:
            filter = (const float*) sharpen_filter;
            break;
        case 2:
            filter = (const float*) edge_detector_filter;
            break;
        case 3:
            filter = (const float*) gaussian_filter;
            break;
        case 4:
            filter = (const float*) box_blur_filter;
            break;
        case 5:
            filter = (const float*) unsharp_kernel;
            break;
    }

    std::string folder_path = ".\\Test images";

    // Copy the filter in the constant memory
    cudaMemcpyToSymbol(constant_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    std::vector<std::string> images = {"360p","720p","2K","4K"};

    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {

        std::string file_path = entry.path().string();
        std::string file_name = entry.path().filename().stem().string();

        std::cout << file_name <<": " << std::endl;

        // Load the image using OpenCV
        cv::Mat input_image = cv::imread(file_path, cv::IMREAD_COLOR);

        if (input_image.empty()) {
            std::cerr << "Error: Impossible to load the image " << file_name << std::endl;
            continue;
        }

        std::vector<std::chrono::duration<double>> sequential_durations;

        std::vector<std::vector<std::chrono::duration<double>>> parallel_durations(3);
        std::vector<std::vector<std::chrono::duration<double>>> parallel_durations_with_load(3);

        std::vector<std::vector<std::chrono::duration<double>>> download_times(3);
        std::vector<std::vector<std::chrono::duration<double>>> upload_times(3);

        int width = input_image.cols;
        int height = input_image.rows;
        int channels = input_image.channels();

        // Compute the size of the input/output image
        int size = width * height * channels * sizeof(uchar);


        for (int t = 0; t < TEST_NUM; t++) {

            auto seq_output_image = new uchar[size];
            auto seq_start_time = std::chrono::high_resolution_clock::now();

            // Call the sequential kernel image processing function
            KernelProcessingSequential(input_image.data, seq_output_image, width, height, channels, filter);

            auto seq_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = seq_end_time - seq_start_time;

            sequential_durations.emplace_back(elapsed_seconds);

            // Read the output image
            cv::Mat seq_output_Mat(height, width, CV_8UC(channels), seq_output_image);

            //cv::imwrite(file_name +"_processed.jpg", seq_output_Mat);
            //cv::imshow("Output Image", seq_output_Mat);
            //cv::waitKey(0);

            delete[] seq_output_image;

        }

        //////// CUDA PART ////////

        // Define the pointers for the pinned memory
        uchar* h_input_image;
        uchar* h_output_image;
        float* h_filter;

        // Allocate space in pinned memory
        cudaHostAlloc((void**)&h_input_image, size, cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_output_image, size, cudaHostAllocDefault);
        cudaHostAlloc((void**)&h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaHostAllocDefault);

        // Copy the input image and the filter on the pinned memory
        memcpy(h_input_image, input_image.data, size);
        memcpy(h_filter,filter,FILTER_SIZE * FILTER_SIZE * sizeof(float));

        // Define block and grid size
        dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y,
                      (channels + blockSize.z - 1) / blockSize.z);

        // Define the pointers for the device's objects
        uchar *d_input_image;
        uchar *d_output_image;
        float *d_filter;

        for (int k = 0; k < 3; k++) {
            for (int t = 0; t < TEST_NUM; t++) {

                // Define CUDA events to profile the execution
                cudaEvent_t start, stop, start_upload, stop_upload, start_download, stop_download;

                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventCreate(&start_upload);
                cudaEventCreate(&stop_upload);

                cudaEventCreate(&start_download);
                cudaEventCreate(&stop_download);

                cudaEventRecord(start_upload);

                // Reserve global memory in the GPU for input and output image and the filter
                cudaMalloc((void **) &d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
                cudaMalloc((void **) &d_input_image, size);
                cudaMalloc((void **) &d_output_image, size);

                // Copy the image and the filter from pinned memory to device
                cudaMemcpy(d_input_image, h_input_image, size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

                cudaEventRecord(stop_upload);
                cudaEventSynchronize(stop_upload);

                // Call the cuda kernel for kernel image processing
                if (k == 0) {
                    cudaEventRecord(start);
                    KernelProcessingGlobal<<<gridSize, blockSize>>>(d_input_image,
                                                                    d_output_image, width, height,
                                                                    channels, d_filter);
                    cudaEventRecord(stop);
                } else if (k == 1) {
                    cudaEventRecord(start);
                    KernelProcessingTiled<<<gridSize, blockSize,
                    TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(d_input_image,
                                                               d_output_image, width, height, channels,
                                                               d_filter);
                    cudaEventRecord(stop);
                } else if (k == 2) {
                    cudaEventRecord(start);
                    KernelProcessingTiledConstant<<<gridSize, blockSize,
                    TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(d_input_image,
                                                               d_output_image, width, height, channels);
                    cudaEventRecord(stop);
                }

                cudaEventSynchronize(stop);

                cudaEventRecord(start_download);

                // Copy the output image from the GPU to pinned memory
                cudaMemcpy(h_output_image, d_output_image, size, cudaMemcpyDeviceToHost);

                cudaEventRecord(stop_download);
                cudaEventSynchronize(stop_download);

                // Compute the elapsed time for kernel execution and loads
                float upload_ms = 0;
                cudaEventElapsedTime(&upload_ms, start_upload, stop_upload);
                auto upload_time = upload_ms/1000.0f;

                float download_ms = 0;
                cudaEventElapsedTime(&download_ms, start_download, stop_download);
                auto download_time = download_ms/1000.0f;

                float kernel_ms = 0;
                cudaEventElapsedTime(&kernel_ms, start, stop);
                auto par_elapsed_seconds = kernel_ms/1000.0f;
                auto par_elapsed_seconds_with_load = par_elapsed_seconds + upload_time + download_time;


                parallel_durations[k].emplace_back(par_elapsed_seconds);
                parallel_durations_with_load[k].emplace_back(par_elapsed_seconds_with_load);
                upload_times[k].emplace_back(upload_time);
                download_times[k].emplace_back(download_time);


                cv::Mat par_output_Mat(height, width, CV_8UC(channels), h_output_image);

                //cv::imshow("Output Image", par_output_Mat);
                //cv::waitKey(0);


                // Free the memory on the GPU
                cudaFree(d_input_image);
                cudaFree(d_output_image);
                cudaFree(d_filter);

                // Destroy the events
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                cudaEventDestroy(start_upload);
                cudaEventDestroy(stop_upload);

                cudaEventDestroy(start_download);
                cudaEventDestroy(stop_download);

            }

        }

        // Free the pinned memory
        cudaFreeHost(h_input_image);
        cudaFreeHost(h_output_image);

        // Compute average sequential and parallel execution times and estimate their coefficent of variation
        auto total_seq_time = std::accumulate(sequential_durations.begin(), sequential_durations.end(),
                                              std::chrono::duration<double>(0));

        auto total_par_time = std::accumulate(parallel_durations[0].begin(), parallel_durations[0].end(),
                                              std::chrono::duration<double>(0));
        auto total_par_time_with_load = std::accumulate(parallel_durations_with_load[0].begin(),
                                                        parallel_durations_with_load[0].end(),
                                                        std::chrono::duration<double>(0));
        auto total_download = std::accumulate(download_times[0].begin(),
                                              download_times[0].end(), std::chrono::duration<double>(0));
        auto total_upload = std::accumulate(upload_times[0].begin(),
                                              upload_times[0].end(), std::chrono::duration<double>(0));

        double avg_download_time = total_download.count() / download_times[0].size();
        double avg_upload_time = total_upload.count() / upload_times[0].size();
        
        double avg_seq_time = total_seq_time.count() / sequential_durations.size();
        double avg_par_time = total_par_time.count() /  parallel_durations[0].size();
        double avg_par_time_with_load = total_par_time_with_load.count() / parallel_durations[0].size();

        double speedup = avg_seq_time / avg_par_time;
        double speedup_with_load = avg_seq_time / avg_par_time_with_load;

        double var_seq_time = std::accumulate(sequential_durations.begin(), sequential_durations.end(), 0.0,
                                                   [avg_seq_time](double sum, const std::chrono::duration<double>& time) {
                                                       return sum + std::pow(time.count() - avg_seq_time, 2);
                                                   }) / sequential_durations.size();

        double var_par_time = std::accumulate(parallel_durations[0].begin(), parallel_durations[0].end(), 0.0,
                                                   [avg_par_time](double sum, const std::chrono::duration<double>& time) {
                                                       return sum + std::pow(time.count() - avg_par_time, 2);
                                                   }) / parallel_durations[0].size();

        double variance_par_time_with_load = std::accumulate(parallel_durations_with_load[0].begin(),
                                                             parallel_durations_with_load[0].end(), 0.0,
                                                             [avg_par_time_with_load](double sum, const std::chrono::duration<double>& time) {
                                                                 return sum + std::pow(time.count() - avg_par_time_with_load, 2);
                                                             }) / parallel_durations_with_load[0].size();

        double std_seq_time = std::sqrt(var_seq_time);
        double std_par_time = std::sqrt(var_par_time);
        double std_par_time_with_load = std::sqrt(variance_par_time_with_load);

        std::cout << "Average sequential execution time: " << avg_seq_time << " secs" << std::endl;
        std::cout << "CV sequential execution time: " << std_seq_time/avg_seq_time << "\n" << std::endl;

        std::cout << "Average parallel execution time (global): " << avg_par_time << " secs" << std::endl;
        std::cout << "Average parallel execution time with CUDA load (global): " << avg_par_time_with_load << " secs"
                  << std::endl;
        std::cout << "Average upload time (global): " << avg_upload_time << " secs " << std::endl;
        std::cout << "Average download time (global): " << avg_download_time << " secs " << std::endl;

        std::cout << "CV parallel execution time (global): " << std_par_time/avg_par_time << std::endl;
        std::cout << "CV parallel execution time with CUDA load (global): " << std_par_time_with_load/avg_par_time_with_load << std::endl;

        std::cout << "Speedup (global): " << speedup << std::endl;
        std::cout << "Speedup considering loads (global): " << speedup_with_load << std::endl;


        // Do the same for the tiled version
        auto total_par_time_tiled = std::accumulate(parallel_durations[1].begin(), parallel_durations[1].end(),
                                                    std::chrono::duration<double>(0));
        auto total_par_time_with_load_tiled = std::accumulate(parallel_durations_with_load[1].begin(),
                                                              parallel_durations_with_load[1].end(),
                                                              std::chrono::duration<double>(0));

        auto total_download_tiled = std::accumulate(download_times[1].begin(),
                                              download_times[1].end(), std::chrono::duration<double>(0));
        auto total_upload_tiled = std::accumulate(upload_times[1].begin(),
                                            upload_times[1].end(), std::chrono::duration<double>(0));

        double avg_download_time_tiled = total_download_tiled.count() / download_times[1].size();
        double avg_upload_time_tiled = total_upload_tiled.count() / upload_times[1].size();

        double avg_par_time_tiled = total_par_time_tiled.count() / parallel_durations[1].size();
        double avg_par_time_with_load_tiled = total_par_time_with_load_tiled.count() / parallel_durations_with_load[1].size();

        double var_par_time_tiled = std::accumulate(parallel_durations[1].begin(), parallel_durations[1].end(), 0.0,
                                                   [avg_par_time_tiled](double sum, const std::chrono::duration<double>& time) {
                                                       return sum + std::pow(time.count() - avg_par_time_tiled, 2);
                                                   }) / parallel_durations[1].size();

        double var_par_time_with_load_tiled = std::accumulate(parallel_durations_with_load[1].begin(),
                                                             parallel_durations_with_load[1].end(), 0.0,
                                                             [avg_par_time_with_load_tiled](double sum, const std::chrono::duration<double>& time) {
                                                                 return sum + std::pow(time.count() - avg_par_time_with_load_tiled, 2);
                                                             }) / parallel_durations_with_load[1].size();

        double std_par_time_tiled = std::sqrt(var_par_time_tiled);
        double std_par_time_with_load_tiled = std::sqrt(var_par_time_with_load_tiled);

        double speedup_tiled = avg_seq_time / avg_par_time_tiled;
        double speedup_with_load_tiled = avg_seq_time / avg_par_time_with_load_tiled;

        std::cout << "\nAverage parallel execution time (tiled): " << avg_par_time_tiled << " secs" << std::endl;
        std::cout << "Average parallel execution time with CUDA load (tiled): " << avg_par_time_with_load_tiled
                  << " secs" << std::endl;

        std::cout << "Average upload time (tiled): " << avg_upload_time_tiled << " secs " << std::endl;
        std::cout << "Average download time (tiled): " << avg_download_time_tiled << " secs " << std::endl;

        std::cout << "CV parallel execution time (tiled): " << std_par_time_tiled/avg_par_time_tiled << std::endl;
        std::cout << "CV parallel execution time with CUDA load (tiled): " << std_par_time_with_load_tiled/avg_par_time_with_load_tiled << std::endl;

        std::cout << "Speedup (tiled): " << speedup_tiled << std::endl;
        std::cout << "Speedup considering loads (tiled): " << speedup_with_load_tiled << std::endl;


        // Do the same for the tiled+const version
        auto total_par_time_const = std::accumulate(parallel_durations[2].begin(), parallel_durations[2].end(),
                                                    std::chrono::duration<double>(0));
        auto total_par_time_with_load_const = std::accumulate(parallel_durations_with_load[2].begin(),
                                                              parallel_durations_with_load[2].end(),
                                                              std::chrono::duration<double>(0));
        auto total_download_const = std::accumulate(download_times[2].begin(),
                                              download_times[2].end(), std::chrono::duration<double>(0));
        auto total_upload_const = std::accumulate(upload_times[2].begin(),
                                            upload_times[2].end(), std::chrono::duration<double>(0));

        double avg_download_time_const = total_download_const.count() / download_times[2].size();
        double avg_upload_time_const = total_upload_const.count() / upload_times[2].size();

        double avg_par_time_const = total_par_time_const.count() / parallel_durations[2].size();
        double avg_par_time_with_load_const = total_par_time_with_load_const.count() / parallel_durations_with_load[2].size();

        double var_par_time_const = std::accumulate(parallel_durations[2].begin(), parallel_durations[2].end(), 0.0,
                                                         [avg_par_time_const](double sum, const std::chrono::duration<double>& time) {
                                                             return sum + std::pow(time.count() - avg_par_time_const, 2);
                                                         }) / parallel_durations[2].size();

        double var_par_time_with_load_const = std::accumulate(parallel_durations_with_load[2].begin(),
                                                                   parallel_durations_with_load[2].end(), 0.0,
                                                                   [avg_par_time_with_load_const](double sum, const std::chrono::duration<double>& time) {
                                                                       return sum + std::pow(time.count() - avg_par_time_with_load_const, 2);
                                                                   }) / parallel_durations_with_load[2].size();

        double std_par_time_const = std::sqrt(var_par_time_const);
        double std_par_time_with_load_const = std::sqrt(var_par_time_with_load_const);

        double speedup_const = avg_seq_time / avg_par_time_const;
        double speedup_with_load_const = avg_seq_time / avg_par_time_with_load_const;

        std::cout << "\nAverage parallel execution time (const): " << avg_par_time_const << " secs" << std::endl;
        std::cout << "Average parallel execution time with CUDA load (const): " << avg_par_time_with_load_const
                  << " secs" << std::endl;

        std::cout << "Average upload time (const): " << avg_upload_time_const << " secs " << std::endl;
        std::cout << "Average download time (const): " << avg_download_time_const << " secs " << std::endl;

        std::cout << "CV parallel execution time (const): " << std_par_time_const/avg_par_time_const << std::endl;
        std::cout << "CV parallel execution time with CUDA load (const): " << std_par_time_with_load_const/avg_par_time_with_load_const  << std::endl;

        std::cout << "Speedup (const): " << speedup_const << std::endl;
        std::cout << "Speedup considering loads (const): " << speedup_with_load_const <<"\n" << std::endl;

    }

    return 0;
}

