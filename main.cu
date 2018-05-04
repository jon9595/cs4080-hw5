#include <chrono>
#include <stdio.h>
#include <string>
#include "main.h"
#include "lib/helper_cuda.h"
#include "lib/helper_functions.h"
#include "lib/helper_image.h"
#include "cuda_runtime.h"

// Assert macro to check for CUDA errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr, "GPUassert:: %s %s %d\n", cudaGetErrorString(code), file, line);
        
        if (abort) 
            exit(code);
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage ./homework5 <filter_size> <input_file> <output_file>" << std::endl;
        exit(1);
    }

    unsigned int filter_size = std::stoi(argv[1]);
    unsigned int radius = filter_size / 2;

    unsigned char* pixels = NULL;
    unsigned int w, h;

    // Load input file into data buffer
    std::chrono::high_resolution_clock c;
    std::chrono::high_resolution_clock::time_point file_load_start = c.now();

    if (sdkLoadPGM<unsigned char>(argv[2], &pixels, &w, &h) != true)
    {
        std::cerr << "Unable to load file " << argv[2] << std::endl;
        return 1;
    }

    std::chrono::high_resolution_clock::time_point file_load_stop = c.now();
    const double file_load_time = (double)std::chrono::duration_cast<std::chrono::microseconds>(file_load_stop - file_load_start).count() / 1000000.0;
    std::cout << "File loaded in " << file_load_time << "s" << std::endl;

    // Begin timer to capture the device copy-compute-copy time
    std::chrono::high_resolution_clock::time_point startTime = c.now();

    size_t vectorSize = sizeof(unsigned char) * w * h;

    // Allocate vector in device memory
    unsigned char* d_pixels;

    gpuErrchk(cudaMalloc((void**) &d_pixels, vectorSize));

    // Allocate other data in device memory
    unsigned int *d_w, *d_h, *d_radius;
    size_t uintSize = sizeof(unsigned int);
    gpuErrchk(cudaMalloc(&d_w, uintSize));
    gpuErrchk(cudaMalloc(&d_h, uintSize));
    gpuErrchk(cudaMalloc(&d_radius, uintSize));

    // Copy data from host memory to device memory
    gpuErrchk(cudaMemcpy(d_pixels, pixels, vectorSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, &w, uintSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_h, &h, uintSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_radius, &radius, uintSize, cudaMemcpyHostToDevice));

    // Invoke kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = 256;

    // Begin timing
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventRecord(start, NULL));
    processImageWithGPU<<<blocksPerGrid,threadsPerBlock>>>(d_pixels, d_w, d_h, d_radius);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result from device memory to host memory
    unsigned char* h_pixels = (unsigned char*) malloc(vectorSize);
    gpuErrchk(cudaMemcpy(h_pixels, d_pixels, vectorSize, cudaMemcpyDeviceToHost));

    // End timer to capture the device copy-compute-copy time
    std::chrono::high_resolution_clock::time_point stopTime = c.now();
    double time = (double) std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count() / 1000000.0;
    std::cout << "Image copied to device, processed, and copied back from device in " << time << "s" << std::endl;

    // Record end event
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Free device memory
    cudaFree(d_pixels);

    // Begin timer to capture host processing time
    startTime = c.now();

    // Generate golden standard version with CPU
    processImageWithCPU(pixels, w, h, radius);

    // End timer to capture host processing time
    stopTime = c.now();
    time = (double) std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count() / 1000000.0;
    std::cout << "Image processed on CPU in " << time << "s" << std::endl;

    // Save buffer to PGM image file
    if (sdkSavePGM<unsigned char>(argv[3], h_pixels, w, h) != true)
    {
        std::cerr << "Unable to save file " << argv[3] << std::endl;
        exit(1);
    }

    // Free host memory
    free(h_pixels);

    return 0;
}

__global__ void processImageWithGPU(unsigned char* pixels, unsigned int* w, unsigned int* h, unsigned int* radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = i / *w;
    int col = i % *w;
    unsigned char values[150];
    int idx = 0;

    if(row > *radius && col > *radius && (*h - row) > *radius && (*w - col) > *radius) {
        // We are ignoring the edge cases in this if-statement, so process neighbor pixels
        for(int filterRow = 0; filterRow <= *radius; filterRow++) {
            for(int filterCol = 0; filterCol <= *radius; filterCol++) {
                int filterIndexTopLeft = i - (*w * filterRow) - filterCol;
                values[idx] = pixels[filterIndexTopLeft];
                idx++;

                if(filterCol != 0) {
                    int filterIndexTopRight = i - (*w * filterRow) + filterCol;
                    values[idx] = pixels[filterIndexTopRight];
                    idx++;
                }

                if(filterRow != 0) {
                    int filterIndexBottomLeft = i + (*w * filterRow) - filterCol;
                    values[idx] = pixels[filterIndexBottomLeft];
                    idx++;

                    if(filterCol != 0) {
                        int filterIndexBottomRight = i + (*w + filterRow) + filterCol;
                        values[idx] = pixels[filterIndexBottomRight];
                        idx++;
                    }
                }
            }
        }

        // Sort the pixels
        for(int j = 0; j < idx - 1; j++) {
            for(int k = 0; k < idx - j - 1; k++) {
                if(values[k] > values[k + 1]) {
                    unsigned char temp = values[k];
                    values[k] = values[k+1];
                    values[k+1] = temp;
                }
            }
        }

        // Change the pixel to the median value
        pixels[i] = values[idx/2];
    }
}

void processImageWithCPU(unsigned char* pixels, unsigned int w, unsigned int h, unsigned int radius)
{
    // Iterate through pixels
    for (int row = 0; row < w; row++) {
        for (int col = 0; col < h; col++) {
            // Index of current pixel to be processed
            int pixelIndex = row * w + col;

            // Pixel values inside the filter
            std::vector<unsigned char> values;

            // Iterate through filter in quadrants
            for (int filterRow = 0; filterRow <= radius; filterRow++) {
                for (int filterCol = 0; filterCol <= radius; filterCol++) {
                    // Grab indices for top rows of filter (including middle row)
                    int filterIndexTopLeft = pixelIndex - (w * filterRow) - filterCol;

                    // Check that index doesn't overflow left or top edge
                    if ((filterIndexTopLeft >= ((row - filterRow) * w)) && (filterIndexTopLeft >= 0)) {
                        values.push_back(pixels[filterIndexTopLeft]);
                    }

                    // Don't duplicate middle col of filter
                    if (filterCol != 0) {
                        int filterIndexTopRight = pixelIndex - (w * filterRow) + filterCol;

                        // Check that index doesn't overflow right or top edge
                        if (filterIndexTopRight <= (((row - filterRow + 1) * w) - 1) && filterIndexTopRight > (pixelIndex - (w * filterRow)) && filterIndexTopRight >= 0) {
                            values.push_back(pixels[filterIndexTopRight]);
                        }
                    }

                    // Grab indices for bottom half of filter (excluding middle row)
                    if (filterRow != 0) {
                        int filterIndexBottomLeft = pixelIndex + (w * filterRow) - filterCol;

                        // Check that index doesn't overflow left or bottom edge
                        if ((filterIndexBottomLeft >= ((row + filterRow) * w)) && (filterIndexBottomLeft < (w * h))) {
                            values.push_back(pixels[filterIndexBottomLeft]);
                        }

                        // Don't duplicate middle col of filter
                        if (filterCol != 0) {
                            int filterIndexBottomRight = pixelIndex + (w * filterRow) + filterCol;

                            // Check that index doesn't overflow right or bottom edge
                            if (filterIndexBottomRight <= ((row + filterRow + 1) * w - 1) && filterIndexBottomRight < (w * h)) {
                                values.push_back(pixels[filterIndexBottomRight]);
                            }
                        }
                    }
                }
            }

            std::sort(values.begin(), values.end());
            int size = (int) values.size();
            pixels[pixelIndex] = values[size / 2];
        }
    }
}
