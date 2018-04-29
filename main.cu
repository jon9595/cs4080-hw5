#include <stdio.h>
#include <string>

#include "main.h"
#include "lib/helper_image.h"
#include "cuda_runtime.h"

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
	if (sdkLoadPGM<unsigned char>(argv[2], &pixels, &w, &h) != true)
	{
		std::cerr << "Unable to load file " << argv[2] << std::endl;
		return 1;
	}

	// Allocate vector in device memory
	unsigned char* d_pixels;
	size_t vectorSize = sizeof(unsigned char) * w * h;
	cudaMalloc(&d_pixels, vectorSize);

	// Allocate other data in device memory
	unsigned int *d_w, *d_h, *d_radius;
	size_t uintSize = sizeof(unsigned int);
	cudaMalloc(&d_w, uintSize);
	cudaMalloc(&d_h, uintSize);
	cudaMalloc(&d_radius, uintSize);

	// Copy data from host memory to device memory
	cudaMemcpy(d_pixels, pixels, vectorSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, &w, uintSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, &h, uintSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_radius, &radius, uintSize, cudaMemcpyHostToDevice);

	// Invoke kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = 256;
	//processImageWithGPU<<<blocksPerGrid, threadsPerBlock>>>(d_pixels, d_w, d_h, d_radius);

	// Copy result from device memory to host memory
	unsigned char* h_pixels = (unsigned char*) malloc(vectorSize);
	cudaMemcpy(d_pixels, h_pixels, vectorSize, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_pixels);

	// Generate golden standard version with CPU
	processImageWithCPU(pixels, w, h, radius);

	// Save buffer to PGM image file
	if (sdkSavePGM<unsigned char>(argv[3], pixels, w, h) != true)
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
	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	//pixels[i] = 0;
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
