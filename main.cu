#include <stdio.h>
#include <string>

#include "cuda_runtime.h"
#include "lib/helper_image.h"

unsigned char* goldenstandard(unsigned char* pixels) {
    unsigned char* output = NULL;

    return output;
}

int main(int argc, char** argv) {

    if(argc != 4) {
        std::cout << "Usage ./homework_5 <filter_size> <input_file> <output_file>" << std::endl;
        exit(1);
    }

    int filter_size = std::stoi(argv[1]);

    unsigned int w, h;
    unsigned char* pixels = NULL;

    if (sdkLoadPGM<unsigned char>(argv[2], &pixels, &w, &h) != true)
    {
        std::cout << "Unable to load PGM image file" << std::endl;
        return 1;
    }

    std::cout << "Successfully loaded PGM image file" << std::endl;

        // First, create a 3x3 filter
	for (int i = 0; i < 512; i++) {
        for (int j = 0; j < 512; j++) {
            if (i > 0 && j > 0 && i < 511 & j < 511) {
                std::vector<unsigned char> v;

                // Index of current pixel
                int k = i * 512 + j;

                // Top row of neighborhood
                v.push_back(pixels[k - 513]);
                v.push_back(pixels[k - 512]);
                v.push_back(pixels[k - 511]);

                // Middle row of neighborhood
                v.push_back(pixels[k - 1]);
                v.push_back(pixels[k]);
                v.push_back(pixels[k + 1]);

                // Bottom row of neighborhood
                v.push_back(pixels[k + 511]);
                v.push_back(pixels[k + 512]);
                v.push_back(pixels[k + 513]);

                std::sort(v.begin(), v.end());

                int size = v.size();
                pixels[k] = v[size / 2];
            }
        }
    }

    if (sdkSavePGM<unsigned char>(argv[3], pixels, w, h) != true)
    {
        std::cout << "Unable to save PGM image file" << std::endl;
        return 1;
    }

    std::cout << "Successfully saved PGM image file" << std::endl;

    return 0;
}
