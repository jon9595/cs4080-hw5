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

    for (int i = 0; i < 262144; i++) {
        //pixels[i] = 0;
    }

    if (sdkSavePGM<unsigned char>(argv[3], pixels, w, h) != true)
    {
        std::cout << "Unable to save PGM image file" << std::endl;
        return 1;
    }

    std::cout << "Successfully saved PGM image file" << std::endl;

    return 0;
}
