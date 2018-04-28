#include <stdio.h>
#include <string>

#include "lib/helper_image.h"

unsigned char* goldenstandard(unsigned char* pixels) {
    unsigned char* output = NULL;

    return output;
}

int main(int argc, char** argv) {

    if(argc != 4) {
        std::cerr << "Usage ./homework_5 <filter_size> <input_file> <output_file>" << std::endl;
        exit(1);
    }

    int filter_size = std::stoi(argv[1]);
    int radius = filter_size / 2;

    unsigned int w, h;
    unsigned char* pixels = NULL;

    if (sdkLoadPGM<unsigned char>(argv[2], &pixels, &w, &h) != true)
    {
        std::cout << "Unable to load PGM image file" << std::endl;
        exit(1);
    }

    // Iterate through pixels
	for (int row = 0; row < w; row++) {
        for (int col = 0; col < h; col++) {
            // Index of current pixel
            int pixelsIndex = row * w + col;

            // Pixel values inside the filter
            std::vector<unsigned char> values;

            // Iterate through filter in quadrants
            for (int filterRow = 0; filterRow <= radius; filterRow++) {
                for (int filterCol = 0; filterCol <= radius; filterCol++) {
                    // Grab indices for top rows of filter (including middle row)
                    int filterIndexTopLeft = pixelsIndex - (w * filterRow) - filterCol;

                    // Check that index doesn't overflow left or top edge
                    if ((filterIndexTopLeft >= (row * w)) && (filterIndexTopLeft >= 0)) {
                        values.push_back(pixels[filterIndexTopLeft]);
                    }

                    // Don't duplicate middle col of filter
                    if (filterCol != 0) {
                        int filterIndexTopRight = pixelsIndex - (w * filterRow) + filterCol;

                        // Check that index doesn't overflow right or top edge or filter
                        if ((filterIndexTopRight <= (((row + 1) * w) - 1)) && filterIndexTopRight >= 0 && filterIndexTopRight > (pixelsIndex - (w * filterRow))) {
                            values.push_back(pixels[filterIndexTopRight]);
                        }
                    }

                    // Grab indices for bottom half of filter (excluding middle row)
                    if (filterRow != 0) {
                        int filterIndexBottomLeft = pixelsIndex + (w * filterRow) - filterCol;

                        // Check that index doesn't overflow left or bottom edge
                        if ((filterIndexBottomLeft >= ((row + 1) * w)) && (filterIndexBottomLeft < (w * h))) {
                            values.push_back(pixels[filterIndexBottomLeft]);
                        }

                        // Don't duplicate middle col of filter
                        if (filterCol != 0) {
                            int filterIndexBottomRight = pixelsIndex + (w * filterRow) + filterCol;

                            // Check that index doesn't overflow right or bottom edge
                            if ((filterIndexBottomRight > (((row + 1) * w) - 1)) && filterIndexBottomRight < w * h) {
                                values.push_back(pixels[filterIndexBottomRight]);
                            }
                        }
                    }
                }
            }

            std::sort(values.begin(), values.end());

            int size = values.size();

            std::cout << "row " << row << "\tcol " << col << "\tsize " << size << std::endl;

            pixels[pixelsIndex] = values[size / 2];
        }
    }

    if (sdkSavePGM<unsigned char>(argv[3], pixels, w, h) != true)
    {
        std::cerr << "Unable to save PGM image file" << std::endl;
        exit(1);
    }

    return 0;
}
