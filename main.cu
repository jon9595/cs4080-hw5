#include <stdio.h>

#include "cuda_runtime.h"
#include "lib/helper_image.h"

int main() {
	unsigned int w, h;
	unsigned char* pixels = NULL;

	if (sdkLoadPGM<unsigned char>("../lena.pgm", &pixels, &w, &h) != true)
	{
	  std::cout << "Unable to load PGM image file" << std::endl;
	  return 1;
	}

	std::cout << "Successfully loaded PGM image file" << std::endl;

	for (int i = 0; i < 262144; i++) {
	  //pixels[i] = 0;
	}

	if (sdkSavePGM<unsigned char>("../lena2.pgm", pixels, w, h) != true)
	{
	  std::cout << "Unable to save PGM image file" << std::endl;
	  return 1;
	}

	std::cout << "Successfully saved PGM image file" << std::endl;
	
	return 0;
}
