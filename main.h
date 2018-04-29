#ifndef __KERNEL_H__
#define __KERNEL_H__

void processImageWithCPU(unsigned char*, unsigned int, unsigned int, unsigned int);
__global__ void processImageWithGPU(unsigned char*, unsigned int*, unsigned int*, unsigned int*);

#endif
