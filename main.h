#ifndef __MAIN_H__
#define __MAIN_H__

void processImageWithCPU(unsigned char *, unsigned int, unsigned int, unsigned int);
__global__ void processImageWithGPU(unsigned char *, unsigned char*, unsigned int *, unsigned int *, unsigned int *);
void compareOutput(unsigned char *, unsigned char *, int);

#endif
