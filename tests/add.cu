extern "C" __global__ void add(char * a, char * b) {
    a[threadIdx.x] += b[threadIdx.x];
}
