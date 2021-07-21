#ifndef NUM_BLOCKS 
#define NUM_BLOCKS 128
#endif 


#ifndef NUM_THREADS  
#define NUM_THREADS 128 
#endif 


#ifndef NUM_BINS
#define NUM_BINS 256
#endif


#ifndef CLAMP 
#define CLAMP(x, a, b) { \
    if (x < (a)) {x = (a);} \
    if (x >= (b)) {x = ((b) - 1);} \
}
#endif 
