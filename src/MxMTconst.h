/**
 * Some compiling flag for generating
 * different cuda thread block sizes
 */
#if BW128
const int BLOCK_WIDTH = 128;
#elif BW64
const int BLOCK_WIDTH = 64;
#elif BW32
const int BLOCK_WIDTH = 32;
#elif BW24
const int BLOCK_WIDTH = 24;
#elif BW16
const int BLOCK_WIDTH = 16;
#elif BW8
const int BLOCK_WIDTH = 8;
#elif BW4
const int BLOCK_WIDTH = 4;
#else
const int BLOCK_WIDTH = 2;
#endif
