const int BLOCK_WIDTH = 8;

#if BWX32
const int BLOCK_WX = 32;
#elif BWX24
const int BLOCK_WX = 24;
#elif BWX16
const int BLOCK_WX = 16;
#elif BWX8
const int BLOCK_WX = 8;
#else
const int BLOCK_WX = 4;
#endif

#if BWY32
const int BLOCK_WY = 32;
#elif BWY24
const int BLOCK_WY = 24;
#elif BWY16
const int BLOCK_WY = 16;
#elif BWY8
const int BLOCK_WY = 8;
#else
const int BLOCK_WY = 4;
#endif
