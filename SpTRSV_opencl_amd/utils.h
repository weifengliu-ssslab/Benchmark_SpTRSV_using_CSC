#ifndef _UTILS_
#define _UTILS_

#include "common.h"

// print 1D array
void print_1darray(int *input, int length)
{
    for (int i = 0; i < length; i++)
        printf(", %i", input[i]);
    printf("\n");
}


// in-place exclusive scan
void exclusive_scan(int *input, int length)
{
    if(length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}


/*
// segmented sum
template<typename vT, typename bT>
void segmented_sum(vT *input, bT *bit_flag, int length)
{
    if(length == 0 || length == 1)
        return;

    for (int i = 0; i < length; i++)
    {
        if (bit_flag[i])
        {
            int j = i + 1;
            while (!bit_flag[j] && j < length)
            {
                input[i] += input[j];
                j++;
            }
        }
    }
}

// reduce sum
template<typename T>
T reduce_sum(T *input, int length)
{
    if(length == 0)
        return 0;

    T sum = 0;

    for (int i = 0; i < length; i++)
    {
        sum += input[i];
    }

    return sum;
}*/

#endif
