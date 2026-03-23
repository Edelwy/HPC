#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>
#include <numa.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

#define TOTALTIME

void copy_image(unsigned char *image_out, const unsigned char *image_in, const size_t size)
{

    #pragma omp parallel
    {
        // Print thread, CPU, and NUMA node information
        #pragma omp single
        printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);

        #pragma omp critical
        printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);

        // Copy the image data in parallel
        #pragma omp for
        for (size_t i = 0; i < size; ++i)
        {
            image_out[i] = image_in[i];
        }
    }

}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image [seams] \n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    int seams = 128;
    if(argc == 4){
        seams = atoi(argv[3]);
    }

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    int real_width = width;

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, width, height, cpp);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    if (image_out == NULL) {
        printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    // Copy the input image into output and mesure execution time
    #ifdef TOTALTIME
    double total_start = omp_get_wtime();
    #endif
    double start = omp_get_wtime();
    copy_image(image_out, image_in, datasize);
    double stop = omp_get_wtime();
    printf("Time to copy: %f s\n", stop - start);

    float *energy_map = malloc(width * height * sizeof(float));
    float *M = malloc(width * height * sizeof(float));
    int *seam = malloc(height * sizeof(int));


//int seams=128;
for(int reps=0; reps<seams; reps++){

    // BEGIN THE MAGIC
/*	TEMPLATE:
    start = omp_get_wtime();

    stop = omp_get_wtime();
    #ifdef STATS
    printf("Step 1 time: %f s\n", stop - start);
    #endif
*/


    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    // Step1: sobel

    for (int i = 0; i < width * height; i++) {
        energy_map[i] = 9999999999;	//not zero :)
    }

    start = omp_get_wtime();

    for (int i = 1; i < real_width - 1; i++) {
        for (int j = 1; j < height - 1; j++) {

            float energy = 0.0f;

            for (int c = 0; c < cpp; c++) {
                int gx = 0;
                int gy = 0;

                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {

                        int x = i + dx;
                        int y = j + dy;

                        int idx = (y * width + x) * cpp + c;
                        int pixel = image_out[idx];

                        gx += pixel * Gx[dx + 1][dy + 1];
                        gy += pixel * Gy[dx + 1][dy + 1];
                    }
                }

                energy += gx * gx + gy * gy;
            }

             energy_map[j * width + i] = energy;
        }
    }

    stop = omp_get_wtime();
    #ifdef STATS
    printf("Step 1 time: %f s\n", stop - start);
    #endif


    // STEP 2: seam finding
    start = omp_get_wtime();

    for (int i = 0; i < real_width; i++) {
        M[(height - 1) * width + i] = energy_map[(height - 1) * width + i];
    }

    for (int j = height - 2; j >= 0; j--) {

        for (int i = 0; i < real_width; i++) {

            float down_left  = (i > 0) ? M[(j + 1) * width + (i - 1)] : 1e9;
            float down       = M[(j + 1) * width + i];
            float down_right = (i < real_width - 1) ? M[(j + 1) * width + (i + 1)] : 1e9;

            float min_below = down;
            if (down_left < min_below)  min_below = down_left;
            if (down_right < min_below) min_below = down_right;

            M[j * width + i] = energy_map[j * width + i] + min_below;
            #ifdef DEBUG
            printf("%f\n", M[j * width + i]);
            #endif
        }
    }

    stop = omp_get_wtime();
    #ifdef STATS
    printf("Step 2 time: %f s\n", stop - start);
    #endif


    //STEP 3:

    start = omp_get_wtime();

    int min_col = 0;
    float min_val = M[0];

    for (int i = 1; i < real_width; i++) {
        if (M[i] < min_val) {
            min_val = M[i];
            min_col = i;
        }
    }

    seam[0] = min_col;

    for (int j = 1; j < height; j++) {
        int prev = seam[j - 1];

        int best_col = prev;
        float best_val = M[j * width + prev];

        // levi
        if (prev > 0) {
            float val = M[j * width + (prev - 1)];
            if (val < best_val) {
                best_val = val;
                best_col = prev - 1;
            }
        }

        // desni
        if (prev < real_width - 1) {
            float val = M[j * width + (prev + 1)];
            if (val < best_val) {
                best_val = val;
                best_col = prev + 1;
            }
        }

        seam[j] = best_col;
        #ifdef DEBUG
        printf("%d\n", best_col);
        #endif
    }

    for (int j = 0; j < height; j++) {
        int col = seam[j];

        for (int i = col; i < real_width - 1; i++) {
            for (int c = 0; c < cpp; c++) {

                int idx  = (j * width + i) * cpp + c;
                int idx2 = (j * width + (i + 1)) * cpp + c;

                image_out[idx] = image_out[idx2];
            }
        }
    }
    real_width--;
    stop = omp_get_wtime();
    #ifdef STATS
    printf("Step 3 time: %f s\n", stop - start);
    #endif

    // END OF MAGIC
}

    #ifdef TOTALTIME
    double total_stop = omp_get_wtime();
    printf("Total time: %f s\n", total_stop - total_start);
    #endif

    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found!\n");
        stbi_image_free(image_in);
        stbi_image_free(image_out);
        exit(EXIT_FAILURE);
    }
    file_type++; // skip the dot

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, real_width, height, cpp, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, real_width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, real_width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);
    stbi_image_free(image_out);
    free(energy_map);
    free(M);
    free(seam);

    return 0;
}
