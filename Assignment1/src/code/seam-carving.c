#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <numa.h>
#include <float.h>
#include <limits.h>
#include <omp.h>


#ifdef USE_OMP_OPTIMIZED
#define USE_OMP
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels.
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

// Marcros for default values.
#define DEFAULT_SEEMS 180
#define TOTALTIME

#ifdef USE_OMP_OPTIMIZED
#define THREADS_ENERGY_CONSTANT   20000
#define THREADS_DP_CONSTANT       2000
#define THREADS_REMOVAL_CONSTANT  2000

static int omp_threads_energy(int max_threads, int width, int height)
{
    int work = width * height;
    int threads = work / THREADS_ENERGY_CONSTANT;
    if (threads > max_threads) threads = max_threads;
    if (threads < 1) threads = 1;

    return threads;
}

static int omp_threads_dp(int max_threads, int width)
{
    int threads = width / THREADS_DP_CONSTANT;
    if (threads < 1) threads = 1;
    if (threads > max_threads / 2) threads = max_threads / 2;
    if (threads < 1) threads = 1;
    return threads;
}

static int omp_threads_removal(int max_threads, int height)
{
    int threads = height / THREADS_REMOVAL_CONSTANT;
    if (threads > max_threads / 2)
        threads = max_threads / 2;
    if (threads < 1) threads = 1;
    return threads;
}
#endif

void copy_image(unsigned char *image_out, const unsigned char *image_in, const size_t size)
{
#ifdef USE_OMP
    #pragma omp parallel
#endif
    {
        // Print thread, CPU, and NUMA node information.
#ifdef USE_OMP
        #pragma omp single
#endif
        printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);

#ifdef USE_OMP
        #pragma omp critical
#endif
        printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);

        // Copy the image data in parallel.
#ifdef USE_OMP
        #pragma omp for
#endif
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

    int seams = DEFAULT_SEEMS;
    if(argc == 4)
    {
        seams = atoi(argv[3]);
    }

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    // Load image from file.
    int width, height, cpp; // Original number of channels in the file.
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, width, height, cpp);
    int color_channels = COLOR_CHANNELS;
    if (color_channels == 0)
        color_channels = cpp;

    if(seams >= width) {
        printf("Seams size too large, fallback on default: %d\n", DEFAULT_SEEMS);
        seams = DEFAULT_SEEMS;
    }

    // Allocate space for the output image.
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    if (image_out == NULL) 
    {
        printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    // Start measuring total time.
#ifdef TOTALTIME
    double total_start = omp_get_wtime();
#endif

    // Measuring copy time.
    double start = omp_get_wtime();
    copy_image(image_out, image_in, datasize);
    double stop = omp_get_wtime();
    printf("Time to copy: %f s\n", stop - start);

    // Allocate memory for the energy map, cumulative energy matrix, and seams.
    float *energy_map = malloc(width * height * sizeof(float));
    float *M = malloc(width * height * sizeof(float));
    int *seam = malloc(height * sizeof(int));

// Set the upper bound of threads for the OpenMP optimization.
#ifdef USE_OMP_OPTIMIZED
    int max_threads = omp_get_max_threads();
#endif

    // BEGIN THE MAGIC.
    int new_width = width;
    for(int reps = 0; reps < seams; reps++)
    {
#ifdef USE_OMP_OPTIMIZED
        int t_energy = omp_threads_energy(max_threads, new_width, height);
        int t_dp = omp_threads_dp(max_threads, new_width);
        int t_removal = omp_threads_removal(max_threads, height);
#endif

        // Frame around pixel in the X direction.
        int Gx[3][3] = 
        {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        // Frame around pixel in the Y direction.
        int Gy[3][3] = 
        {
            { 1, 2, 1},
            { 0, 0, 0},
            {-1,-2,-1}
        };

        // Energy map calculation per channel and then using the average as the value.
        start = omp_get_wtime();
#ifdef USE_OMP_OPTIMIZED
        omp_set_num_threads(t_energy);
#endif
#ifdef USE_OMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < new_width; i++) 
        {
            for (int j = 0; j < height; j++) 
            {
                float energy = 0.0f;
                for (int c = 0; c < color_channels; c++) 
                {
                    // Looping over the frame.
                    int gx = 0;
                    int gy = 0;
                    for (int dx = -1; dx <= 1; dx++) 
                    {
                        for (int dy = -1; dy <= 1; dy++) 
                        {
                            int x = i + dx;
                            int y = j + dy;

                            // Handling border cases: use the value of the closest pixel. 
                            if (x < 0) x = 0;
                            if (x >= new_width) x = new_width - 1;
                            if (y < 0) y = 0;
                            if (y >= height) y = height - 1;

                            int idx = (y * width + x) * color_channels + c;
                            int pixel = image_out[idx];

                            gx += pixel * Gx[dx + 1][dy + 1];
                            gy += pixel * Gy[dx + 1][dy + 1];
                        }
                    }
                    // Sum energy (instead of average) per color channel.
                    energy += gx * gx + gy * gy;
                }
                energy_map[j * width + i] = energy;
            }
        }
        stop = omp_get_wtime();
#ifdef STATS
        printf("Sobel filter took: %f s\n", stop - start);
#endif

        // Now we find the seams.
        start = omp_get_wtime();
#ifdef USE_OMP_OPTIMIZED
        omp_set_num_threads(t_dp);
#endif
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int i = 0; i < new_width; i++) 
        {
            // Use the the cumulative energy matrix used for dynamic programming.
            // This is the bottom row which is fixed.
            M[(height - 1) * width + i] = energy_map[(height - 1) * width + i];
        }

        // Moving up the image finding minimal adjecent neighbour below.
        for (int j = height - 2; j >= 0; j--) 
        {
#ifdef USE_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < new_width; i++)
            {
                float down_left  = (i > 0) ? M[(j + 1) * width + (i - 1)] : FLT_MAX;
                float down       = M[(j + 1) * width + i];
                float down_right = (i < new_width - 1) ? M[(j + 1) * width + (i + 1)] : FLT_MAX;

                float min_below = down;
                if (down_left < min_below)  min_below = down_left;
                if (down_right < min_below) min_below = down_right;

                M[j * width + i] = energy_map[j * width + i] + min_below;
            }
        }
        stop = omp_get_wtime();
#ifdef STATS
        printf("Dynamic seam finding took: %f s\n", stop - start);
#endif

        // Now we remove the minimal value seams based on the number of seams.
        start = omp_get_wtime();
        int min_col = 0;
        float min_val = M[0];

        // Find the minimal column to start the seam.
        for (int i = 1; i < new_width; i++) 
        {
            if (M[i] < min_val) 
            {
                min_val = M[i];
                min_col = i;
            }
        }

        // Find the path down and save the seam into seams array.
        seam[0] = min_col;
        for (int j = 1; j < height; j++)
        {
            int prev = seam[j - 1];
            int best_col = prev;
            float best_val = M[j * width + prev];

            // Left adjecent.
            if (prev > 0) 
            {
                float val = M[j * width + (prev - 1)];
                if (val < best_val) 
                {
                    best_val = val;
                    best_col = prev - 1;
                }
            }

            // Right adjecent.
            if (prev < new_width - 1) 
            {
                float val = M[j * width + (prev + 1)];
                if (val < best_val) 
                {
                    best_val = val;
                    best_col = prev + 1;
                }
            }
            seam[j] = best_col;
        }

        // Delete the seam from the image.
#ifdef USE_OMP_OPTIMIZED
        omp_set_num_threads(t_removal);
#endif
#ifdef USE_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < height; j++) 
        {
            int col = seam[j];
            for (int i = col; i < new_width - 1; i++) 
            {
                for (int c = 0; c < cpp; c++) 
                {
                    int idx  = (j * width + i) * cpp + c;
                    int idx2 = (j * width + (i + 1)) * cpp + c;
                    image_out[idx] = image_out[idx2];
                }
            }
        }

        new_width--;
        stop = omp_get_wtime();
#ifdef STATS
        printf("Seam finding and deletion took: %f s\n", stop - start);
#endif
    }
    // END OF MAGIC.
#ifdef TOTALTIME
    double total_stop = omp_get_wtime();
    printf("Total time: %f s\n", total_stop - total_start);
#endif

    // Write the output image to file.
    char image_out_name_temp[MAX_FILENAME];
    strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) 
    {
        printf("Error: No file extension found!\n");
        stbi_image_free(image_in);
        stbi_image_free(image_out);
        exit(EXIT_FAILURE);
    }
    file_type++; // Skip the dot.

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, new_width, height, color_channels, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, new_width, height, color_channels, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, new_width, height, color_channels, image_out);
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory.
    free(energy_map);
    free(M);
    free(seam);
    stbi_image_free(image_in);
    stbi_image_free(image_out);
    return 0;
}
