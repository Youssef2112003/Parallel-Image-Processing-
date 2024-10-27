#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mpi.h>
using namespace std;
using namespace cv;
string input_image()
{
    char input_img[100];
    cout << "\n\nPlease enter the filename of the input image (e.g., input.jpg):";
    cin >> input_img;
    cout << "\n\n";
    return input_img;
}
string output_image()
{
    string output_img;
    cout << "Please enter the filename for the output blurred image (e.g., output.jpg):";
    cin >> output_img;
    cout << "\n\n";
    return output_img;
}
void print()
{
    cout << "Welcome to parallel image processing with MPI\n\n\n\n"
        << "Please choose an image processing operation:\n\n"
        << "01- Gaussian Blur\n\n"
        << "02- Edge Detection\n\n"
        << "03- Image Rotation\n\n"
        << "04- Image Scaling\n\n"
        << "05- Histogram Equalization\n\n"
        << "06- Color Space Conversion\n\n"
        << "07- Global Thresholding\n\n"
        << "08- Local Thresholding\n\n"
        << "09- Image Compression\n\n"
        << "10- Median\n\n"
        << "Enter your choice(1-10):";
}
Mat readImage(const string& filePath) {
    ifstream file(filePath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Unable to open image file" << endl;
        return Mat();
    }
    vector<unsigned char> buffer(istreambuf_iterator<char>(file), {});
    Mat image = imdecode(buffer, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Unable to decode image" << endl;
        return Mat();
    }
    return image;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size,choice;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    string output_img;
    double start_time, end_time;
    Mat image;
    cout << "\n\n";
    
    if (rank == 0) {
        print();
        cin >> choice;
        image = readImage(input_image());
        if (image.empty()) {
            printf("Error: Unable to load image\n");
            MPI_Finalize();
            return -1;
        }
        output_img = output_image();
    }
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int rows, cols;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&output_img, 1, MPI_CHAR, 0, MPI_COMM_WORLD);


    // Divide rows among processes
    int rows_per_process = rows / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? rows : start_row + rows_per_process;

    // Allocate memory for the local chunk of the image
    Mat local_image(end_row - start_row, cols, CV_8UC3);
    // Scatter image data to all processes
    MPI_Scatter(image.data, local_image.rows * cols * 3, MPI_UNSIGNED_CHAR,
        local_image.data, local_image.rows * cols * 3, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);
    int blur_radius=0;
    if (rank == 0 && choice == 1)
    {
        cout << "Please enter the blur radius(e.g., 3) :";
        cin >> blur_radius;
        cout << "\n\n";
    }
    MPI_Bcast(&blur_radius, 1, MPI_INT, 0, MPI_COMM_WORLD);
















    if (choice == 1) {
        double start_time, end_time;
        cout << "Processing image with Gaussian Blur...\n\n";
        start_time = MPI_Wtime();

        // Disable parallel execution in OpenCV
        cv::setUseOptimized(false);
        cv::setNumThreads(1);

        MPI_Bcast(&blur_radius, 1, MPI_INT, 0, MPI_COMM_WORLD);
        GaussianBlur(local_image, local_image, Size(blur_radius, blur_radius), 0);
        string output_filename = "blur_rank" + to_string(rank) + ".jpg";
        imwrite(output_filename, local_image);
        Mat final_image;
        if (rank == 0) {
            final_image.create(rows, cols, CV_8UC3);
        }
        MPI_Gather(local_image.data, rows_per_process * cols * 3, MPI_UNSIGNED_CHAR,
            final_image.data, rows_per_process * cols * 3, MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);
        if (rank == 0) {
            imwrite(output_img, final_image);
            end_time = MPI_Wtime();
            cout << "Gaussian Blur operation completed successfully in " << end_time - start_time << " seconds.\n\n"
                << "Blurred image saved as output_image.jpg.\n\n"
                << "Thank you for using Parallel Image Processing with MPI.\n\n";
        }
        MPI_Finalize();
    }
    else if (choice == 2) {
        double start_time, end_time;
        start_time = MPI_Wtime();

        // Perform edge detection on local image
        Mat edges;
        Sobel(local_image, edges, CV_16S, 1, 0); // Compute gradient along x-axis
        convertScaleAbs(edges, edges); // Convert back to CV_8U

        // Save the edge-detected image with a filename based on the rank
        string output_filename = "edges_rank" + to_string(rank) + ".jpg";
        imwrite(output_filename, edges);

        // Create a Mat object to hold the final image on rank 0
        Mat final_image;
        if (rank == 0) {
            final_image.create(rows, cols, CV_8UC3);
        }

        // Gather processed parts from all processes to the root process (rank 0)
        MPI_Gather(edges.data, rows_per_process * cols * 3, MPI_UNSIGNED_CHAR,
            final_image.data, rows_per_process * cols * 3, MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);

        // On root process (rank 0), save the final image and print completion message
        if (rank == 0) {
            imwrite(output_img, final_image);
            end_time = MPI_Wtime();
            cout << "Edge Detection operation completed successfully in " << end_time - start_time << " seconds.\n\n"
                << "Edges image saved as " << output_img << ".\n\n"
                << "Thank you for using Parallel Image Processing with MPI.\n\n";
        }

        // Finalize MPI
        MPI_Finalize();
    }
    else if (choice == 3) {
        double angle = 90; // Fixed rotation angle to 90 degrees

        // Broadcast rotation angle to all processes
        MPI_Bcast(&angle, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double start_time, end_time;
        start_time = MPI_Wtime();

        // Perform image rotation on local image
        Mat rotated_image;
        transpose(local_image, rotated_image);
        flip(rotated_image, rotated_image, 1); // Flip around y-axis to correct orientation

        // Save the rotated image with the rank appended to the filename
        string output_filename = "rotated_rank" + to_string(rank) + ".jpg";
        imwrite(output_filename, rotated_image);

        // Get the size of the rotated part on each process
        int rotated_rows = rotated_image.rows;
        int rotated_cols = rotated_image.cols;
        int rotated_size = rotated_rows * rotated_cols * 3; // Size of the rotated part

        // Gather sizes of rotated parts from all processes to the root process (rank 0)
        vector<int> all_rotated_sizes(size); // Vector to hold sizes on root process
        MPI_Gather(&rotated_size, 1, MPI_INT, all_rotated_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate total size for the final image on root process (rank 0)
        int total_size = 0;
        if (rank == 0) {
            for (int size_i : all_rotated_sizes) {
                total_size += size_i;
            }
        }

        // Define the displacements vector
        vector<int> displacements(size, 0);

        // Calculate displacements for gathering the rotated image parts
        if (rank == 0) {
            for (int i = 1; i < size; ++i) {
                displacements[i] = displacements[i - 1] + all_rotated_sizes[i - 1];
            }
        }

        // Gather rotated image parts from all processes into a single buffer on the root process
        vector<unsigned char> gathered_buffer;
        if (rank == 0) {
            gathered_buffer.resize(total_size);
        }
        MPI_Gatherv(rotated_image.data, rotated_size, MPI_UNSIGNED_CHAR,
            gathered_buffer.data(), all_rotated_sizes.data(), displacements.data(), MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);

        // On root process (rank 0), concatenate the rotated image parts horizontally to form the final image
        if (rank == 0) {
            Mat final_image(rotated_rows, rotated_cols * size, CV_8UC3);
            for (int i = size - 1; i >= 0; --i) {
                int offset = (size - 1 - i) * rotated_cols;
                Mat part(rotated_rows, rotated_cols, CV_8UC3, gathered_buffer.data() + displacements[i]);
                part.copyTo(final_image.colRange(offset, offset + rotated_cols));
            }

            // Rotate the output_img filename based on the specified angle
            imwrite(output_img, final_image);
            end_time = MPI_Wtime();
            cout << "Image Rotation operation completed successfully in " << end_time - start_time << " seconds.\n\n"
                << "Rotated image saved as " << output_img << ".\n\n"
                << "Thank you for using Parallel Image Processing with MPI.\n\n";
        }

        // Finalize MPI
        MPI_Finalize();
    }
    else if (choice == 4) {
    double start_time, end_time;
    start_time = MPI_Wtime();

    // Constant scaling factor to double the size
    double scale_factor = 1.5;

    // Perform scaling on local image
    Mat scaled_image;
    resize(local_image, scaled_image, Size(), scale_factor, scale_factor, INTER_LINEAR);

    // Save the scaled image with a filename based on the rank
    string output_filename = "scaled_rank" + to_string(rank) + ".jpg";
    imwrite(output_filename, scaled_image);

    // Get the size of the scaled part on each process
    int scaled_rows = scaled_image.rows;
    int scaled_cols = scaled_image.cols;
    int scaled_size = scaled_rows * scaled_cols * 3; // Size of the scaled part

    // Gather sizes of scaled parts from all processes to the root process (rank 0)
    vector<int> sizes(size); // Vector to hold sizes on root process
    MPI_Gather(&scaled_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for gathering the scaled image parts
    vector<int> displacements(size, 0); // Vector to hold displacements for gathering
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        }
    }

    // Gather scaled image parts from all processes into a single buffer on the root process
    vector<unsigned char> gathered_buffer;
    if (rank == 0) {
        gathered_buffer.resize(displacements.back() + sizes.back());
    }
    MPI_Gatherv(scaled_image.data, scaled_size, MPI_UNSIGNED_CHAR,
        gathered_buffer.data(), sizes.data(), displacements.data(), MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // On root process (rank 0), concatenate the scaled image parts vertically to form the final image
    if (rank == 0) {
        Mat final_image(scaled_rows * size, scaled_cols, CV_8UC3);
        for (int i = 0; i < size; ++i) {
            int offset = i * scaled_rows;
            Mat part(scaled_rows, scaled_cols, CV_8UC3, gathered_buffer.data() + displacements[i]);
            part.copyTo(final_image.rowRange(offset, offset + scaled_rows));
        }

        // Save the final image and print completion message
        imwrite(output_img, final_image);
        end_time = MPI_Wtime();
        cout << "Scaling operation completed successfully in " << end_time - start_time << " seconds.\n\n"
            << "Scaled image saved as " << output_img << ".\n\n"
            << "Thank you for using Parallel Image Processing with MPI.\n\n";
    }

    // Finalize MPI
    MPI_Finalize();
}
    else if (choice == 5) {
    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform histogram equalization on local image
    Mat equalized_image;
    cvtColor(local_image, equalized_image, COLOR_BGR2GRAY);
    equalizeHist(equalized_image, equalized_image);

    // Save the equalized image with a filename based on the rank
    string output_filename = "equalized_rank" + to_string(rank) + ".jpg";
    imwrite(output_filename, equalized_image);

    // Get the size of the equalized part on each process
    int equalized_rows = equalized_image.rows;
    int equalized_cols = equalized_image.cols;
    int equalized_size = equalized_rows * equalized_cols; // Size of the equalized part

    // Gather sizes of equalized parts from all processes to the root process (rank 0)
    vector<int> sizes(size); // Vector to hold sizes on root process
    MPI_Gather(&equalized_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for gathering the equalized image parts
    vector<int> displacements(size, 0); // Vector to hold displacements for gathering
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        }
    }

    // Gather equalized image parts from all processes into a single buffer on the root process
    vector<unsigned char> gathered_buffer;
    if (rank == 0) {
        gathered_buffer.resize(displacements.back() + sizes.back());
    }
    MPI_Gatherv(equalized_image.data, equalized_size, MPI_UNSIGNED_CHAR,
        gathered_buffer.data(), sizes.data(), displacements.data(), MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // On root process (rank 0), concatenate the equalized image parts vertically to form the final image
    if (rank == 0) {
        Mat final_image(equalized_rows * size, equalized_cols, CV_8UC1);
        for (int i = 0; i < size; ++i) {
            int offset = i * equalized_rows;
            Mat part(equalized_rows, equalized_cols, CV_8UC1, gathered_buffer.data() + displacements[i]);
            part.copyTo(final_image.rowRange(offset, offset + equalized_rows));
        }

        // Save the final image and print completion message
        imwrite(output_img, final_image);
        end_time = MPI_Wtime();
        cout << "Histogram Equalization operation completed successfully in " << end_time - start_time << " seconds.\n\n"
            << "Equalized image saved as " << output_img << ".\n\n"
            << "Thank you for using Parallel Image Processing with MPI.\n\n";
    }

    // Finalize MPI
    MPI_Finalize();
}
    else if (choice == 6) {
    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform color space conversion on local image
    Mat converted_image;
    cvtColor(local_image, converted_image, COLOR_BGR2HSV); // Example: Convert BGR to HSV

    // Save the converted image with a filename based on the rank
    string output_filename = "converted_rank" + to_string(rank) + ".jpg";
    imwrite(output_filename, converted_image);

    // Get the size of the converted part on each process
    int converted_rows = converted_image.rows;
    int converted_cols = converted_image.cols;
    int converted_size = converted_rows * converted_cols * converted_image.channels(); // Size of the converted part

    // Gather sizes of converted parts from all processes to the root process (rank 0)
    vector<int> sizes(size); // Vector to hold sizes on root process
    MPI_Gather(&converted_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for gathering the converted image parts
    vector<int> displacements(size, 0); // Vector to hold displacements for gathering
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        }
    }

    // Gather converted image parts from all processes into a single buffer on the root process
    vector<unsigned char> gathered_buffer;
    if (rank == 0) {
        gathered_buffer.resize(displacements.back() + sizes.back());
    }
    MPI_Gatherv(converted_image.data, converted_size, MPI_UNSIGNED_CHAR,
        gathered_buffer.data(), sizes.data(), displacements.data(), MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // On root process (rank 0), concatenate the converted image parts vertically to form the final image
    if (rank == 0) {
        Mat final_image(converted_rows * size, converted_cols, converted_image.type());
        for (int i = 0; i < size; ++i) {
            int offset = i * converted_rows;
            Mat part(converted_rows, converted_cols, converted_image.type(), gathered_buffer.data() + displacements[i]);
            part.copyTo(final_image.rowRange(offset, offset + converted_rows));
        }

        // Save the final image and print completion message
        imwrite(output_img, final_image);
        end_time = MPI_Wtime();
        cout << "Color Space Conversion operation completed successfully in " << end_time - start_time << " seconds.\n\n"
            << "Converted image saved as " << output_img << ".\n\n"
            << "Thank you for using Parallel Image Processing with MPI.\n\n";
    }

    // Finalize MPI
    MPI_Finalize();
}
    else if (choice == 7) {
    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform global thresholding on local image
    Mat thresholded_image;
    cvtColor(local_image, thresholded_image, COLOR_BGR2GRAY); // Convert to grayscale for thresholding
    threshold(thresholded_image, thresholded_image, 128, 255, THRESH_BINARY); // Example: Global thresholding

    // Save the thresholded image with a filename based on the rank
    string output_filename = "thresholded_rank" + to_string(rank) + ".jpg";
    imwrite(output_filename, thresholded_image);

    // Get the size of the thresholded part on each process
    int thresholded_rows = thresholded_image.rows;
    int thresholded_cols = thresholded_image.cols;
    int thresholded_size = thresholded_rows * thresholded_cols; // Size of the thresholded part

    // Gather sizes of thresholded parts from all processes to the root process (rank 0)
    vector<int> sizes(size); // Vector to hold sizes on root process
    MPI_Gather(&thresholded_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for gathering the thresholded image parts
    vector<int> displacements(size, 0); // Vector to hold displacements for gathering
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        }
    }

    // Gather thresholded image parts from all processes into a single buffer on the root process
    vector<unsigned char> gathered_buffer;
    if (rank == 0) {
        gathered_buffer.resize(displacements.back() + sizes.back());
    }
    MPI_Gatherv(thresholded_image.data, thresholded_size, MPI_UNSIGNED_CHAR,
        gathered_buffer.data(), sizes.data(), displacements.data(), MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // On root process (rank 0), concatenate the thresholded image parts vertically to form the final image
    if (rank == 0) {
        Mat final_image(thresholded_rows * size, thresholded_cols, thresholded_image.type());
        for (int i = 0; i < size; ++i) {
            int offset = i * thresholded_rows;
            Mat part(thresholded_rows, thresholded_cols, thresholded_image.type(), gathered_buffer.data() + displacements[i]);
            part.copyTo(final_image.rowRange(offset, offset + thresholded_rows));
        }

        // Save the final image and print completion message
        imwrite(output_img, final_image);
        end_time = MPI_Wtime();
        cout << "Global Thresholding operation completed successfully in " << end_time - start_time << " seconds.\n\n"
            << "Thresholded image saved as " << output_img << ".\n\n"
            << "Thank you for using Parallel Image Processing with MPI.\n\n";
    }

    // Finalize MPI
    MPI_Finalize();
}
    else if (choice == 8) {
    double start_time, end_time;
    start_time = MPI_Wtime();

    // Convert the local image to grayscale
    Mat grayscale_image;
    cvtColor(local_image, grayscale_image, COLOR_BGR2GRAY);

    // Define local variables for local thresholding
    int block_size = 21; // Example: Local thresholding block size
    double threshold_value = 10; // Example: Threshold value for local thresholding

    // Perform local thresholding on grayscale image
    Mat thresholded_image;
    adaptiveThreshold(grayscale_image, thresholded_image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, threshold_value);

    // Save the thresholded image with a filename based on the rank
    string output_filename = "local_thresholded_rank" + to_string(rank) + ".jpg";
    imwrite(output_filename, thresholded_image);

    // Get the size of the thresholded part on each process
    int thresholded_rows = thresholded_image.rows;
    int thresholded_cols = thresholded_image.cols;
    int thresholded_size = thresholded_rows * thresholded_cols * thresholded_image.channels(); // Size of the thresholded part

    // Gather sizes of thresholded parts from all processes to the root process (rank 0)
    vector<int> sizes(size); // Vector to hold sizes on root process
    MPI_Gather(&thresholded_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for gathering the thresholded image parts
    vector<int> displacements(size, 0); // Vector to hold displacements for gathering
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        }
    }

    // Gather thresholded image parts from all processes into a single buffer on the root process
    vector<unsigned char> gathered_buffer;
    if (rank == 0) {
        gathered_buffer.resize(displacements.back() + sizes.back());
    }
    MPI_Gatherv(thresholded_image.data, thresholded_size, MPI_UNSIGNED_CHAR,
        gathered_buffer.data(), sizes.data(), displacements.data(), MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // On root process (rank 0), concatenate the thresholded image parts vertically to form the final image
    if (rank == 0) {
        Mat final_image(thresholded_rows * size, thresholded_cols, thresholded_image.type());
        for (int i = 0; i < size; ++i) {
            int offset = i * thresholded_rows;
            Mat part(thresholded_rows, thresholded_cols, thresholded_image.type(), gathered_buffer.data() + displacements[i]);
            part.copyTo(final_image.rowRange(offset, offset + thresholded_rows));
        }

        // Save the final image and print completion message
        imwrite(output_img, final_image);
        end_time = MPI_Wtime();
        cout << "Local Thresholding operation completed successfully in " << end_time - start_time << " seconds.\n\n"
            << "Thresholded image saved as " << output_img << ".\n\n"
            << "Thank you for using Parallel Image Processing with MPI.\n\n";
    }

    // Finalize MPI
    MPI_Finalize();
}
    else if (choice == 9) {
    }
    else if (choice == 10) {
    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform median filtering on local image
    Mat median_filtered_image;
    medianBlur(local_image, median_filtered_image, 5); // Adjust kernel size as needed

    // Save the median-filtered image with a filename based on the rank
    string output_filename = "median_filtered_rank" + to_string(rank) + ".jpg";
    imwrite(output_filename, median_filtered_image);

    // Create a Mat object to hold the final image on rank 0
    Mat final_image;
    if (rank == 0) {
        final_image.create(rows, cols, CV_8UC3);
    }

    // Gather processed parts from all processes to the root process (rank 0)
    MPI_Gather(median_filtered_image.data, rows_per_process * cols * 3, MPI_UNSIGNED_CHAR,
        final_image.data, rows_per_process * cols * 3, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // On root process (rank 0), save the final image and print completion message
    if (rank == 0) {
        imwrite(output_img, final_image);
        end_time = MPI_Wtime();
        cout << "Median Filtering operation completed successfully in " << end_time - start_time << " seconds.\n\n"
            << "Median filtered image saved as " << output_img << ".\n\n"
            << "Thank you for using Parallel Image Processing with MPI.\n\n";
    }

    // Finalize MPI
    MPI_Finalize();
}









    return 0;
}