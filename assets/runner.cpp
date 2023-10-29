#include <fstream>
#include <iostream>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 3 || (argv[1][0] > '7' || argv[1][0] < '0') ||
      (argv[2][0] > '7' || argv[2][0] < '0')) {
    std::cout << "You need to write number of task witch you want to get data\n"
              << "0 KernalAdd\n"
              << "1 KernalMul\n"
              << "2 KernalMatrixAdd\n"
              << "3 MatrixVectorMul\n"
              << "4 ScalarMul\n"
              << "5 CosineVector\n"
              << "6 MatrixMul\n"
              << "7 Filter\n";
    return 1;
  }
  const std::string data_file =
      std::string() + "graph_data" + argv[1][0] + ".txt";
  const std::string runner_file =
      std::string() + "task" + argv[1][0] + "/task.cu";
  const std::string compile_command = std::string() + "nvcc " + runner_file;
  if (argv[1][0] != '4' && argv[1][0] != '5') {
    // потому что для для скалярного произведения там 2 файла
    system((std::string() + "rm " + data_file).data());
    system((std::string() + "touch " + data_file).data());
  }
  const size_t counter = argv[2][0] - '0'; // number of videocard
  if (argv[1][0] == '0' || argv[1][0] == '1') {
    const size_t min_block_size = 32;
    const size_t max_block_size = 1024;
    const size_t min_array_size = 1 << 5;
    const size_t max_array_size = 1 << 28;
    std::string cur_str;

    // better to take one videocard its behavior is more predictable
    for (size_t block_size = min_block_size; block_size <= max_block_size;
         block_size <<= 1) {
      for (size_t array_size = min_array_size; array_size <= max_array_size;
           array_size <<= 1) {
        std ::cout << array_size << '\t' << block_size << '\n';
        cur_str = "CUDA_VISIBLE_DEVICES=" + std::to_string(counter) +
                  " ./a.out " + std::to_string(array_size) + " " +
                  std::to_string(block_size);
        system(cur_str.data());
      }
    }
  } else if (argv[1][0] == '2' || argv[1][0] == '3') {
    const size_t min_block_size = 32;
    const size_t max_block_size = 1024;
    const size_t min_array_size = 8;
    const size_t max_array_size = 1 << 10;
    std::string cur_str;

    // better to take one videocard its behavior is more predictable
    for (size_t block_size = min_block_size; block_size <= max_block_size;
         block_size <<= 1) {
      for (size_t array_size_y = min_array_size; array_size_y <= max_array_size;
           array_size_y <<= 1) {
        for (size_t array_size_x = min_array_size;
             array_size_x <= max_array_size; array_size_x <<= 1) {
          std ::cout << array_size_y << '\t' << array_size_x << '\t'
                     << block_size << '\n';
          cur_str = "CUDA_VISIBLE_DEVICES=" + std::to_string(counter) +
                    " ./a.out " + std::to_string(array_size_y) + " " +
                    std::to_string(array_size_x) + " " +
                    std::to_string(block_size);
          system(cur_str.data());
        }
      }
    }
  } else if (argv[1][0] == '4' || argv[1][0] == '5') {
    const size_t min_block_size = 32;
    const size_t max_block_size = 1024;
    const size_t min_array_size = 1 << 5;
    const size_t max_array_size = 1 << 28;
    std::string cur_str;
    const std::string data_file1 =
        std::string() + "graph_data" + argv[1][0] + "1.txt";
    const std::string data_file2 =
        std::string() + "graph_data" + argv[1][0] + "2.txt";
    system((std::string() + "rm " + data_file1).data());
    system((std::string() + "touch " + data_file1).data());
    // Первые данные для ScalarMulTwoReduction
    system((std::string() + "rm " + data_file2).data());
    system((std::string() + "touch " + data_file2).data());
    // Вторые данные для ScalarMulSumPlusReduction

    for (size_t block_size = min_block_size; block_size <= max_block_size;
         block_size <<= 1) {
      for (size_t array_size = min_array_size; array_size <= max_array_size;
           array_size <<= 1) {
        std ::cout << array_size << '\t' << block_size << '\t' << 0 << '\n';
        cur_str = "CUDA_VISIBLE_DEVICES=" + std::to_string(counter) +
                  " ./a.out " + std::to_string(array_size) + " " +
                  std::to_string(block_size) + " 0";
        system(cur_str.data());
      }
      for (size_t array_size = min_array_size; array_size <= max_array_size;
           array_size <<= 1) {
        std ::cout << array_size << '\t' << block_size << '\t' << 1 << '\n';
        cur_str = "CUDA_VISIBLE_DEVICES=" + std::to_string(counter) +
                  " ./a.out " + std::to_string(array_size) + " " +
                  std::to_string(block_size) + " 1";
        system(cur_str.data());
      }
    }
  } else if (argv[1][0] == '6') {
    const size_t min_block_size = 32;
    const size_t max_block_size = 1024;
    const size_t min_array_size = 8;
    const size_t max_array_size = 1 << 14;
    std::string cur_str;

    // better to take one videocard its behavior is more predictable
    for (size_t block_size = min_block_size; block_size <= max_block_size;
         block_size <<= 1) {
      for (size_t heightA = min_array_size; heightA <= max_array_size;
           heightA <<= 1) {
        for (size_t widthA = min_array_size; widthA <= max_array_size;
             widthA <<= 1) {
          for (size_t widthB = min_array_size; widthB <= max_array_size;
               widthB <<= 1) {
            std ::cout << heightA << '\t' << widthA << '\t' << widthB << '\t'
                       << block_size << '\n';
            cur_str = "CUDA_VISIBLE_DEVICES=" + std::to_string(counter) +
                      " ./a.out " + std::to_string(heightA) + " " +
                      std::to_string(widthA) + " " + std::to_string(widthB) +
                      " " + std::to_string(block_size);
            system(cur_str.data());
          }
        }
      }
    }
  }
  return 0;
}