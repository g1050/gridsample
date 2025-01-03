#include "gridsample.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

void saveVectorToBinFile(const std::vector<float>& vec, const std::string& filename) {
    // 打开文件，std::ios::binary表示二进制模式
    std::ofstream outFile(filename, std::ios::binary);
    
    // 检查文件是否打开成功
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    
    // 写入数据的大小（可选，用于之后读取时验证）
    // size_t size = vec.size();
    // outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    // 将vector中的每个元素写入文件
    outFile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    
    // 关闭文件
    outFile.close();
    
    std::cout << "Vector saved to " << filename << std::endl;
}

std::vector<float> loadBinaryFile(const std::string& filename) {
    // 打开二进制文件
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open the file: " + filename);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 检查文件大小是否是float的整数倍
    if (fileSize % sizeof(float) != 0) {
        throw std::runtime_error("File size is not a multiple of float size.");
    }

    // 读取数据
    std::vector<float> data(fileSize / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(data.data()), fileSize)) {
        throw std::runtime_error("Failed to read the file.");
    }

    return data;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    const std::string input_filename = "../py/input_tensor.bin"; 
    const std::string grid_filename = "../py/grid.bin"; 

    try {
        std::vector<float> input_tensor = loadBinaryFile(input_filename);
        std::vector<float> grid = loadBinaryFile(grid_filename);
        int64_t input_dims[4] = {1, 3, 10, 10};
        int64_t grid_dims[4] = {1, 10, 10, 2};
        // std::vector<int64_t> output_dims = {N, C, out_H, out_W};
        std::vector<float> output_data;
        output_data.resize(300);
        DgGridSample sample(true,0,0,input_tensor.data(),grid.data(),output_data.data(),input_dims,grid_dims);
        sample.Compute();
        saveVectorToBinFile(output_data,"../py/cpp_output.bin");
        // 打印数据用于验证
        // auto data = output_data;
        // std::cout << "Loaded " << data.size() << " floats from " << filename << ":\n";
        // for (size_t i = 0; i < data.size(); ++i) {
        //     std::cout << data[i] << " ";
        //     if ((i + 1) % 10 == 0) std::cout << "\n"; // 每10个值换行
        // }
        // std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}