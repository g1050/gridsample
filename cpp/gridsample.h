#include <cmath>
#include <iostream>
#include <vector>
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define CLIP_COORDINATES(in, out, clip_limit) \
  out = MIN((clip_limit - 1), MAX(in, 0))

// modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/GridSampler.cpp

// 双线性差值、最近邻差值、双三次差值
enum GridSamplerInterpolation { Bilinear = 0, Nearest = 1, Bicubic = 2 };
// 零填充、边界填充、反射填充
enum GridSamplerPadding { Zeros = 0, Border = 1, Reflection = 2 };

class DgGridSample
{
public:
  DgGridSample(int64_t align_corners = 0,
               int64_t interpolation_mode = 0,
               int64_t padding_mode = 0,
               float *input_data=nullptr,
               float *grid_data=nullptr,
               float *out_ptr=nullptr,
               const int64_t input_dims[4] = {0},
               const int64_t grid_dims[2] = {0})
      : align_corners_(align_corners),
        interpolation_mode_(interpolation_mode),
        padding_mode_(padding_mode),
        input_data(input_data),
        grid_data(grid_data),
        out_ptr(out_ptr){
          std::copy(input_dims, input_dims + 4, this->input_dims);
          std::copy(grid_dims, grid_dims + 2, this->grid_dims);
        }
    void Compute();
private:
    // Helper functions
    static inline float grid_sampler_unnormalize(float coord, int64_t size, bool align_corners);
    static inline float clip_coordinates(float in, int64_t clip_limit);
    static inline float reflect_coordinates(float in, int64_t twice_low, int64_t twice_high);
    static inline float compute_coordinates(float coord, int64_t size, int64_t padding_mode, bool align_corners);
    static inline float grid_sampler_compute_source_index(float coord, int64_t size, int64_t padding_mode, bool align_corners);
    static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W);
    static inline float get_value_bounded(const float *data, float x, float y, int64_t W, int64_t H, int64_t sW, int64_t sH, int64_t padding_mode, bool align_corners);
    static inline float cubic_convolution1(float x, float A);
    static inline float cubic_convolution2(float x, float A);
    static inline void get_cubic_upsample_coefficients(float coeffs[4], float t);
    static inline float cubic_interp1d(float x0, float x1, float x2, float x3, float t);
    
    int64_t align_corners_;
    int64_t interpolation_mode_;
    int64_t padding_mode_;
    float *input_data = nullptr; // // 输入张量
    float *grid_data = nullptr; // 网格张量
    float *out_ptr = nullptr;
    int64_t input_dims[4] = {0};
    int64_t grid_dims[2] = {0};
};