
#include "gridsample.h"

/// @brief 将归一化的网格坐标转换为原始尺度的坐标
/// @param coord 网格坐标，范围通常为[-1, 1]
/// @param size 原始图像尺寸
/// @param align_corners 是否对齐角点
/// @return 原始尺度的坐标
inline float DgGridSample::grid_sampler_unnormalize(float coord, int64_t size, bool align_corners) {
    if (align_corners) {
        return ((coord + 1) / 2) * (size - 1);
    } else {
        return ((coord + 1) * size - 1) / 2;
    }
}

/// @brief 裁剪坐标，使其在指定范围内
/// @param in 输入坐标
/// @param clip_limit 裁剪范围的上限
/// @return 裁剪后的坐标
inline float DgGridSample::clip_coordinates(float in, int64_t clip_limit) {
    return MIN(clip_limit - 1, MAX(in, 0));
}

/// @brief 反射坐标，用于处理超出边界的情况
/// @param in 输入坐标
/// @param twice_low 边界的两倍下限值
/// @param twice_high 边界的两倍上限值
/// @return 反射后的坐标
inline float DgGridSample::reflect_coordinates(float in, int64_t twice_low, int64_t twice_high) {
    if (twice_low == twice_high) {
        return static_cast<float>(0);
    }
    float min = static_cast<float>(twice_low) / 2;
    float span = static_cast<float>(twice_high - twice_low) / 2;
    in = std::fabs(in - min);
    float extra = std::fmod(in, span);
    int flips = static_cast<int>(std::floor(in / span));
    if (flips % 2 == 0) {
        return extra + min;
    } else {
        return span - extra + min;
    }
}

/// @brief 计算实际使用的坐标，根据填充模式调整
/// @param coord 输入坐标
/// @param size 原始图像尺寸
/// @param padding_mode 填充模式（边界、反射或零填充）
/// @param align_corners 是否对齐角点
/// @return 调整后的坐标
inline float DgGridSample::compute_coordinates(float coord, int64_t size, int64_t padding_mode, bool align_corners) {
    if (padding_mode == GridSamplerPadding::Border) {
        coord = clip_coordinates(coord, size);
    } else if (padding_mode == GridSamplerPadding::Reflection) {
        if (align_corners) {
            coord = reflect_coordinates(coord, 0, 2 * (size - 1));
        } else {
            coord = reflect_coordinates(coord, -1, 2 * size - 1);
        }
        coord = clip_coordinates(coord, size);
    }
    return coord;
}

/// @brief 计算源图像的索引坐标
/// @param coord 输入的网格坐标
/// @param size 图像的尺寸
/// @param padding_mode 填充模式
/// @param align_corners 是否对齐角点
/// @return 图像的源索引坐标
inline float DgGridSample::grid_sampler_compute_source_index(float coord, int64_t size, int64_t padding_mode, bool align_corners) {
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    coord = compute_coordinates(coord, size, padding_mode, align_corners);
    return coord;
}

/// @brief 判断坐标是否在二维范围内
/// @param h 高度方向的坐标
/// @param w 宽度方向的坐标
/// @param H 图像的高度
/// @param W 图像的宽度
/// @return 是否在范围内
inline bool DgGridSample::within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

/// @brief 根据坐标获取图像中对应的值（处理边界情况）
/// @param data 图像数据指针
/// @param x 横向坐标
/// @param y 纵向坐标
/// @param W 图像宽度
/// @param H 图像高度
/// @param sW 横向步幅
/// @param sH 纵向步幅
/// @param padding_mode 填充模式
/// @param align_corners 是否对齐角点
/// @return 对应的像素值
inline float DgGridSample::get_value_bounded(const float *data, float x, float y, int64_t W, int64_t H, int64_t sW, int64_t sH, int64_t padding_mode, bool align_corners) {
    x = compute_coordinates(x, W, padding_mode, align_corners);
    y = compute_coordinates(y, H, padding_mode, align_corners);

    int64_t ix = static_cast<int64_t>(x);
    int64_t iy = static_cast<int64_t>(y);

    if (within_bounds_2d(iy, ix, H, W)) {
        return data[iy * sH + ix * sW];
    }
    return static_cast<float>(0);
}

/// @brief 计算一维三次卷积的第一个公式
/// @param x 输入值
/// @param A 参数A
/// @return 计算结果
inline float DgGridSample::cubic_convolution1(float x, float A) {
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

/// @brief 计算一维三次卷积的第二个公式
/// @param x 输入值
/// @param A 参数A
/// @return 计算结果
inline float DgGridSample::cubic_convolution2(float x, float A) {
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

/// @brief 获取三次上采样的系数
/// @param coeffs 存储系数的数组
/// @param t 输入值
inline void DgGridSample::get_cubic_upsample_coefficients(float coeffs[4], float t) {
    float A = -0.75;
    float x1 = t;
    coeffs[0] = cubic_convolution2(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1(x1, A);

    float x2 = 1.0 - t;
    coeffs[2] = cubic_convolution1(x2, A);
    coeffs[3] = cubic_convolution2(x2 + 1.0, A);
}

/// @brief 使用一维三次插值计算插值值
/// @param x0 插值点左侧第一个值
/// @param x1 插值点左侧第二个值
/// @param x2 插值点右侧第一个值
/// @param x3 插值点右侧第二个值
/// @param t 插值因子
/// @return 插值结果
inline float DgGridSample::cubic_interp1d(float x0, float x1, float x2, float x3, float t) {
    float coeffs[4];
    get_cubic_upsample_coefficients(coeffs, t);

    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

void DgGridSample::Compute(){
  std::cout << align_corners_ << std::endl;
  
  const bool align_corners = align_corners_;
  const int64_t padding_mode = padding_mode_;
  const int64_t interpolation_mode = interpolation_mode_;
  // 维度信息
  int64_t N = input_dims[0];
  int64_t C = input_dims[1];
  int64_t inp_H = input_dims[2];
  int64_t inp_W = input_dims[3];
  int64_t out_H = grid_dims[1];
  int64_t out_W = grid_dims[2];

  // 输出维度
  std::vector<int64_t> output_dims = {N, C, out_H, out_W};
//   float *out_ptr = ort_.GetTensorMutableData<float>(output);

  int64_t inp_sN = input_dims[1] * input_dims[2] * input_dims[3];
  int64_t inp_sC = input_dims[2] * input_dims[3];
  int64_t inp_sH = input_dims[3];
  int64_t inp_sW = 1;
  int64_t grid_sN = grid_dims[1] * grid_dims[2] * grid_dims[3];
  int64_t grid_sH = grid_dims[2] * grid_dims[3];
  int64_t grid_sW = grid_dims[3];
  int64_t grid_sCoor = 1;
  int64_t out_sN = output_dims[1] * output_dims[2] * output_dims[3];
  int64_t out_sC = output_dims[2] * output_dims[3];
  int64_t out_sH = output_dims[3];
  int64_t out_sW = 1;

  // loop over each output pixel
  for (int64_t n = 0; n < N; ++n) {
    const float *grid_ptr_N = grid_data + n * grid_sN;
    const float *inp_ptr_N = input_data + n * inp_sN;
    for (int64_t h = 0; h < out_H; ++h) {
      for (int64_t w = 0; w < out_W; ++w) {
        const float *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
        float x = *grid_ptr_NHW;
        float y = grid_ptr_NHW[grid_sCoor];

        float ix = grid_sampler_compute_source_index(x, inp_W, padding_mode,
                                                     align_corners);
        float iy = grid_sampler_compute_source_index(y, inp_H, padding_mode,
                                                     align_corners);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          // get corner pixel values from (x, y)
          // for 4d, we use north-east-south-west
          int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
          int64_t iy_nw = static_cast<int64_t>(std::floor(iy));

          int64_t ix_ne = ix_nw + 1;
          int64_t iy_ne = iy_nw;

          int64_t ix_sw = ix_nw;
          int64_t iy_sw = iy_nw + 1;

          int64_t ix_se = ix_nw + 1;
          int64_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          float nw = (ix_se - ix) * (iy_se - iy);
          float ne = (ix - ix_sw) * (iy_sw - iy);
          float sw = (ix_ne - ix) * (iy - iy_ne);
          float se = (ix - ix_nw) * (iy - iy_nw);

          // calculate bilinear weighted pixel value and set output pixel
          const float *inp_ptr_NC = inp_ptr_N;
          float *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
          for (int64_t c = 0; c < C;
               ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
            auto res = static_cast<float>(0);
            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
              res += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
            }
            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
              res += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
            }
            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
              res += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
            }
            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
              res += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
            }
            *out_ptr_NCHW = res;
          }
        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
          int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
          int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));

          // assign nearest neighbor pixel value to output pixel
          float *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
          const float *inp_ptr_NC = inp_ptr_N;
          for (int64_t c = 0; c < C;
               ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
            if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
              *out_ptr_NCHW =
                  inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
            } else {
              *out_ptr_NCHW = static_cast<float>(0);
            }
          }
        } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
          // grid_sampler_compute_source_index will "clip the value" of idx
          // depends on the padding,
          // which would cause calculation to be wrong,
          // for example x = -0.1 -> ix = 0 for zero padding, but in bicubic ix
          // = floor(x) = -1
          // There would be more problem in reflection padding, since the -1 and
          // +1 direction is not fixed in boundary condition
          ix = grid_sampler_unnormalize(x, inp_W, align_corners);
          iy = grid_sampler_unnormalize(y, inp_H, align_corners);

          float ix_nw = std::floor(ix);
          float iy_nw = std::floor(iy);

          const float tx = ix - ix_nw;
          const float ty = iy - iy_nw;

          const float *inp_ptr_NC = inp_ptr_N;
          float *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
          for (int64_t c = 0; c < C;
               ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
            float coefficients[4];

            // Interpolate 4 values in the x direction
            for (int64_t i = 0; i < 4; ++i) {
              coefficients[i] = cubic_interp1d(
                  get_value_bounded(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i,
                                           inp_W, inp_H, inp_sW, inp_sH,
                                           padding_mode, align_corners),
                  get_value_bounded(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i,
                                           inp_W, inp_H, inp_sW, inp_sH,
                                           padding_mode, align_corners),
                  get_value_bounded(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i,
                                           inp_W, inp_H, inp_sW, inp_sH,
                                           padding_mode, align_corners),
                  get_value_bounded(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i,
                                           inp_W, inp_H, inp_sW, inp_sH,
                                           padding_mode, align_corners),
                  tx);
            }

            // Interpolate in the y direction
            *out_ptr_NCHW =
                cubic_interp1d(coefficients[0], coefficients[1],
                                      coefficients[2], coefficients[3], ty);
          }
        }
      }
    }
  }
}