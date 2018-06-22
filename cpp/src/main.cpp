#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gdal_priv.h>

using band_matrix = Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>;
using bayer_block = Eigen::Matrix<uint16_t, 2, 2>;


int main() {
  // Register GDAL drivers.
  GDALAllRegister();

  // Open a GDAL dataset.
  auto const file_path = "/root/data/examples/a965ccdcc83d466386649b1a21a927b1078a71bb/RawMeasurementCube.dat";
  auto const dataset = static_cast<GDALDataset*>(GDALOpen(file_path, GA_ReadOnly));

  // Grab raster metadata.
  auto const raster_width = dataset->GetRasterXSize();
  auto const raster_height = dataset->GetRasterYSize();
  auto const raster_bands = dataset->GetRasterCount();

  for (auto band_num = 1; band_num < raster_bands; band_num++) {
    auto const band = dataset->GetRasterBand(band_num);
    auto pixels = band_matrix(raster_height, raster_width);

    auto const error = band->RasterIO(
      GF_Read,
      0,
      0,
      raster_width,
      raster_height,
      pixels.data(),
      raster_width,
      raster_height,
      GDT_UInt16,
      0,
      0
    );

    if (error > 0) {
      std::cout << "read error: " << error << "\n";
    }

    auto const image_width = static_cast<uint>(std::ceil(raster_width / 2));
    auto const image_height = static_cast<uint>(std::ceil(raster_height / 2));

    auto r = band_matrix(image_height, image_width);
    auto g = band_matrix(image_height, image_width);
    auto b = band_matrix(image_height, image_width);

    for (auto col = 0; col < raster_width; col++) {
      for (auto row = 0; row < raster_height; row++) {
        auto const pixel = pixels(row, col);
        auto const mapped_row = static_cast<uint>(std::floor(row / 2));
        auto const mapped_col = static_cast<uint>(std::floor(col / 2));

        auto const is_upper_half = (row % 2 != 0);
        auto const is_left_half = (col % 2 != 0);

        if (is_upper_half) {
          if (is_left_half) {
            b(mapped_row, mapped_col) = pixel;
          } else {
            g(mapped_row, mapped_col) = (g(mapped_row, mapped_col) + pixel) / 2;
          }
        } else {
          if (is_left_half) {
            g(mapped_row, mapped_col) = pixel;
          } else {
            r(mapped_row, mapped_col) = pixel;
          }
        }
      }
    }
  }

  // Clean up GDAL stuff...
  GDALClose(dataset);
}
