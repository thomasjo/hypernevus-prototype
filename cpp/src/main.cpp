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
  auto file_path = "/root/data/examples/a965ccdcc83d466386649b1a21a927b1078a71bb/RawMeasurementCube.dat";
  auto dataset = static_cast<GDALDataset*>(GDALOpen(file_path, GA_ReadOnly));

  // Grab raster metadata.
  auto raster_width = dataset->GetRasterXSize();
  auto raster_height = dataset->GetRasterYSize();
  auto raster_bands = dataset->GetRasterCount();

  for (auto band_num = 1; band_num < raster_bands; band_num++) {
    auto band = dataset->GetRasterBand(band_num);
    // band_matrix pixels(raster_height, raster_width);
    auto pixels = band_matrix(raster_height, raster_width);

    auto error = band->RasterIO(
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

    auto image_width = static_cast<size_t>(std::ceil(raster_width / 2));
    auto image_height = static_cast<size_t>(std::ceil(raster_height / 2));

    auto r = band_matrix(image_height, image_width);
    auto g = band_matrix(image_height, image_width);
    auto b = band_matrix(image_height, image_width);

    for (auto row = 0; row < raster_height; row++) {
      for (auto col = 0; col < raster_width; col++) {
        auto pixel = pixels(row, col);
        auto mapped_row = static_cast<size_t>(std::ceil(row / 2));
        auto mapped_col = static_cast<size_t>(std::ceil(col / 2));
    for (auto col = 0; col < raster_width; col++) {
      for (auto row = 0; row < raster_height; row++) {

        if (row % 2 == 0) {
          if (col % 2 == 0) {
            b(mapped_row, mapped_col) = pixel;
          } else {
            g(mapped_row, mapped_col) = (g(mapped_row, mapped_col) + pixel) / 2;
          }
        } else {
          if (col % 2 == 0) {
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
