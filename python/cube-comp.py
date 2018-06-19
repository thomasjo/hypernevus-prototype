import sys
import fpipy
from pathlib import Path
from preprocessing import (
    convert_to_radiance_cube,
    crop,
    extract_truecolor_image,
    load_raw_cube,
    normalize,
)


def main():
    root_dir = Path("/root/data/Anonymous")
    for cube_file in root_dir.rglob("RawMeasurementCube.hdr"):
        shasum = cube_file.parents[0].name
        if shasum == "094bd4add160dda9652093c8b51fc0f9beee8b2e":
            continue

        sys.stdout.write(shasum)
        sys.stdout.flush()

        # cube = fpipy.read_cfa(str(cube_file))
        # cube = fpipy.raw_to_radiance(cube)
        cube = load_raw_cube(str(cube_file))
        cube = convert_to_radiance_cube(cube)
        cube = crop(cube)

        if cube.images.ndim < 3:
            continue

        flat_file = cube_file.with_name("WhiteReference.hdr")
        if not flat_file.exists():
            continue

        # flat_cube = fpipy.read_cfa(str(flat_file))
        # flat_cube = fpipy.raw_to_radiance(flat_cube)
        flat_cube = load_raw_cube(str(flat_file))
        flat_cube = convert_to_radiance_cube(flat_cube)
        flat_cube = crop(flat_cube)

        # print((flat_cube.values[:, :, :] > cube.values[:,:,:]).sum())
        bad_pixels = cube.images > flat_cube.images
        bad_pixels = bad_pixels.sum()
        # print("{} {}".format(shasum, bad_pixels))

        sys.stdout.write(" {}\n".format(str(bad_pixels)))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
