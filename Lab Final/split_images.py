from split_image import split_image
import sys

file=sys.argv[1]+sys.argv[2]
split_image(file,10, 10, False, False, output_dir=sys.argv[3])
#split_image(image_path, rows, cols, should_square, should_cleanup, [output_dir])
# e.g. split_image("bridge.jpg", 2, 2, True, False)