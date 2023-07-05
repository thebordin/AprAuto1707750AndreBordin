from PIL import Image
import sys

file=sys.argv[1]+sys.argv[2]
print(sys.argv[2])
image = Image.open(file)
print(f"Original size : {image.size}") # 5464x3640
resized = image.resize((25, 25))
name = sys.argv[2]
print(name)
output_file = sys.argv[3]+'/'+name
print(output_file)
resized.save(output_file)


'''image = Image.open('./output/s1707272_0.jpg')
print(f"Original size : {image.size}") # 5464x3640

sunset_resized = image.resize((10, 10))
sunset_resized.save('./output/sunset_400.jpeg')'''