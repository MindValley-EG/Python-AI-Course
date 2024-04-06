from IPython.display import display
from PIL import Image
from yolov5.ultralytics import yolov5

image_path = "image.jpg"
image = Image.open(image_path)
display(image)

model = yolov5("yolov5s")

results = model(image_path)

num_objects = len(results.pred[0])
print("Number of objects:", num_objects)

results.show()
