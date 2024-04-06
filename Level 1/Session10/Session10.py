from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO('yolov8n.pt')  # load an official model


results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

type(results)

results[0].boxes

print(model.names[5])
print(model.names[0])
print(model.names[11])

plt.imshow(cv2.cvtColor(results[0].plot(conf=False), cv2.COLOR_BGR2RGB))

# Another Example
results = model("https://i.ytimg.com/vi/NyLF8nHIquM/maxresdefault.jpg")

plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
