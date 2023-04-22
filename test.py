from deepface import DeepFace
import cv2

# 识别给定图像的年龄
img = cv2.imread("2.jpg")

# 显示图片
cv2.imshow("img", img)
cv2.waitKey(0)

# 检测图像中人脸的年龄
result = DeepFace.analyze(img, actions = ['age'])

# 给图像人脸加框
img = DeepFace.detectFace(img)

# 显示图片
cv2.imshow("img", img)
cv2.waitKey(0)

print(result)
