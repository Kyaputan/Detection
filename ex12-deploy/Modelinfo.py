from ultralytics import YOLO , RTDETR
import os
from onnx_opcounter import calculate_params
import onnx


folder_path = os.path.dirname(os.path.realpath(__file__))
modelYopt = os.path.join(folder_path, "ModelYolo.pt")
modelRTpt = os.path.join(folder_path, "ModelRT.pt")
modelYoonnx = os.path.join(folder_path, "ModelYolo.onnx")
modelRTonnx = os.path.join(folder_path, "ModelRT.onnx")

modelRT = RTDETR(modelYopt)
modelYo = YOLO(modelRTpt)

print("names:", modelYo.names)

print("==="*30)
print(f"|Model RTDETR info : {modelRT.info()} |")
print("==="*30)
print(f"|Model Yolo info : {modelYo.info()} |")
print("==="*30)
# ===============================================================================
modelYo = onnx.load_model(modelYoonnx)
modelRo = onnx.load_model(modelRTonnx)

paramsY = calculate_params(modelYo)
paramsR = calculate_params(modelRo)



print("==="*30)
print(f"|Model RTDETR info : {paramsR} |")
print("==="*30)
print(f"|Model Yolo info : {paramsY} |")
print("==="*30)