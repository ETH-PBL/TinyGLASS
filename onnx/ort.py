import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    # read image
 #   image = Image.open("/datasets/sutterra/mvtec/mms_stretch/test/crack_hole/IMG_0038.png").convert("RGB")
    image = Image.open("/datasets/sutterra/manual/normal.png").convert("RGB")
    image = image.resize((288, 288))
    image = np.array(image).astype(np.float32) / 255.0

    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)
    image_batch = np.repeat(image_batch, 1, axis=0)

    # load model
    r50_session = ort.InferenceSession("/home/sutterra/GLASS/results/models/backbone_0/mvtec_mms_stretch_resnet18_v2/mvtec_mms_stretch_resnet18_v2_simplified.onnx",
                                       providers=['CUDAExecutionProvider'])
    output = r50_session.run(None, {"input": image_batch})[0]
    print(output.shape)

    # segmentation output
    output = np.expand_dims(output, axis=1)[0]
    output = output.transpose((1, 2, 0))
    output = cv2.resize(output, (288, 288), interpolation=cv2.INTER_LINEAR)
    output = cv2.GaussianBlur(output, (33, 33), 4)

    # binary or heatmap
    # ret, mask = cv2.threshold(output, 0.5, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    mask = (mask * 255).astype('uint8')
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    cv2.imwrite("resnet18v2_normal.png", mask)
