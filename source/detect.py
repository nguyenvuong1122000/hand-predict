import time
import numpy as np
import cv2
import torch
cap = cv2.VideoCapture(0)
from SSD_Pure import *
from datasets import read_content
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = '/home/vuong/PycharmProjects/hand-predict/checkpoint/checkpoint_ssd300.pth_SSD_pure.tar'
checkpoint = torch.load(checkpoint, map_location= device)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect_to_tensor(original_image, min_score, max_overlap,  top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    start = time.time()

    # Move to default device
    image = original_image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to(device)



    return det_boxes, det_labels, det_scores

def detect(original_image, min_score, max_overlap,  top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    start = time.time()

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to(device)

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0).to(device)
    det_boxes = det_boxes * original_dims
    print("labels la {}".format(det_labels))

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to(device).tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 50)
    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])
        print(det_labels[i])
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='blue',
                  font=font)
    del draw
    test_time = time.time() - start
    return annotated_image


if __name__ == '__main__':
    # for filename in os.listdir("/home/vuong/PycharmProjects/hand-predict/hand-sign/Data-project-1619886132/test"):
    #     if filename.__contains__("jpeg"):
    #         img_path = os.path.join("/home/vuong/PycharmProjects/hand-predict/hand-sign/Data-project-1619886132/test", filename)
    #         original_image = Image.open(img_path, mode='r')
    #         original_image = original_image.convert('RGB')
    #         detect(original_image, min_score=0.5, max_overlap=0.5, top_k=1).save(os.path.join("final", filename))
    # img_path = "/home/vuong/PycharmProjects/hand-predict/V.jpg"
    # original_image = Image.open(img_path, mode='r')
    # original_image = original_image.convert('RGB')
    # a, scores = detect(original_image, min_score=0.6, max_overlap=0.5, top_k=200)
    #
    out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame = Image.fromarray(np.uint8(frame))
            # Our operations on the frame come here
            frame = detect(frame, min_score=0.5, max_overlap=0.5, top_k=1)
            # Display the resulting frame
            frame = np.array(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindoqws()