# Import Libraries/Functions/Modules
import gc
import json
import string
from django.shortcuts import render
from .models import *
from django.views.decorators import gzip
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
import cv2
import threading
import os

import torch
import torch.nn as nn

import numpy as np

import torch.nn.functional as F
from .WSASL.pytorch_i3d import InceptionI3d

import cv2

from django.db.models import Case, When
from django.views.decorators.csrf import csrf_exempt

# Global Variables

# Define CUDA Device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Keep Track of If the Camera Is On Or Not
camera_on = False
camera_thread = None

# Model Variables
i3d = None
frames = []

offset = 0
text = " "
batch = 50
text_list = []
word_list = []
sentence = ""
text_count = 0

model_loaded = False


# Home Page
def home(request):
    unload_model()
    return render(request, "index.html")


# Sign Language To Text Translation From Live Camera Feed
def SignTotext(request):
    SetModel(request, "model100")

    return render(request, "translator.html")


# Function To Change WSASL Model
def SetModel(request, model):
    unload_model()
    num_classes = 0

    weights = ""

    model = str(model)

    if model == "model100":
        weights = os.path.join(
            os.path.dirname(__file__),
            "WSASL/archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt",
        )

        num_classes = 100

        print("\nWSASL100 Selected\n")

    elif model == "model300":
        weights = os.path.join(
            os.path.dirname(__file__),
            "WSASL/archived/asl300/FINAL_nslt_300_iters=2997_top1=56.14_top5=79.94_top10=86.98.pt",
        )

        num_classes = 300

        print("\nWSASL300 Selected\n")

    elif model == "model1000":
        weights = os.path.join(
            os.path.dirname(__file__),
            "WSASL/archived/asl1000/FINAL_nslt_1000_iters=5104_top1=47.33_top5=76.44_top10=84.33.pt",
        )

        num_classes = 1000

        print("\nWSASL1000 Selected\n")

    elif model == "model2000":
        weights = os.path.join(
            os.path.dirname(__file__),
            "WSASL/archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt",
        )

        num_classes = 2000

        print("\nWSASL2000 Selected\n")

    create_WLASL_dictionary()

    load_model(weights, num_classes)

    return JsonResponse(
        {"message": f"Django view called successfully with WSASL{model}"}
    )


def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


# Text To Sign Language Video Translation Page
def TextToSign(request):
    unload_model()
    if request.method == "POST":
        userSentence = request.POST.get("sentence")
        userSentence = str(userSentence)
        userSentence = remove_punctuation(userSentence)
        userSentence = userSentence.lower()

        tokens = []

        # Check is the words in the sentence is available in the database of Sign Language videos. If not available, split the word into letters and show videos of the letters of the word to spell the word.
        for word in userSentence.split():
            if Video.objects.filter(title=word).exists():
                tokens.append(word)
            else:
                tokens.extend(list(word))

        videos = Video.objects.filter(title__in=tokens).order_by(
            Case(*[When(title=token, then=pos) for pos, token in enumerate(tokens)])
        )

        response_data = {"videos": []}

        for token in tokens:
            matching_videos = videos.filter(title=token)
            response_data["videos"].extend(
                [{"video_file": video.video_file.url} for video in matching_videos]
            )

        response_json = json.dumps(response_data)

        return JsonResponse(json.loads(response_json), safe=False)
    return render(request, "signToText.html")


# Get Detected Word List From The Page Using AJAX
@csrf_exempt
def get_sentence(request):
    return JsonResponse({"content": sentence})


# Clear Current Detected Sentence String/Word List
@csrf_exempt
def clear_sentence(request):
    if request.method == "POST":
        global i3d, frames, camera_thread, text_count, sentence, word_list, text_list, text, offset, batch, camera_on, camera_thread

        offset = 0
        text = " "
        batch = 50
        text_list = []
        word_list = []
        sentence = ""
        text_count = 0

        torch.cuda.empty_cache()
        print("Freed Up GPU VRAM From Clear Sentence")

        return JsonResponse({"content": sentence})
    else:
        return HttpResponse(status=400)


# Load Pretarined Weights
def load_model(weights, num_classes):
    global i3d, model_loaded
    i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    model_loaded = True


# Unload Pretarined Weights
def unload_model():
    global i3d, frames, camera_thread, text_count, sentence, word_list, text_list, text, offset, batch, camera_on, camera_thread

    camera_on = False

    if camera_thread is not None:
        camera_thread.join()
        camera_thread = None
        print("Camera Thread Closed")

        camera_thread = None

        i3d = None
        frames = []

        offset = 0
        text = " "
        batch = 50
        text_list = []
        word_list = []
        sentence = ""
        text_count = 0

        gc.collect()
        print("Garbage Collected")
        torch.cuda.empty_cache()
        print("Freed Up GPU VRAM From Unload Model")


# Validate Frame Sequence and Detect A Word
def run_on_tensor(ip_tensor):
    ip_tensor = ip_tensor[None, :]

    t = ip_tensor.shape[2]
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)

    predictions = F.interpolate(per_frame_logits, t, mode="linear")

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    arr = predictions.cpu().detach().numpy()[0]

    print(float(max(F.softmax(torch.from_numpy(arr[0]), dim=0))))
    print(wlasl_dict[out_labels[0][-1]])

    """
    The 0.5 is threshold value, it varies if the batch sizes are reduced.
    """
    if max(F.softmax(torch.from_numpy(arr[0]), dim=0)) > 0.5:
        torch.cuda.empty_cache()
        # print("Freed Up GPU VRAM From Run On Tensor")
        return wlasl_dict[out_labels[0][-1]]
    else:
        return " "


# Create A Dictionary BAsed On The Total Available Words (2000)
def create_WLASL_dictionary():
    global wlasl_dict
    wlasl_dict = {}

    file_path = os.path.join(
        os.path.dirname(__file__), "WSASL/preprocess/wlasl_class_list.txt"
    )
    with open(file_path) as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value


# Live Camera Feed
@gzip.gzip_page
def webcam_view(request):
    if model_loaded:
        try:
            global camera_on, camera_thread
            camera_on = True
            cam = VideoCamera()
            camera_thread = threading.Thread(target=cam.update, args=())
            camera_thread.start()
            if camera_thread is not None:
                print("Camera Thread Created")
            return StreamingHttpResponse(
                gen(cam), content_type="multipart/x-mixed-replace;boundary=frame"
            )
        except:
            pass
    return render(request, "webcam.html")


# Show Captured Camera Feed and Detect Words From Sign. And Other Stuffs.
class VideoCamera(object):
    # Constructor
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.ret, self.frame1) = self.video.read()
        # Start New Thread
        self.lock = threading.Lock()

    # Release Camera
    def __del__(self):
        self.video.release()

    # Capture Frames One by One And Encode to .jpg
    def get_frame(self):
        image = self.frame1

        _, jpeg = cv2.imencode(".jpg", image)

        return jpeg.tobytes()

    # Update Frame One After Another
    def update(self):
        while True:
            global camera_on
            with self.lock:
                if not camera_on:
                    break
                (self.ret, self.frame1) = self.video.read()

                # WSASL Code
                global offset, text, batch, text_list, word_list, sentence, text_count

                offset = offset + 1
                font = cv2.FONT_HERSHEY_TRIPLEX

                # If Frame Capture Is Successful
                if self.ret == True:
                    w, h, c = self.frame1.shape
                    sc = 224 / w
                    sx = 224 / h
                    frame = cv2.resize(self.frame1, dsize=(0, 0), fx=sx, fy=sc)
                    self.frame1 = cv2.resize(self.frame1, dsize=(1280, 720))

                    frame = (frame / 255.0) * 2 - 1

                    if offset > batch:
                        frames.pop(0)
                        frames.append(frame)

                        if offset % 25 == 0:
                            text = run_on_tensor(
                                torch.from_numpy(
                                    (np.asarray(frames, dtype=np.float32)).transpose(
                                        [3, 0, 1, 2]
                                    )
                                )
                            )
                            if text != " ":
                                text_count = text_count + 1

                                if (
                                    bool(text_list) != False
                                    and bool(word_list) != False
                                    and text_list[-1] != text
                                    and word_list[-1] != text
                                    or bool(text_list) == False
                                ):
                                    text_list.append(text)
                                    word_list.append(text)
                                    sentence = sentence + " " + text
                        torch.cuda.empty_cache()
                        # print("Freed Up GPU VRAM From If")
                    else:
                        frames.append(frame)
                        if offset == batch:
                            text = run_on_tensor(
                                torch.from_numpy(
                                    (np.asarray(frames, dtype=np.float32)).transpose(
                                        [3, 0, 1, 2]
                                    )
                                )
                            )
                            if text != " ":
                                text_count = text_count + 1
                                if (
                                    bool(text_list) != False
                                    and bool(word_list) != False
                                    and text_list[-1] != text
                                    and word_list[-1] != text
                                    or bool(text_list) == False
                                ):
                                    text_list.append(text)
                                    word_list.append(text)
                                    sentence = sentence + " " + text
                                    torch.cuda.empty_cache()
                                    # print("Freed Up GPU VRAM From Else")
                    if len(text_list) > 10:
                        text_list.pop()
                        text_list.pop()
                        text_list.pop()

                else:
                    break


# Generate The Frames to Show
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
