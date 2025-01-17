from os.path import isfile, join
import cv2
import time
import os
from torchvision import transforms


TRAINING_SET_FOLDER = "../training"
GENERATED_DATA_SET_FOLDER = "training/generated"
TRAINING_IMAGES_VIDEOS_SET_FOLDER = "training/training_images"
GENERATED_FRAMES_FOLDER = "/frames"
GENERATED_VIDEOS_FOLDER = "/video"
VIDEO_DATASET_FOLDER = "../dataset/videos"


def resize_video(path, filename, new_path):
    # create a new folder
    cap = cv2.VideoCapture(path + filename)
    new_path_file = os.makedirs(new_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(new_path_file + '1' + filename, fourcc, 5, (256, 256))

    if cap.isOpened() == False:
        print("Error opening video stream or file")
    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame, (256, 256), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.write(b)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def generate_frames_from_video(source_path, output_path):
    print(source_path)
    print(output_path)
    try:
        os.mkdir(output_path)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(source_path)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_path + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds forconversion." % (time_end - time_start))
            break


def generate_video_from_frames(source_path, output_path, fps):
    try:
        os.mkdir(output_path)
    except OSError:
        pass
    image_array = []
    files = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]
    files.sort(key=lambda x: int(float(x.split('.')[0])))
    for i in range(len(files)):
        img = cv2.imread(source_path + "/" + files[i])
        size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()


def save_image_to_folder(path,name,image):
    try:
        os.mkdir(path)
    except OSError:
        pass
    print(path)
    cv2.imwrite(path + name, image)


def generate_frames_from_videos(folder):
    files = os.listdir(folder)
    for vid_inx in range(len(files)):
        generate_frames_from_video(folder + "/" + files[vid_inx], "../training" + "/" + files[vid_inx])


def get_training_set(training_folder):
    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()
    # Convert the image to PyTorch tensor
    #print(training_folder)
    training_images = []
    temp_images = []
    training_list = os.listdir(training_folder)
    for folder in training_list:
        if not folder.startswith("."):
            files = os.listdir(training_folder + "/" + folder)
            files.sort(key=lambda x: int(float(x.split('.')[0])))
            for file in files:
                if not file.startswith("."):
                    img = transform(cv2.imread(training_folder + "/" + folder + "/" + file))
                    temp_images.append(img)
            training_images.append(temp_images)
    return training_images





