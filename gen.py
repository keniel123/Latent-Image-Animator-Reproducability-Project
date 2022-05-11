


import os
import cv2
import time

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

        
def generate_frames_from_videos(folder):
    files = os.listdir(folder)
    for vid_inx in range(len(files)):
        generate_frames_from_video(folder + "/" + files[vid_inx], "train_vox" + "/" + files[vid_inx])

        
