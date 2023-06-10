import cv2
import sys

from line_profiler import LineProfiler
profile = LineProfiler()

@profile
def main():
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    counter = 0 
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            last_frame = frame
        counter += 1
        print(f"{counter}/{frame_count}, {frame.shape}")
        cv2.imwrite("x2.jpg",last_frame)
        exit()
    cap.release()
    print(counter)

if __name__ == "__main__":
    main()
    profile.print_stats()


# python3 test_cv2.py sample_yata_yoho.mp4