import numpy as np
import cv2


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

def showim(im):
    cv2.imshow('test', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def app(inPutFile):
    video = cv2.VideoCapture(inPutFile)
    outVideo = None
    counter = 0
    first_frame = None
    endFilter = None
    endHeatmap= None
    gray = None
    framesNum= int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pontOfinterests = []
    for _ in range(framesNum):

        check, frame = video.read()
        if check:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 1.1)
            if first_frame is None:
                first_frame = gray
                height, width = frame.shape[:2]
                outVideo = cv2.VideoWriter(inPutFile[:-4]+'_output.avi', cv2.VideoWriter_fourcc(*"MJPG") , 30.0, (width, height))
                endHeatmap = np.zeros((height, width), np.uint8)
                endFilter = np.zeros((height, width), np.uint8)
                continue

            delta_frame = cv2.absdiff(first_frame, gray)

            thresh_frame = cv2.threshold(delta_frame, 40, 70, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
            endHeatmap = cv2.addWeighted(endHeatmap, 1, thresh_frame, 0.01, 0)
            endFilter = cv2.addWeighted(endFilter, 0.99, thresh_frame, 0.1, 0)

            frame = cv2.addWeighted(frame, 0.7, cv2.applyColorMap(endFilter, cv2.COLORMAP_JET), 0.7, 0)


            outVideo.write(frame)


            if counter == 20:
                first_frame = gray
                counter = 0
            else:
                counter += 1

            print(inPutFile,'  ||  ',_/framesNum)

    video.release()
    outVideo.release()

    cv2.imwrite(inPutFile[:-4]+'_heatmap.jpg', cv2.applyColorMap(endHeatmap, cv2.COLORMAP_JET))


if __name__ == '__main__':
    files=['People_Walking_In_The_Mall_pz11_2046.mp4' , 'People_Chilling_In_The_Mall_pz16_2906.mp4'
    , 'People_In_The_Mall_pz18_3205.mp4']

    for vid in files:
        app(vid)
