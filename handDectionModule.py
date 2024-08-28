import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__ (self, mode=False, maxHands=2,modelComplex=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode,maxHands,modelComplex,detectionCon,trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # draw dot in when tracking hand

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # draw line in multiple hands
        return img

    def findPostion(self,img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    lmList.append([id, cx, cy])
        return lmList
    def countNumber(self,img,lmList,show=True):
        tipIds = [4, 8, 12, 16, 20]
        fingers = []
        if len(lmList) != 0:
            # for thumb
            tipsThumb = abs(lmList[tipIds[0]][1] - lmList[9][1])
            neartipsThumb = abs(lmList[tipIds[0] - 1][1] - lmList[9][1])
            if tipsThumb > neartipsThumb:
                fingers.append(1)
            else:
                fingers.append(0)
            # for the left 4 finger
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        if show:
            cv2.putText(img,"count: "+ str(sum(fingers)), (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (230, 250, 250), 2)
        return fingers

def main():
    pTime = 0
    cTime = 0

    # implement camera
    wCam,hCam = 800, 600
    cap = cv2.VideoCapture(0)
    cap.set(3,wCam)
    cap.set(4,hCam)

    detector = handDetector(maxHands=1)
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        img = detector.findHands(img)
        lmList = detector.findPostion(img)
        fingers = detector.countNumber(img,lmList)
 
        #display FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, "fps: " + str(int(fps)), (10,40),cv2.FONT_HERSHEY_PLAIN,2,(100, 250, 250),2)

        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()