import cv2 
import pickle

bbox_width, bbox_height = 107, 44
parkList = []

try:
    with open('CarParkPos', 'rb') as f:
        parkList = pickle.load(f)
except:
    parkList = []
    
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        parkList.append((x, y, bbox_width, bbox_height ))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(parkList):
            x1, y1 = pos[0], pos[1]
            if x1 < x < x1 + bbox_width and y1 < y < y1 + bbox_height:
                parkList.pop(i)
                
    with open('CarParkPos', 'wb') as f:
        pickle.dump(parkList, f)    
while True:
    img = cv2.imread(r'D:\UIT\AI Project\Parking spot detection and counter\Final_Project\carParkImg.png')
    print(parkList)
    for pos in parkList:
        cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + bbox_width, pos[1] + bbox_height), (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break