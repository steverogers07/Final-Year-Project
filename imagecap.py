# from picamera import PiCamera
# from time import sleep

# def take_photo():
    
#     camera = PiCamera()

#     camera.start_preview()
#     sleep(5)
#     camera.capture('image.jpg')
#     camera.stop_preview()
    
# take_photo()
import cv2
def take_Photo():
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame) 
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                #img_new = cv2.imshow("Captured Image", img_new)  
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        except(KeyboardInterrupt):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
# take_Photo()
