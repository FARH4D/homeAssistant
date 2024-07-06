import cv2
import numpy as np
import mediapipe as mp
import pygame
from pygame import mixer
from datetime import datetime, time

class Main():
    
    def __init__(self):
        self.morningEnd = time(12, 0)
        self.afternoonEnd = time(18, 0)

    def initialise(self):

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.1,
                            min_tracking_confidence=0.1)
        self.mp_drawing = mp.solutions.drawing_utils

        self.M = np.load("calibrate/M.npy") # Load calibration matrices

        self.width, self.height = 2560, 1440

    def mainLoop(self, screen):
        
        titleFont = pygame.font.SysFont('Roboto Bold', 100)  # Defining the font that will be used
        timeFont = pygame.font.SysFont('Roboto Bold', 50)
        
        timer = pygame.USEREVENT + 1 # Set a custom event for timer
        pygame.time.set_timer(timer, 60000)
        
        currentTime = datetime.now().strftime("%H:%M")
        timerText = timeFont.render(currentTime, True, (255, 255, 255))
        
        running = True
        while running:
            now = datetime.now().time()
            currentTime = datetime.now().strftime("%H:%M")
            ret, frame = cap.read()
            if not ret:
                break
            try:
                warpedImage = cv2.warpPerspective(frame, self.M, (self.width, self.height)) # Warps the image
                
                rgbFrame = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB) # Converts to RGB
                
                results = self.hands.process(rgbFrame) # Process the frame with MediaPipe
                
                warpedImage = np.zeros((self.height, self.width, 3), np.uint8) # Creates a blank image
                
                if results.multi_hand_landmarks:
                    for handLandmarks in results.multi_hand_landmarks: # Draws hand landmarks
                        #self.mp_drawing.draw_landmarks(warpedImage, handLandmarks, self.mp_hands.HAND_CONNECTIONS)
                        indexFingerTip = handLandmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP] # Gets the position of the index finger tip
                        indexFingerX = int(indexFingerTip.x * self.width) # Normalises the coordinates to pixel coordinates (the position on my 1360x768 projector)
                        indexFingerY = int(indexFingerTip.y * self.height)
                        cv2.circle(warpedImage, (indexFingerX, indexFingerY), 15, (255, 0, 0), -1) # Draws a circle at the tip of the index finger
                        
                warpedImage = cv2.rotate(warpedImage, cv2.ROTATE_180) # Rotates the image because the webcam is upside down
                
                warpedImage = cv2.resize(warpedImage, (1360, 768)) # Resizes the image to fit the Pygame window
                
                warpedImage = cv2.flip(warpedImage, 1) # Flip the image horizontally if needed
                
                pygameImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB)
                pygameImage = np.rot90(pygameImage)
                pygameImage = pygame.surfarray.make_surface(pygameImage)
                
                if now < self.morningEnd: 
                    welcomeText = titleFont.render("Good Morning!", True, (255, 255, 255))  # Setting the title to render, along with the colour of the text
                elif self.morningEnd <= now < self.afternoonEnd: welcomeText = titleFont.render("Good Afternoon!", True, (255, 255, 255))
                else: welcomeText = titleFont.render("Good Evening!", True, (255, 255, 255))

                screen.blit(pygameImage, (0, 0)) # Displays the image in Pygame window
                screen.blit(welcomeText, (450, 50))
                screen.blit(timerText, (1150, 45))
                
                for event in pygame.event.get(): # Handles Pygame events
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.type == timer:
                        timerText = timeFont.render(currentTime, True, (255, 255, 255))
                pygame.display.update()
                
            except Exception as e:
                print(f"An error has occurred: {e}")
                continue

if __name__ == "__main__":
    pygame.init() # Initialize Pygame, pygame font and mixer (mixer is for the sounds)
    mixer.init()
    pygame.font.init()
    
    screenWidth, screenHeight = 1360, 768

    screen = pygame.display.set_mode((screenWidth, screenHeight), pygame.NOFRAME, display = 2)
    pygame.display.set_caption("Hand Tracking")

    cap = cv2.VideoCapture(0) # Initialize OpenCV and start camera
    
    mainClass = Main()
    mainClass.initialise()
    mainClass.mainLoop(screen)
    
    cap.release() # Releases resources
    pygame.quit()