import cv2
import numpy as np
import mediapipe as mp
import pygame

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.1,
                       min_tracking_confidence=0.1)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()

screenWidth, screenHeight = 1360, 768

screen = pygame.display.set_mode((screenWidth, screenHeight), pygame.NOFRAME, display = 2)
pygame.display.set_caption("Hand Tracking")

# Initialize OpenCV
cap = cv2.VideoCapture(0)

M = np.load("calibrate/M.npy") # Load calibration matrices
homography = np.load("calibrate/homography_matrix.npy")

width, height = 2560, 1440

def convert_cv2_to_pygame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rot90(image)
    image = pygame.surfarray.make_surface(image)
    return image

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break
    try:
        warpedImage = cv2.warpPerspective(frame, M, (width, height)) # Warps the image
        
        rgbFrame = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB) # Converts to RGB
        
        # Process the frame with MediaPipe
        results = hands.process(rgbFrame)
        
        warpedImage = np.zeros((height, width, 3), np.uint8) # Creates a blank image
        
        
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks: # Draws hand landmarks
                mp_drawing.draw_landmarks(warpedImage, handLandmarks, mp_hands.HAND_CONNECTIONS) 
        
        warpedImage = cv2.rotate(warpedImage, cv2.ROTATE_180) # Rotates the image because the webcam is upside down
        
        warpedImage = cv2.resize(warpedImage, (1360, 768)) # Resizes the image to fit the Pygame window
        
        warpedImage = cv2.flip(warpedImage, 1) # Flip the image horizontally if needed
        
        pygameImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB)
        pygameImage = np.rot90(pygameImage)
        pygameImage = pygame.surfarray.make_surface(pygameImage)
        
        screen.blit(pygameImage, (0, 0)) # Displays the image in Pygame window
        pygame.display.update()
        
        for event in pygame.event.get(): # Handles Pygame events
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
    except Exception as e:
        print(f"An error has occurred: {e}")
        continue

cap.release() # Releases resources
pygame.quit()
