import cv2
import mediapipe as mp

# Set up Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Main hand tracking loop
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a mirror-like effect
        image = cv2.flip(image, 1)
        
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with Mediapipe
        results = hands.process(image_rgb)
        
        # Draw hand landmarks on the image
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the resulting image
        cv2.imshow('Hand Tracking', image)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release OpenCV resources
cap.release()
cv2.destroyAllWindows()
