import os
import cv2
import numpy as np
from groq import Groq
from PIL import Image
from io import BytesIO
import base64
from cvzone.HandTrackingModule import HandDetector

# Suppress TensorFlow Lite warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Your Groq API key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1480)  # Set width
cap.set(4, 820)   # Set height

# Initialize the HandDetector class
detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7, minTrackCon=0.5)

def get_hand_info(img):
    """
    Detect hand landmarks and finger states.
    """
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]  # Get the first detected hand
        lmList = hand["lmList"]  # List of 21 landmarks
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None, None

def draw(info, prev_pos, canvas):
    """
    Draw lines on the canvas based on index finger movement.
    """
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][0:2]  # Index finger tip coordinates
        if prev_pos is not None:
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
        prev_pos = current_pos
    elif fingers == [0, 1, 1, 0, 0]:  # Index and middle finger up (Pause drawing)
        prev_pos = None  # Reset previous position to avoid drawing
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up (clear canvas)
        canvas = np.zeros_like(canvas)
        prev_pos = None
    return prev_pos, canvas

def send_to_ai(canvas, fingers):
    """
    Convert the canvas to a base64 image and send it to the Groq API for processing.
    """
    if fingers == [1, 1, 1, 1, 0]:  # All fingers up except the pinky
        try:
            # Convert the canvas (NumPy array) to a PIL Image
            pil_image = Image.fromarray(canvas)

            # Convert the PIL Image to a byte array
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            # Encode the byte array to base64
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # Send the base64-encoded image as part of the API request
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",  # Replace with your desired model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "find hypotenuse of this right angle triangle and answer in one word"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=200,
                top_p=1,
                stream=False,
                stop=None
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error sending image to Groq API:", e)
            return None
    return None

prev_pos = None
canvas = None
output_text = ""

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    # Initialize the canvas if it's not already set
    if canvas is None:
        canvas = np.zeros_like(img)

    # Get hand information
    info = get_hand_info(img)
    if info[0]:  # Check if hand landmarks are detected
        prev_pos, canvas = draw(info, prev_pos, canvas)
        response = send_to_ai(canvas, info[0])
        if response:
            output_text = response

    # Overlay the canvas on the webcam feed
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    cv2.putText(image_combined, output_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), 2)
    cv2.imshow("Hand Tracking and AI Response", image_combined)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
