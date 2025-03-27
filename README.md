
# 🖐️ Hand Gesture-Based AI Drawing & Vision Assistant
![Screenshot 2025-03-27 214024](https://github.com/user-attachments/assets/969d966d-9dc2-4a63-a40b-5fa167929787)
Here’s a clean, professional, and well-structured `README.md` file for your project, including project description, features, setup instructions, and usage:

---

This project is a **Hand Gesture-Controlled AI Drawing Application** integrated with **Groq's Vision LLM API**.  
It allows you to **draw on the screen using hand gestures** and automatically send the drawn image to the **Groq Vision Model (Llama 3.2 11B Vision Preview)** for real-time AI-based analysis and response.

---

## 🚀 Features

- **Hand Gesture Detection** using OpenCV & CVZone
- **Gesture-Based Drawing & Canvas Clearing**
  - ✏️ *Index Finger Up → Draw*
  - ✌️ *Index + Middle Finger Up → Pause Drawing*
  - 👍 *Thumb Up → Clear Canvas*
  - 🖐️ *All Fingers Except Pinky Up → Send Drawing to Groq AI for Response*
- **Groq API Integration** to analyze your drawn image
- Real-time **Webcam Feed + Drawing Canvas Overlay**
- Clean and responsive **Streamlit UI**

---

## 📂 Folder Structure

```
project/
│
├── app.py               # Main Streamlit Application
├── .env                 # Environment Variables (contains GROQ_API_KEY)
├── requirements.txt     # Python dependencies
└── README.md            # Project Documentation
```

---

## ⚙️ Tech Stack & Libraries

- **Python**
- **Streamlit** – For frontend UI
- **OpenCV** – For webcam feed and image processing
- **CVZone** – For Hand Gesture Detection
- **NumPy** – For image manipulation
- **PIL** – For image format conversion
- **Groq API** – Llama 3.2 Vision Model for AI analysis
- **dotenv** – For API key management

---

## 🌐 Environment Variables

Create a `.env` file in the root directory and add your **Groq API Key**:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🧩 Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/hand-gesture-ai-drawing.git
cd hand-gesture-ai-drawing
```

2. **Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Add Your API Key**
Create a `.env` file and paste:
```
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the Application**
```bash
streamlit run app.py
```

---



![Screenshot 2025-03-27 214024](https://github.com/user-attachments/assets/0e1a1865-5683-40a7-bb53-709334609711)

## 🎯 How to Use

| Gesture                       | Action                                             |
|-------------------------------|----------------------------------------------------|
| **Index Finger Up**           | Draw on the screen                                  |
| **Index + Middle Finger Up**  | Pause drawing                                      |
| **Thumb Up**                  | Clear the entire canvas                           |
| **All Fingers Except Pinky Up** | Send the canvas image to Groq AI for analysis |

Once you draw a shape like a **right-angle triangle**, the app will send it to Groq's Llama-3 Vision Model and respond with the **hypotenuse** value in real-time.

---

## 🖼️ Example Workflow

```
Hand Gesture → Draw on Screen → Show Drawing → Raise all fingers (except pinky) → 
↓
Image Sent to Groq Vision API → 
↓
AI Response Displayed in Streamlit Panel
```

---

## 📝 License

This project is for educational and personal use only.  
For commercial use, please check Groq API's license & terms.

---

If you want, I can also prepare a **`requirements.txt` file** and **example `.env` template** along with this README.  
**Shall I?**







