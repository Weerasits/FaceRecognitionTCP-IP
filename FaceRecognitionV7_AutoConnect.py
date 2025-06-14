import face_recognition as face 
import numpy as np 
import cv2
import socket
import time
import threading

# ===== ตั้งค่าการเชื่อมต่อ TCP Client =====
SERVER_IP = '127.0.0.1'  # หรือ IP ของ server
SERVER_PORT = 6601

client_socket = None
connected = False

# ===== ฟังก์ชันเชื่อมต่อ server แบบ auto retry =====
def connect_to_server():
    global client_socket, connected
    while not connected:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((SERVER_IP, SERVER_PORT))
            connected = True
            print("✅ Connected to server")
        except Exception:
            print("⏳ Server not available, retrying...")
            time.sleep(1)

# ===== โหลดภาพใบหน้าที่รู้จัก =====
known_face_encodings = []
known_face_names = []

# === Weerasit ===
try:
    known_image = face.load_image_file("weerasit.jpg")
    known_encoding = face.face_encodings(known_image)[0]
    known_face_encodings.append(known_encoding)
    known_face_names.append("Weerasit")
except IndexError:
    print("❌ ไม่สามารถสร้าง encoding จาก weerasit.jpg ได้")

# === Aphisit ===
try:
    aphisit_image = face.load_image_file("aphisit.jpg")
    aphisit_encoding = face.face_encodings(aphisit_image)[0]
    known_face_encodings.append(aphisit_encoding)
    known_face_names.append("Aphisit")
except IndexError:
    print("❌ ไม่สามารถสร้าง encoding จาก Aphisit.jpg ได้")

if not known_face_encodings:
    print("❌ ไม่มีใบหน้าใดที่โหลดได้เลย")
    exit()

# ===== เริ่มกล้อง =====
video_capture = cv2.VideoCapture(0)
process_this_frame = True

# เริ่ม thread เชื่อมต่อ server
threading.Thread(target=connect_to_server, daemon=True).start()

# ตัวแปรจำค่าก่อนหน้าไว้สำหรับวาดกรอบ
last_face_locations = []
last_face_names = []
last_face_percent = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = []
    face_names = []
    face_percent = []

    if process_this_frame:
        face_locations = face.face_locations(rgb_small_frame, model="hog")

        if face_locations:
            try:
                encodings = face.face_encodings(rgb_small_frame, face_locations)
            except Exception as e:
                print("❌ Encoding error:", e)
                encodings = []

            for face_encoding in encodings:
                distances = face.face_distance(known_face_encodings, face_encoding)
                best_idx = np.argmin(distances)
                confidence = 1 - distances[best_idx]

                if confidence >= 0.5:
                    name = known_face_names[best_idx]
                    percent = round(confidence * 100, 2)
                else:
                    name = "UNKNOWN"
                    percent = 0.0

                face_names.append(name)
                face_percent.append(percent)

                if connected:
                    try:
                        client_socket.sendall(name.encode())
                        print(f"📤 ส่งชื่อ: {name}")
                    except Exception as e:
                        print(f"❌ ส่งชื่อไม่สำเร็จ: {e}")
                        connected = False
                        threading.Thread(target=connect_to_server, daemon=True).start()

            # เก็บค่าล่าสุดไว้ใช้วาดกรอบ
            last_face_locations = face_locations
            last_face_names = face_names
            last_face_percent = face_percent
        else:
            # ไม่มีใบหน้า → ล้างค่าที่จำไว้
            last_face_locations = []
            last_face_names = []
            last_face_percent = []

    process_this_frame = not process_this_frame

    # วาดกรอบจากข้อมูลล่าสุดที่มี (ถ้ามี)
    for (top, right, bottom, left), name, percent in zip(last_face_locations, last_face_names, last_face_percent):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
        cv2.putText(frame, f"{name} ({percent:.2f}%)", (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== ปิดการเชื่อมต่อ =====
if connected:
    client_socket.close()
video_capture.release()
cv2.destroyAllWindows()
