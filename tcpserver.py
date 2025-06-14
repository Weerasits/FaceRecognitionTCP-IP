import socket

HOST = '127.0.0.1'  # ให้รับจากทุก IP
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"🟢 รอการเชื่อมต่อที่พอร์ต {PORT}...")

conn, addr = server_socket.accept()
print(f"✅ เชื่อมต่อจาก: {addr}")

try:
    while True:
        data = conn.recv(1024)
        if not data:
            break
        name = data.decode()
        print(f"📥 รับชื่อ: {name}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")
finally:
    conn.close()
    server_socket.close()
    print("🛑 ปิดการเชื่อมต่อแล้ว")
