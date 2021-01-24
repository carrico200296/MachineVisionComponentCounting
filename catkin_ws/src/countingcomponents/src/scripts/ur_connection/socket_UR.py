import socket 
HOST = "192.168.12.245"
PORT = 30002

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

print("connection done")
s.send("set_digital_out(2,True)" + "\n")
s.send("movej(p[0.19662, 0.29202, 0.33466, 2.213, -2.276, -0.02], a=0.1, v=0.1)" + "\n")
print("command sent")
data = s.recv(1024)

s.close()

print("Received", repr(data))