import socket
# AF_INET == IPV4
# SOCK_STREAM == TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))

msg = s.recv(1024)
msg_received = msg.decode("utf-8")
print(msg_received)
s.close()