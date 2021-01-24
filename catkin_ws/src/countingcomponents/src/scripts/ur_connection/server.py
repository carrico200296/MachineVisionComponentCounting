import socket
# Create the socket
# AF_INET == IPV4
# SOCK_STREAM == TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 50000))
s.listen(5)

while True:
    clientsocket, address = s.accept()
    print("Connection from {} has been established!".format(address))
    clientsocket.send(bytes("15\n", "utf-8"))
    clientsocket.close()