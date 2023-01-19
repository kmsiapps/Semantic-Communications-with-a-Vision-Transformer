CLIENT_ADDR = "0.0.0.0"
CLIENT_PORT = 50000
# Client address and port
# Client relays encoded/received constellations between USRP and the server and visualizes the results.

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
# Server address and port
# Server conducts encode/decode functions of the given images/constellations with its GPU.

NORMALIZE_CONSTANT = 10
# note: inversely proportional to signal power

TEMP_DIRECTORY = './temp'
# directory to save temporary files
# e.g., encoded/received constellations (numpy array, .npz) and images (.jpg or .png)