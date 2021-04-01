import sys
import os
import bluetooth

""" Do this before executing this script:

* Ensure that the bluetooth-daemon runs in compatibility mode. (option -C for bluetoothd, check via systemctl)
* Ensure that your local user and /var/run/sdp have the same group. If not, create a group and change permissions.
  As this file is re-created after every reboot, maybe add an additional unit or and entry in .bash_profile that
  doe the group-change for you.
* execute "sudo hciconfig hci0 piscan" in shell
* Finally, run the script.
"""


server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]

uuid = "53eef5bf-6702-4ab6-bcf3-ae02a401d70e"

bluetooth.advertise_service(server_sock, "SampleServer", service_id=uuid,
                            service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE],
                            # protocols=[bluetooth.OBEX_UUID]
                            )

print("Waiting for connection on RFCOMM channel", port)

client_sock, client_info = server_sock.accept()
print("Accepted connection from", client_info)

try:
    while True:
        data = client_sock.recv(1024)
        if not data:
            break
        print("Received", data)
except OSError:
    pass

print("Disconnected.")

client_sock.close()
server_sock.close()
print("All done.")
