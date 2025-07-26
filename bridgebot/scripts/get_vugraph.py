import pycurl
import io
import time

for i in range(66138,68294):
    buf = io.BytesIO()
    c = pycurl.Curl()
    print("downloading #{}".format(i))
    c.setopt(c.URL, "https://www.bridgebase.com/tools/vugraph_linfetch.php?id={}".format(i))
    c.setopt(c.WRITEDATA, buf)
    c.perform()
    c.close()
    body = buf.getvalue()
    with open("/home/ubuntu/bridge/records/{:06d}.lin".format(i), "w") as f:
        f.write(body.decode('utf-8'))
