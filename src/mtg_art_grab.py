import pandas as pd
import requests
import time
from glob import glob
import cv2
import os

cards = pd.read_json(
    "input/magic-the-gathering-cards/scryfall-artwork-cards.json"
)
total = 0
for uri in cards["image_uris"].dropna():
    try:
        file_path = f"input/magic-the-gathering-art/{total:05}.jpg"
        if os.path.exists(file_path):
            total += 1
            continue
        r = requests.get(uri["small"], timeout=2.5)
        with open(file_path, "wb") as f:
            f.write(r.content)
        print(f"downloading {total}")
        total += 1
    except:
        print(f"error downloading {total}")
    time.sleep(0.05)

for file_path in glob("input/magic-the-gathering-art/*/*.jpg"):
	delete = False
	try:
		image = cv2.imread(file_path)
		if image is None:
			delete = True
	except:
		delete = True
	if delete:
		os.remove(file_path)
