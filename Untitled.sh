python - << 'PY'
import pickle
from pprint import pprint

with open('known_faces.pkl', 'rb') as f:
    data = pickle.load(f)

pprint(data)
PY
