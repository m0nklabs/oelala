import hashlib
import os

key = "opmobiel_1234567890abcdef12345678"
h = hashlib.sha256(key.encode()).hexdigest()
print(h)
