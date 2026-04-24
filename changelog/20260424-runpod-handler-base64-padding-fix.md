### Fixed

- RunPod workers (`wan22`, `i2i`, `ltx23`): `save_input_images()` now defensively
  strips `data:` URL prefixes, removes whitespace/newlines, and pads base64 input
  to a multiple of 4 before decoding. Fixes `binascii.Error: Incorrect padding`
  errors that crashed handler jobs when callers sent raw data URLs or trimmed
  base64 strings.
