import numpy as np
import sys
import struct

def parse_bmp(file):
  # File header
  BMPFILEHEADER = "<2sLHHL"
  BMPFILEHEADERSIZE = struct.calcsize(BMPFILEHEADER)
  fo = open(file, "rb")
  raw_bmpfileheader = fo.read(BMPFILEHEADERSIZE)
  bmpfileheader = struct.unpack(BMPFILEHEADER, raw_bmpfileheader)
#  print bmpfileheader

  # Header
  raw = fo.read(40)
#  print struct.unpack("<LLLHHLLLLLL", raw)

  # Color Table
  raw = fo.read(4)
#  print struct.unpack("<BBBB", raw)

  # ??? data
  raw = fo.read(4)

  # Image data
  image = []
  #for i in range(28):
  raw = fo.read(112)
  FORMAT = "<" + "B"*112
  images= struct.unpack(FORMAT, raw)
  for i in range(28)[::-1]:
    for j in range(4):
      bit = images[i*4 + j]
      ByteToBitList(image, bit)
      if j == 3:
        for k in range(4):
          image.pop()
  fo.close
  return np.array(image, dtype=np.float32)

def ByteToBitList(image, bit):
  for i in range(8)[::-1]:
    if (bit & (2**i)) == 0:
      image.append(1)
    else:
      image.append(0)
  return image

if __name__ == "__main__":
  data = parse_bmp("1.bmp")
  for i in range(len(data)):
    if data[i] > 0:
      sys.stdout.write("1")
    else:
      sys.stdout.write("0")
    if (i+1) % 28 == 0:
      print("")
