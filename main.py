import machine
import time


# Setup LED pin as an output
pin = machine.Pin(2, machine.Pin.OUT)

while True:
  pin.on()
  time.sleep(5)
  pin.off()
  time.sleep(5)