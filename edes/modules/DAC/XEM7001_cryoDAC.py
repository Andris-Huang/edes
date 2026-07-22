#import sys
#sys.path.append("/home/electron/edes/edes/experiments/devices/FPGA/")
import ok


class Pin:
    """
    A pin on the XEM7001 board that can be turned on and off.
    """
    def __init__(self, num):
        self.number = num
        self.on = False
        

class Board:
    """
    A board with 16 pins that can be turned on and off as well as a pin that can read in. The pins are connected to a DAC.
    """
    binary = [0] * 16
    
    def __init__(self):
        self.pins = [Pin(x) for x in range(16)]
        devices = ok.FrontPanelDevices()

        self.dev = devices.Open()
        self.dev.ConfigureFPGA('/home/electron/fpga/r-2r_dac_1.bit')
        self.dp = self.dev.GetFPGADataPortClassic()


    def turn_on(self, pin):
        self.binary[pin] = 1
        self.pins[pin].on = True
        self.update_pins()
    
    def turn_off(self, pin):
        self.binary[pin] = 0
        self.pins[pin].on = False
        self.update_pins()
    
    def update_pins(self):
        val = 0
        binary_string = "".join(map(str, self.binary))
        val = int(binary_string[::-1], 2)
        #print(bin(val))
        #print(binary_string, val)
        self.dp.SetWireInValue(0x03, val)
        self.dp.UpdateWireIns()
        
    def read(self):
        self.dp.UpdateWireOuts()
        val = self.dp.GetWireOutValue(0x21)
        b = str(bin(val))[2:][::-1][2:]
        l = len(b)
        if l < 8:
            for i in range(8-l):
                b += '0'
        return b
    
    def all_off(self, excluding = []):
        for pin in range(16):
            if pin not in excluding:
                self.turn_off(pin)
    

if __name__ == "__main__":
    board = Board() 
    