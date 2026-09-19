from XEM7001_cryoDAC import *
import time


class FPGA:
    binary = [0] * 128
    def __init__(self):
         self.board = Board()
         self.restart()
    def restart(self):
        self.board.all_off()
        self.board.turn_on(10)
        time.sleep(1)
        self.board.turn_off(10)
    def set_voltage(self, pin, voltage):
            if pin > 7:
                raise Exception("Pin is invalid")
            if voltage > 1.8 or voltage < 0:
                raise Exception("Voltage is invalid")
            b = bin(int(((2**16-1)/1.8) * voltage))[2:]
            l = len(b)
            if l < 16:
                for i in range(16-l):
                    b = '0' + b
            self.binary[(pin*16):pin*16+16] = map(int, list(str(b)))
            return self.binary

    def pin_off(self, pin):
        self.set_voltage(pin, 0)

    def all_off(self):
         for i in range(0,7):
              self.pin_off(i)
    def send(self):
        parity = True
        self.board.all_off()
        self.board.turn_on(10)
        time.sleep(0.1)
        self.board.turn_off(10)
        for x in range(16):
            while self.board.read()[0] == 1:
                print("wait")
                continue
            self.board.all_off([0, 10])
            slice = self.binary[x*8:x*8+8]
            #print("slice:"+str(slice))
            for i,s in enumerate(slice):
                if int(s):
                    self.board.turn_on(i+1)
            if parity:
                self.board.turn_on(0)
            else:
                self.board.turn_off(0)
            parity = not parity
    def read(self):
        parity = True
        bits_out = []
        for x in range(16):
            #print(board.read())
            if parity:
                self.board.turn_on(9)
            else:
                self.board.turn_off(9)
            parity = not parity
            br = self.board.read()
            while (len(br) > 2 and br[1] == 1):
                print("wait")
                br = self.board.read()
                continue
            slice = self.board.read()
            bits_out.append("".join(map(str, slice)))
        return bits_out