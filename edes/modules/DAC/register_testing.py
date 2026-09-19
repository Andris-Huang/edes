from FPGA import *
from XEM7001_cryoDAC import *
import time

board = FPGA()

board.set_voltage(0, 1.8/10 * 4)

start = time.time()
board.send()

# board = Board()
# board.all_off()
# board.turn_on(10)
# time.sleep(1)
# board.turn_off(10)
# board.turn_on(1)

print("".join(board.read()))
print("start")
board.board.dev.Close()

        














# from XEM7001_cryoDAC import *
# import time

# board = Board()
# parity = True
# bits_in = [10101010] * 16
# board.all_off()
# board.turn_on(10)
# time.sleep(1)
# board.turn_off(10)
# for x in range(16):
#     while board.read()[0] == 1:
#         print("wait")
#         continue
#     board.all_off([0, 10])
#     slice = bits_in[x]
#     #print("slice:"+str(slice))
#     for i,s in enumerate(str(slice)):
#         if int(s):
#             board.turn_on(i+1)
#     if parity:
#         board.turn_on(0)
#     else:
#         board.turn_off(0)
#     parity = not parity


# parity = True
# bits_out = []
# for x in range(16):
#     #print(board.read())
#     if parity:
#         board.turn_on(9)
#     else:
#         board.turn_off(9)
#     parity = not parity
#     br = board.read()
#     while (len(br) > 2 and br[1] == 1):
#         print("wait")
#         br = board.read()
#         continue
#     slice = board.read()
#     bits_out.append("".join(map(str, slice)))

# print(bits_out)
