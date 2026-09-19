import sys
sys.path.append("/home/electron/edes/edes/experiments/devices/FPGA/")
import ok
from DacConfiguration import hardwareConfiguration

class api(object):
    '''class containing all commands for interfacing with the fpga'''
    def __init__(self):
        self.xem = None
        self.okDeviceID = hardwareConfiguration.okDeviceID
        self.okDeviceFile = hardwareConfiguration.okDeviceFile
        
    def checkConnection(self):
        if self.xem is None: raise Exception("FPGA not connected")

    def connectOKBoard(self):
        # 1. Initialize the modern device discovery manager
        dev_manager = ok.FrontPanelDevices()
        
        # 2. Get the count of connected devices
        count = dev_manager.GetCount()
        print(f"Found {count} connected Opal Kelly modules")
        
        for i in range(count):
            serial = dev_manager.GetSerial(i)
            
            # 3. Open the device (returns an okCFrontPanel-based proxy object)
            tmp_dev = dev_manager.Open(serial)
            
            if tmp_dev is not None:
                # 4. Instantiate an empty DeviceInfo structure
                dev_info = ok.okTDeviceInfo()
                
                # 5. Fill the structure from the device
                tmp_dev.GetDeviceInfo(dev_info)
                
                # 6. Read the custom string ID assigned to the device
                iden = dev_info.deviceID
                if iden == self.okDeviceID:
                    self.xem = tmp_dev
                    print(f'Connected to {iden} with serial {serial}')
                    self.programOKBoard()
                    return True
                else:
                    # Close the device if it's the wrong one
                    # Note: Depending on your exact build, the smart pointer might handle 
                    # cleanup, but explicit Close() or letting it go out of scope protects it.
                    tmp_dev.Close()
                    
        print("Device matching targeted ID not found.")
        return False
    
    def programOKBoard(self):
        prog = self.xem.ConfigureFPGA(self.okDeviceFile)
        print(prog)
        if prog: raise("Not able to program FPGA")
        pll = ok.PLL22150()
        self.xem.GetEepromPLL22150Configuration(pll)
        pll.SetDiv1(pll.DivSrc_VCO,4)
        self.xem.SetPLL22150Configuration(pll)
        
    def programBoard(self, sequence):
        self.xem.WriteToBlockPipeIn(0x80, 2, sequence)
    
    def resetFIFODAC(self):
        self.xem.ActivateTriggerIn(0x40,8)  
        
    def setDACVoltage(self, volstr):
        self.xem.WriteToBlockPipeIn(0x82, 2, volstr)   
