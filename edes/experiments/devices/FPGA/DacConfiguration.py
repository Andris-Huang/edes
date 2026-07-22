class channelConfiguration(object):
    """
    Stores complete information for each DAC channel
    """


    def __init__(self, dacChannelNumber, trapElectrodeNumber = None, smaOutNumber = None, name = None, boardVoltageRange = (-28, 28), allowedVoltageRange = (-28, 28)):
    # def __init__(self, dacChannelNumber, trapElectrodeNumber = None, smaOutNumber = None, name = None, boardVoltageRange = (-10, 10), allowedVoltageRange = (-10, 10)):
        self.dacChannelNumber = dacChannelNumber
        self.trapElectrodeNumber = trapElectrodeNumber
        self.smaOutNumber = smaOutNumber
        self.boardVoltageRange = boardVoltageRange
        self.allowedVoltageRange = allowedVoltageRange
        if (name == None) & (trapElectrodeNumber != None):
            self.name = str(trapElectrodeNumber).zfill(2)
        else:
            self.name = name

    def computeDigitalVoltage(self, analogVoltage):
        return int(round(sum([ self.calibration[n] * analogVoltage ** n for n in range(len(self.calibration)) ])))

class hardwareConfiguration(object):
    EXPNAME = 'electron'
    default_multipoles = ['Ex', 'Ey', 'Ez', 'U1', 'U2', 'U3', 'U4', 'U5']
    okDeviceID = 'Opal Kelly XEM7001'#'DAC Controller'
    okDeviceFile = 'control_noninverted.bit'
    centerElectrode = False #write False if no Centerelectrode
    PREC_BITS = 16
    pulseTriggered = True
    maxCache = 126
    filter_RC = 5e4 * 4e-7

    elec_dict = {
        'bl1': channelConfiguration(25, trapElectrodeNumber='bl1'),
        'bl2': channelConfiguration(15, trapElectrodeNumber='bl2'),
        'bl3': channelConfiguration(5, trapElectrodeNumber='bl3'),
        'bl4': channelConfiguration(13, trapElectrodeNumber='bl4'),
        'bl5': channelConfiguration(22, trapElectrodeNumber='bl5'), 
        'br1': channelConfiguration(4, trapElectrodeNumber='br1'),
        'br2': channelConfiguration(10, trapElectrodeNumber='br2'),
        'br3': channelConfiguration(24, trapElectrodeNumber='br3'),
        'br4': channelConfiguration(3, trapElectrodeNumber='br4'),
        'br5': channelConfiguration(9, trapElectrodeNumber='br5'),
        # 'tg': channelConfiguration( 27 , trapElectrodeNumber='tg'),
        'tl1': channelConfiguration(19, trapElectrodeNumber='tl1'),
        'tl2': channelConfiguration(21, trapElectrodeNumber='tl2'),
        'tl3': channelConfiguration(16, trapElectrodeNumber='tl3'),
        'tl4': channelConfiguration(8, trapElectrodeNumber='tl4'),
        'tl5': channelConfiguration(28, trapElectrodeNumber='tl5'),
        'tr1': channelConfiguration(26, trapElectrodeNumber='tr1'), 
        'tr2': channelConfiguration(12, trapElectrodeNumber='tr2'),
        'tr3': channelConfiguration(11, trapElectrodeNumber='tr3'),
        'tr4': channelConfiguration(17, trapElectrodeNumber='tr4'),
        'tr5': channelConfiguration(23, trapElectrodeNumber='tr5'),
    
        }

    '''
    #neha and izze

 
    elec_dict = {
        'bl1': channelConfiguration(23, trapElectrodeNumber='bl1'),
        'bl2': channelConfiguration(22, trapElectrodeNumber='bl2'),
        'bl3': channelConfiguration(21, trapElectrodeNumber='bl3'),
        'bl4': channelConfiguration(4, trapElectrodeNumber='bl4'),
        'bl5': channelConfiguration(19, trapElectrodeNumber='bl5'), 
        'br1': channelConfiguration(28, trapElectrodeNumber='br1'),
        'br2': channelConfiguration(8, trapElectrodeNumber='br2'),
        'br3': channelConfiguration(26, trapElectrodeNumber='br3'),
        'br4': channelConfiguration(25, trapElectrodeNumber='br4'),
        'br5': channelConfiguration(24, trapElectrodeNumber='br5'),
        # 'tg': channelConfiguration( 27 , trapElectrodeNumber='tg'),
        'tl1': channelConfiguration(3, trapElectrodeNumber='tl1'),
        'tl2': channelConfiguration(5, trapElectrodeNumber='tl2'),
        'tl3': channelConfiguration(16, trapElectrodeNumber='tl3'),
        'tl4': channelConfiguration(17, trapElectrodeNumber='tl4'),
        'tl5': channelConfiguration(15, trapElectrodeNumber='tl5'),
        'tr1': channelConfiguration(13, trapElectrodeNumber='tr1'), 
        'tr2': channelConfiguration(12, trapElectrodeNumber='tr2'),
        'tr3': channelConfiguration(11, trapElectrodeNumber='tr3'),
        'tr4': channelConfiguration(10, trapElectrodeNumber='tr4'),
        'tr5': channelConfiguration(9, trapElectrodeNumber='tr5'),
    
        }
    '''
    '''
    #andris pin matching
    elec_dict = {
        'bl1': channelConfiguration(15, trapElectrodeNumber='bl1'),
        'bl2': channelConfiguration(8, trapElectrodeNumber='bl2'),
        'bl3': channelConfiguration(13, trapElectrodeNumber='bl3'),
        'bl4': channelConfiguration(3, trapElectrodeNumber='bl4'),
        'bl5': channelConfiguration(9, trapElectrodeNumber='bl5'), 
        'br1': channelConfiguration(6, trapElectrodeNumber='br1'),
        'br2': channelConfiguration(16, trapElectrodeNumber='br2'),
        'br3': channelConfiguration(28, trapElectrodeNumber='br3'),
        'br4': channelConfiguration(17, trapElectrodeNumber='br4'),
        'br5': channelConfiguration(5, trapElectrodeNumber='br5'),
        # 'tg': channelConfiguration( 27 , trapElectrodeNumber='tg'),
        'tl1': channelConfiguration(24, trapElectrodeNumber='tl1'),
        'tl2': channelConfiguration(22, trapElectrodeNumber='tl2'),
        'tl3': channelConfiguration(12, trapElectrodeNumber='tl3'),
        'tl4': channelConfiguration(21, trapElectrodeNumber='tl4'),
        'tl5': channelConfiguration(10, trapElectrodeNumber='tl5'),
        'tr1': channelConfiguration(19, trapElectrodeNumber='tr1'), 
        'tr2': channelConfiguration(26, trapElectrodeNumber='tr2'),
        'tr3': channelConfiguration(11, trapElectrodeNumber='tr3'),
        'tr4': channelConfiguration(4, trapElectrodeNumber='tr4'),
        'tr5': channelConfiguration(25, trapElectrodeNumber='tr5'),
    
        }
    '''

    #in case we need this, 8/7/2024
    '''
    elec_dict = {
        'bl1': channelConfiguration(23, trapElectrodeNumber='bl1'),
        'bl2': channelConfiguration(22, trapElectrodeNumber='bl2'),
        'bl3': channelConfiguration(21, trapElectrodeNumber='bl3'),
        'bl4': channelConfiguration(4, trapElectrodeNumber='bl4'),
        'bl5': channelConfiguration(19, trapElectrodeNumber='bl5'), 
        'br1': channelConfiguration(28, trapElectrodeNumber='br1'),
        'br2': channelConfiguration(8, trapElectrodeNumber='br2'),
        # 'br3': channelConfiguration(21, trapElectrodeNumber='bl3'),
        'br4': channelConfiguration(25, trapElectrodeNumber='br4'),
        'br5': channelConfiguration(24, trapElectrodeNumber='br5'),
        # 'tg': channelConfiguration( 27 , trapElectrodeNumber='tg'),
        'tl1': channelConfiguration(3, trapElectrodeNumber='tl1'),
        'tl2': channelConfiguration(5, trapElectrodeNumber='tl2'),
        'tl3': channelConfiguration(17, trapElectrodeNumber='tl3'),
        'tl4': channelConfiguration(16, trapElectrodeNumber='tl4'),
        'tl5': channelConfiguration(15, trapElectrodeNumber='tl5'),
        'tr1': channelConfiguration(13, trapElectrodeNumber='tr1'), 
        'tr2': channelConfiguration(12, trapElectrodeNumber='tr2'),
        'tr3': channelConfiguration(11, trapElectrodeNumber='tr3'),
        'tr4': channelConfiguration(10, trapElectrodeNumber='tr4'),
        'tr5': channelConfiguration(9, trapElectrodeNumber='tr5'),
    
        }
    '''


    '''
    #block commented out 7/25/24 to use merged bl3-br3 cfile 
    elec_dict = {
        'bl1': channelConfiguration(28, trapElectrodeNumber='br1'),
        'bl2': channelConfiguration(8, trapElectrodeNumber='br2'),
        'bl3': channelConfiguration(26, trapElectrodeNumber='br3'),
        'bl4': channelConfiguration(25, trapElectrodeNumber='br4'),
        'bl5': channelConfiguration(24, trapElectrodeNumber='br5'), 
        'br1': channelConfiguration(23, trapElectrodeNumber='bl1'),
        'br2': channelConfiguration(22, trapElectrodeNumber='bl2'),
        'br3': channelConfiguration(21, trapElectrodeNumber='bl3'),
        'br4': channelConfiguration(4, trapElectrodeNumber='bl4'),
        'br5': channelConfiguration(19, trapElectrodeNumber='bl5'),
	    # 'tg': channelConfiguration( 27 , trapElectrodeNumber='tg'),
        'tl1': channelConfiguration(5, trapElectrodeNumber='tl1'),
        'tl2': channelConfiguration(17, trapElectrodeNumber='tl2'),
        'tl3': channelConfiguration(16, trapElectrodeNumber='tl3'),
        'tl4': channelConfiguration(15, trapElectrodeNumber='tl4'),
        'tl5': channelConfiguration(3, trapElectrodeNumber='tl5'),
        'tr1': channelConfiguration(13, trapElectrodeNumber='tr1'), 
        'tr2': channelConfiguration(12, trapElectrodeNumber='tr2'),
        'tr3': channelConfiguration(11, trapElectrodeNumber='tr3'),
        'tr4': channelConfiguration(10, trapElectrodeNumber='tr4'),
        'tr5': channelConfiguration(9, trapElectrodeNumber='tr5'),
    
        }
    '''

    notused_dict = {
        
        '22': channelConfiguration( 6, trapElectrodeNumber=22),
        'GND': channelConfiguration(7, trapElectrodeNumber='GND'),
 
        
               }#D3 and D9 GND

    sma_dict = { }
        # 'RF bias': channelConfiguration(25, smaOutNumber=1, name='RF bias', boardVoltageRange=(-40., 40.), allowedVoltageRange=(-2.0, 0))
        # }
        # 'bl1': channelConfiguration(15, trapElectrodeNumber=15),
        # 'bl2': channelConfiguration(13, trapElectrodeNumber=14),
        # 'bl3': channelConfiguration(16, trapElectrodeNumber=13),
        # 'bl4': channelConfiguration(18, trapElectrodeNumber=12),
        # 'bl5': channelConfiguration(17, trapElectrodeNumber=11), 
        # 'br1': channelConfiguration(25, trapElectrodeNumber=20),
        # 'br2': channelConfiguration(24, trapElectrodeNumber=19),
        # 'br3': channelConfiguration(22, trapElectrodeNumber=18),
        # 'br4': channelConfiguration(23, trapElectrodeNumber=17),
        # 'br5': channelConfiguration(19, trapElectrodeNumber=16),
        # 'tg': channelConfiguration(4, trapElectrodeNumber=21),
        # 'tl1': channelConfiguration(3, trapElectrodeNumber=10),#amp board ground
        # 'tl2': channelConfiguration(1, trapElectrodeNumber=9),#higher voltage
        # 'tl3': channelConfiguration(2, trapElectrodeNumber=8),
        # 'tl4': channelConfiguration(6, trapElectrodeNumber=7),
        # 'tl5': channelConfiguration(5, trapElectrodeNumber=6),
        # 'tr1': channelConfiguration(10, trapElectrodeNumber=5), 
        # 'tr2': channelConfiguration(12, trapElectrodeNumber=4),
        # 'tr3': channelConfiguration(26, trapElectrodeNumber=3),
        # 'tr4': channelConfiguration(8, trapElectrodeNumber=2),
        # 'tr5': channelConfiguration(7, trapElectrodeNumber=1),
