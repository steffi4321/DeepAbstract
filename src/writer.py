import os
"""
This class is used in main_eran for writing results into timeFile
"""

class localwriter:
    def __init__(self, timeFile, precisionFile=None, unique=False):
        if unique:
            mainName = timeFile.split('.')[0]
            ending = timeFile.split('.')[1]
            i =0
            while os.path.isfile(mainName+'_'+str(i)+"."+ending):
                i += 1
            self.indx = i
            timeFile = mainName+'_'+str(i)+"."+ending
        elif os.path.isfile(timeFile):
            print("ERROR FILE",timeFile,"ALREADY EXISTS")
            raise ValueError
        self.timeFile = timeFile
        if not precisionFile is None:
            if os.path.isfile(precisionFile):
                print("ERROR FILE",precisionFile,"ALREADY EXISTS")
                raise ValueError
        self.precisionFile = precisionFile
        return

    def printT(self, st):
        with open(self.timeFile, 'a+') as f:
            f.write(st)

    def printP(self, st):
        if self.precisionFile is None:
            raise ValueError
        with open(self.precisionFile, 'a+') as f:
            f.write(st)
