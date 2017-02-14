import utility.utility as util

Dir = '/home/gabriel/test/'

l = []
l.append(util.readVTKPD(Dir+'OSMSC0171.truth.R_incrd_b1.vtp'))
l.append(util.readVTKPD(Dir+'OSMSC0171.truth.R_incrd.vtp'))
l.append(util.readVTKPD(Dir+'OSMSC0171.truth.R_incrd_b2.vtp'))
l.append(util.readVTKPD(Dir+'OSMSC0171.truth.R_incrd_b3.vtp'))
l.append(util.readVTKPD(Dir+'OSMSC0171.truth.R_incrd_b4.vtp'))

util.VTKScreenshotPD(l,l,'blah')
