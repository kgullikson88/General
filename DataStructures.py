import numpy
import sys

class GridSearchOut:
  def __init__(self, size):
    self.wave = numpy.zeros(size)
    self.rect = numpy.zeros(size)
    self.opt = numpy.zeros(size)
    self.recterr = numpy.zeros(size)
    self.opterr = numpy.zeros(size)
    self.cont = numpy.zeros(size)
  def copy(self):
    copy = GridSearchOut(self.wave.size)
    copy.wave = self.wave.copy()
    copy.rect = self.rect.copy()
    copy.opt = self.opt.copy()
    copy.recterr = self.recterr.copy()
    copy.opterr = self.opterr.copy()
    copy.cont = self.cont.copy()
    return copy
  def size(self):
    return self.wave.size
  def ToXypoint(self):
    output = xypoint(self.wave.size)
    output.x = self.wave.copy()
    output.y = self.opt.copy()
    output.err = self.opterr.copy()
    output.cont = self.cont.copy()
    return output



class xypoint:
  def __init__(self, size=100, x=None, y=None, cont=None, err=None):
    if x != None:
      size = x.size
    if y != None:
      size = y.size
    if cont != None:
      size = cont.size
    if err != None:
      size = err.size
      
    if x == None:
      self.x = numpy.zeros(size)
    else:
      self.x = x.copy()
    if y == None:
      self.y = numpy.zeros(size)
    else:
      self.y = y.copy()
    if cont == None:
      self.cont = numpy.ones(size)
    else:
      self.cont = cont.copy()
    if err == None:
      self.err = numpy.sqrt(self.y)
    else:
      self.err = err.copy()
      self.err[self.err <=0] = 1e9    #Making sure we don't divide by zero
      
  def copy(self):
      copy = xypoint(self.x.size)
      copy.x = self.x.copy()
      copy.y = self.y.copy()
      copy.cont = self.cont.copy()
      copy.err = self.err.copy()
      return copy
  def size(self):
      return self.x.size
  def ToGridSearchOut(self):
    output = GridSearchOut(self.x.size)
    output.wave = self.x.copy()
    output.opt = self.y.copy()
    output.opterr = self.err.copy()
    output.cont = self.cont.copy()
    return output


def ReadGridSearchFile(filename, headerflag=False):
  infile = open(filename)
  lines = infile.readlines()
  infile.close()
  wave = []
  rect = []
  opt = []
  recterror = []
  opterror = []
  cont = []
  chips = []
  all_headers = []
  header = []
  for line in lines:
    if not (line.startswith("#") or line == "\n"):
      try:
        wave.append(line.split()[0])
        rect.append(line.split()[1])
        opt.append(line.split()[2])
        recterror.append(line.split()[3])
        opterror.append(line.split()[4])
        cont.append(line.split()[5])
      except IndexError:
        print "Format incorrect for file: ", filename, "\nExitting"
        sys.exit(0)
    elif line == "\n" and len(wave) > 0:
      chip = GridSearchOut(len(wave))
      all_headers.append(header)
      header = []
      chip.wave = numpy.array(wave).astype(float)
      chip.rect = numpy.array(rect).astype(float)
      chip.opt = numpy.array(opt).astype(float)
      chip.recterr = numpy.array(recterror).astype(float)
      chip.opterr = numpy.array(opterror).astype(float)
      chip.cont = numpy.array(cont).astype(float)
      wave = []
      rect = []
      opt = []
      recterror = []
      opterror = []
      cont = []
      chips.append(chip)
    elif line != "\n":
      header.append(line)
  if headerflag:
    return chips, all_headers
  else:
    return chips


#Same as above, except the uncorrected spectrum and telluric model are given separately,
#  instead of pre-divided
def ReadCombinedGridSearchFile(filename, headerflag=False):
  infile = open(filename)
  lines = infile.readlines()
  infile.close()
  wave = []
  rect = []
  opt = []
  recterror = []
  opterror = []
  cont = []
  telluric = []
  chips = []
  models = []
  all_headers = []
  header = []
  for line in lines:
    if not (line.startswith("#") or line == "\n"):
      try:
        wave.append(line.split()[0])
        rect.append(line.split()[1])
        opt.append(line.split()[2])
        recterror.append(line.split()[3])
        opterror.append(line.split()[4])
        cont.append(line.split()[5])
        telluric.append(line.split()[6])
      except IndexError:
        print "Format incorrect for file: ", filename, "\nExitting"
        sys.exit(0)
    elif line == "\n" and len(wave) > 0:
      chip = GridSearchOut(len(wave))
      Telluric = xypoint(len(telluric))
      all_headers.append(header)
      header = []
      chip.wave = numpy.array(wave).astype(float)
      chip.rect = numpy.array(rect).astype(float)
      chip.opt = numpy.array(opt).astype(float)
      chip.recterr = numpy.array(recterror).astype(float)
      chip.opterr = numpy.array(opterror).astype(float)
      chip.cont = numpy.array(cont).astype(float)
      Telluric.x = chip.wave.copy()
      Telluric.y = numpy.array(telluric).astype(float)
      wave = []
      rect = []
      opt = []
      recterror = []
      opterror = []
      cont = []
      telluric = []
      chips.append(chip)
      models.append(Telluric)
    elif line != "\n":
      header.append(line)
  if headerflag:
    return chips, models, all_headers
  else:
    return chips, models



def OutputGridSearchFile(chips, outfilename, modeflag="w"):
  outfile = open(outfilename, modeflag)
  for chip in chips:
    for j in range(chip.wave.size):
      outfile.write("%.15g\t" %chip.wave[j] + 
                    "%.15g\t" %chip.rect[j] +
                    "%.15g\t" %chip.opt[j] +
                    "%.15g\t" %chip.recterr[j] +
                    "%.15g\t" %chip.opterr[j] +
                    "%.15g\n" %chip.cont[j])
    outfile.write("\n\n\n\n")
  outfile.close()
