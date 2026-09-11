import inspect
import os

from pyBADA.configuration import getAircraftList
import pyBADA

print(pyBADA.__file__)
aircraft = getAircraftList(
    badaFamily="BADA3",
    badaVersion="3.16"
)

print(len(aircraft))
print(aircraft[:20])

print(inspect.getsource(getAircraftList))


root = os.path.dirname(pyBADA.__file__)

for f in os.listdir(root):
    print(f)