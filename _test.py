from pathlib import Path
from yippy import Coronagraph

yip = Path("coronagraphs/LUVOIR-B-VC6_timeseries/")

coro = Coronagraph(yip)
breakpoint()
