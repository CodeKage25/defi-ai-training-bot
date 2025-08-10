from .sources.coingecko import CoinGecko
from .sources.zerox import ZeroX

class DataRegistry:
    def __init__(self):
        self.coingecko = CoinGecko()
        self.zx = ZeroX()

registry = DataRegistry()
