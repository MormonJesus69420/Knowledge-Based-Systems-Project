from Sellers import Seller, ZISeller
from dataclasses import dataclass, field
from typing import List


@dataclass
class Bazaar:
    sale_demand: int = 100
    zi_seller: ZISeller = ZISeller(goal=sale_demand)

    def simulate_fixed_day(self, seller: Seller) -> None:
        print(ZISeller)
        seller.pick_price()
        sold = self.sale_demand - seller.current_price * 5
        if sold > seller.shoes_available:
            sold = seller.shoes_available

        seller.units_sold.append(sold)
        seller.earning += sold * seller.current_price


b = Bazaar()
for _ in range(100):
    b.simulate_fixed_day(b.zi_seller)
    print(b.zi_seller.price)

print(b.zi_seller.units_sold)
