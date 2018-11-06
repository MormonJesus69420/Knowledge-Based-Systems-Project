from random import randint, choice
from dataclasses import dataclass, field

from datetime import timedelta
from matplotlib import pyplot as plt
from numpy import ceil, random
from typing import List

from Actors import Passenger, Taxi
from Location import Location


@dataclass
class Simulation:
    prices: List[int] = field(default_factory=list, repr=False)
    taxis: List[Taxi] = field(default_factory=list, repr=False)
    occupied_taxis: List[Taxi] = field(default_factory=list, repr=False)
    passengers: List[Passenger] = field(default_factory=list, repr=False)
    locations: List[Location] = field(default_factory=list, repr=False)

    def simulate(self, intelligent: bool = True, days: int = 100) -> None:
        now = timedelta(hours=20, minutes=30)
        time_delta = timedelta(minutes=1)
        # taken_taxis = []
        averages = []
        i = 0

        self.setup_locations()
        self.setup_taxis(50)

        while i < days:
            if now == timedelta(days=0, hours=23, minutes=59):
                now = timedelta(days=0, hours=20, minutes=30)
                averages.append(self.average_price())

                self.taxis.extend(self.occupied_taxis)
                self.occupied_taxis.clear()

                i += 1
                continue

            self.move_occupied_taxis(time_delta)

            self.close_locations(now)

            while self.passengers and self.taxis:
                if intelligent:
                    self.smart_fucks_auction()
                else:
                    self.dumb_fucks_auction()

            # taken_taxis.append(len(self.occupied_taxis))

            now += time_delta

        # plt.figure()
        # plt.title(f"Taken taxis through day {'Roth Erev' if intelligent else 'Zero intelligence'}")
        # plt.plot(taken_taxis)
        # plt.xlabel("Minutes",)
        # plt.ylabel("Taken taxis")
        # plt.show()
        # plt.close()

        plt.figure()
        plt.title(f"Average of prices after each iteration {'Roth Erev' if intelligent else 'Zero intelligence'}")
        plt.plot(averages)
        plt.xlabel("Iterations")
        plt.ylabel("Price")
        plt.show()
        plt.close()

    def close_locations(self, now:timedelta) -> None:
        for location in self.locations:
            if now == location.closing_hour:
                location.close()
                self.setup_passengers(location.clients)

    def move_occupied_taxis(self, time_delta: timedelta) ->None:
        for taxi in self.occupied_taxis:
            taxi.busy(time_delta)
            if taxi.travel_time == timedelta(days=0, hours=0, minutes=0):
                self.taxis.append(taxi)

        self.occupied_taxis = [t for t in self.occupied_taxis if t not in self.taxis]

    def setup_taxis(self, taxi_count: int) -> None:
        for i in range(taxi_count):
            self.taxis.append(Taxi())

    def setup_locations(self) -> None:
        cinema1 = Location("Cinema", 150, timedelta(hours=20, minutes=30))
        opera = Location("Opera", 250, timedelta(hours=21, minutes=15))
        restaurant = Location("Restaurant", 50, timedelta(hours=21, minutes=45))
        cinema2 = Location("Cinema", 150, timedelta(hours=22, minutes=15))
        self.locations = list([cinema1, opera, restaurant, cinema2])

    def setup_passengers(self, passenger_count: int) -> None:
        i = 0
        while i <= passenger_count:
            size = ceil(0 + (random.beta(1, 1) * (4 - 0)))
            time = timedelta(minutes=randint(10, 40) * 2)
            self.passengers.append(Passenger(travel_time=time, group_size=size))
            i += size

    def dumb_fucks_auction(self) -> None:
        passenger = choice(self.passengers)
        passenger.zero_intelligence_bidding()
        taxi = choice(self.taxis)
        taxi.zero_intelligence_bidding()

        if len(self.passengers) > len(self.taxis):
            if passenger.bid >= taxi.bid:
                taxi.travel_time = passenger.travel_time
                self.pair_passenger_to_taxi(passenger, taxi, passenger.bid)
        else:
            if taxi.bid <= passenger.bid:
                taxi.travel_time = passenger.travel_time
                self.pair_passenger_to_taxi(passenger, taxi, taxi.bid)

    def smart_fucks_auction(self) -> None:
        passenger = choice(self.passengers)
        passenger.roth_erev_bidding()
        taxi = choice(self.taxis)
        taxi.roth_erev_bidding()

        if len(self.passengers) > len(self.taxis):
            if passenger.bid >= taxi.bid:
                passenger.accept()
                taxi.accept()

                taxi.travel_time = passenger.travel_time
                self.pair_passenger_to_taxi(passenger, taxi, passenger.bid)
            else:
                passenger.reject()
                taxi.reject()
        else:
            if taxi.bid <= passenger.bid:
                passenger.accept()
                taxi.accept()

                taxi.travel_time = passenger.travel_time
                self.pair_passenger_to_taxi(passenger, taxi, taxi.bid)
            else:
                passenger.reject()
                taxi.reject()

    def pair_passenger_to_taxi(self, passenger: Passenger, taxi: Taxi, bid: int) -> None:
        self.prices.append(bid)
        self.passengers.remove(passenger)

        self.taxis.remove(taxi)
        self.occupied_taxis.append(taxi)

    def average_price(self) -> float:
        average = sum(self.prices) / len(self.prices)
        self.prices.clear()

        return average


if __name__ == "__main__":
    s = Simulation()
    s.simulate(True, 1000)
    s.simulate(False, 1000)
