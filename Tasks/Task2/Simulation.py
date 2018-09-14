from Boat import Boat
from Passenger import Passenger
from Luggage import Luggage
from itertools import combinations


class Simulation:
    boat: Boat = Boat()
    p1: Passenger = Passenger(luggage=[Luggage(20), Luggage(80), Luggage(30), Luggage(100)])
    p2: Passenger = Passenger(luggage=[Luggage(50), Luggage(90), Luggage(30), Luggage(50)])
    times_sank: int = 0
    times_okay: int = 0

    def begin_travel_random(self) -> None:
        if self.p1.luggage:
            self.p1.pick_luggage()
            self.p1.pay_for_trip()

        print(f"Passenger one took {self.p1.on_person}")

        if self.p2.luggage:
            self.p2.pick_luggage()
            self.p2.pay_for_trip()

        print(f"Passenger two took {self.p2.on_person}")

        self.boat.board(self.p1, self.p2)

    def simulate_random(self) -> None:
        while self.p1.luggage or self.p2.luggage:
            self.begin_travel_random()

            if self.boat.is_sinking():
                self.check_sinking()
                continue

            print(f"Boat made it through with weight of {self.boat.current_weight}")
            self.boat.de_board()

            if self.p1.luggage:
                self.p1.pay_for_return()
            else:
                self.p1.wrong_side = False

            if self.p2.luggage:
                self.p2.pay_for_return()
            else:
                self.p2.wrong_side = False

            self.times_okay += 1

        if self.p1.wrong_side:
            self.p1.payed += 100

        if self.p2.wrong_side:
            self.p2.payed += 100

        print("Both passengers went over.")
        print(f"Passenger one payed {self.p1.payed}")
        print(f"Passenger two payed {self.p2.payed}")

    def check_sinking(self):
        print(f"Boat sank with weight of: {self.boat.current_weight}")
        self.p1.payed += 10000
        self.p2.payed += 10000
        self.boat.de_board()
        self.times_sank += 1

    def random_game(self) -> None:
        for _ in range(1000):
            self.simulate_random()
            print()

            self.boat = Boat()
            self.p1 = Passenger(luggage=[Luggage(20), Luggage(80), Luggage(30), Luggage(100)])
            self.p2 = Passenger(luggage=[Luggage(50), Luggage(90), Luggage(30), Luggage(50)])

        print(f"Times sank {self.times_sank}")
        print(f"Times okay {self.times_okay}")
        total_trips = self.times_okay + self.times_sank
        print(f"Failure rate: {round(self.times_sank / total_trips * 100,1)}")
        print(f"Success rate: {round(self.times_okay / total_trips * 100,1)}")


s = Simulation()
s.random_game()


from itertools import chain, combinations

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))

for subset in all_subsets([20, 30, 80, 100]):
    print(subset)
