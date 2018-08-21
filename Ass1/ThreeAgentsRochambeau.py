from dataclasses import dataclass, field
from random import choice as choose
from Player import Player
from Move import Move
from typing import List
from matplotlib import pyplot
from numpy import arange


def make_moves() -> List[Move]:
    return [Move.ROCK, Move.PAPER, Move.SCISSORS]


@dataclass
class ThreeAgentsRochambeau:
    player_one: Player = Player()
    player_two: Player = Player()
    player_three: Player = Player()
    moves: List[Move] = field(default_factory=make_moves, init=False, repr=False)

    def generate_moves(self) -> None:
        self.player_one.move = choose(self.moves)
        self.player_two.move = choose(self.moves)

    def check_win(self) -> int:
        p1 = self.player_one.move
        p2 = self.player_two.move

        if p1 == p2 == Move.ROCK:
            return 3
        if p1 == Move.PAPER and p2 == Move.SCISSORS:
            return 2
        if p1 == Move.SCISSORS and p2 == Move.PAPER:
            return 1

        return 0

    def run(self) -> None:
        runs = 0

        while runs != 100:
            self.generate_moves()
            result = self.check_win()

            if result == 0:
                continue
            elif result == 1:
                self.player_one.wins += 1
            elif result == 2:
                self.player_two.wins += 1
            else:
                self.player_three.wins += 1

            runs += 1

    def show_results(self) -> None:
        wins = [self.player_one.wins, self.player_two.wins, self.player_three.wins]
        players = ["Player One " + str(wins[0]), "Player Two " + str(wins[1]), "Player Three " + str(wins[2])]
        position = arange(len(players))

        pyplot.bar(position, wins, align="center", alpha=0.25)
        pyplot.xticks(position, players)
        pyplot.ylabel("Wins")
        pyplot.title("Win distribution")

        pyplot.show()


temp = ThreeAgentsRochambeau()
temp.run()
temp.show_results()
