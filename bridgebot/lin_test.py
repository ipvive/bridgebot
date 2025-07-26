import io
import tensorflow as tf

import bridge.game as bridgegame
import bridge.lin as lin


class Reader(io.StringIO):
    def __init__(self, buffer=None):
        super().__init__(buffer)
        self.name = "test"
    

class LinTest(tf.test.TestCase):
    def test_basic(self):
        parser = lin.Parser()
        game = bridgegame.Game()
        r = Reader("rs|3NE-1,3NE-1|pn|x,y,z,w,A,B,C,D|qx|o1|md|1SATC2,SK,SQ,SJ|qx|c1|md|1SAC2,SK,SQ,SJ|mb|p|")
        boards, _ = parser.parse(r, game)
        self.assertEqual(len(boards), 1)
        self.assertEqual(boards['1'].tables['o'].players, {"South": "x", "West": "y", "North": "z", "East": "w"})
        self.assertEqual(boards['1'].tables['c'].players, {"South": "a", "West": "b", "North": "c", "East": "d"})
        deal = boards['1'].tables['c']
        self.assertEqual(deal.vulnerability, [])
        # TODO(njt): check cards.


    def test_variant_input(self):
        parser = lin.Parser()
        game = bridgegame.Game()
        inputs = {
                "dash equals pass": "rs|3NE-1|pn|,,,,,,,|qx|o1|mb|-|pg||",
                "mixed-case calls": "rs|3NE-1|pn|,,,,,,,|qx|o1|mb|p|mb|P|mb|1c|mb|d|mb|r|mb|1d|mb|D|mb|R|pg||",
                "two deals": "rs|3NE-1,3NE-1|pn|,,,,,,,|qx|o1|mb|p|pg||qx|c1|mb|p|pg||",
                "per-deal seating": "rs|3NE-1,3NE-1|pn|x,y,z,w,A,B,C,D|mb|p|pg||pn|A,B,C,D|pg||qx|c1|mb|p|pg||",  # 000547.lin
                "10": "rs|3NE-1|pn|,,,,,,,|qx|o1|mb|1c|mb|p|mb|p|mb|p|pc|C10|",
                "three hands specified":"rs|3NE-1|pn|,,,,,,,|qx|o1|md|2SAT8H762DQT764CK3,SKQ74HAQT3DJCJ954,S953HJ84D82CAQT76|mb|p|",
                "split board number": "rs|3NE-1|pn|,,,,,,,|qx|o1,board 1|mb|p|pg||",
                "trailing empty commentary": "rs|3NE-1|pn|,,,,,,,|qx|o1|mb|p|nt||",
                "alerted call": "rs|3NE-1|pn|,,,,,,,|qx|o1|mb|1c|mb|d!|mb|r!|mb|p!|",
                }
        for _, input in inputs.items():
            r = Reader(input)
            result, _ = parser.parse(r, game)
            self.assertNotEqual(result, {})

    def test_malformed_input(self):
        parser = lin.Parser()
        game = bridgegame.Game()
        inputs = {
                "Missing player names": "qx|o1|pg||",
                "too few hands": "pn|,,,,,,,|qx|o1|md|1,|",
                "too few players": "pn|,,,,,|qx|c1|",
                "no suit":"pn|,,,,,,,|qx|o1|md|2A,,,|",
                "missing card rank": "pn|,,,,,,,|qx|o1|mb|1c|mb|p|mb|p|mb|p|pc|C|",
                }
        for _, input in inputs.items():
            r = Reader(input)
            result, _ = parser.parse(r, game)
            self.assertEqual(result, {})

if __name__ == "__main__":
    tf.test.main()
