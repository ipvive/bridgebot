use std::{convert, fmt};

use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer,
};

use crate::{Card, Deck, Rank, Suit};
struct DeckVisitor;

impl<'de> Visitor<'de> for DeckVisitor {
    type Value = Deck;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a set of cards")
    }
    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if v.len() % 2 != 0 {
            return Err(E::custom("not a set of cards"));
        }
        let cards = v
            .chunks_exact(2)
            .map(|pairs| {
                // there are two possible errors here, one is that we can't convert the single
                // character into a valid utf8 string, and the second is we can't find the
                // rank or suit from the given character. We return the same error for both.
                #[allow(clippy::indexing_slicing)]
                let rank = std::str::from_utf8(&[pairs[0]])
                    .map_err(|_| E::custom("invalid rank char"))
                    .map(|c| Rank::try_from(c).map_err(|_| E::custom("invalid rank char")))
                    .and_then(convert::identity);

                #[allow(clippy::indexing_slicing)]
                let suit = std::str::from_utf8(&[pairs[1]])
                    .map_err(|_| E::custom("invalid suit char"))
                    .map(|c| Suit::try_from(c).map_err(|_| E::custom("invalid suit char")))
                    .and_then(convert::identity);
                (rank, suit)
            })
            .map(|(rank, suit)| flatten((rank, suit)).map(|(rank, suit)| Card::from((rank, suit))))
            .collect::<Result<Vec<_>, E>>()?;
        Ok(Deck { cards })
    }
}

impl<'de> Deserialize<'de> for Deck {
    fn deserialize<D>(deserializer: D) -> Result<Deck, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_bytes(DeckVisitor)
    }
}

fn flatten<T, U, E>(input: (Result<T, E>, Result<U, E>)) -> Result<(T, U), E> {
    match input.0 {
        Ok(left) => match input.1 {
            Ok(right) => Ok((left, right)),
            Err(e) => Err(e),
        },
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod test {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::*;
    use serde_test::{assert_de_tokens, Token};
    use test_case::test_case;

    #[test_case(b"QH", Deck::from([
            Card::from((Rank::Queen, Suit::Heart))
        ]); "queen of hearts")
    ]
    #[test_case(b"QHKD", Deck::from([
            Card::from((Rank::Queen, Suit::Heart)),
            Card::from((Rank::King, Suit::Diamond))
        ]); "deck of 2s")
    ]

    fn test_deck(input: &'static [u8], expect: Deck) {
        assert_de_tokens(&expect, &[Token::Bytes(input)]);
    }

    #[test_case(b"Q", "not a set of cards"; "odd char count")]
    #[test_case(b"XH", "invalid rank char"; "invalid rank char")]
    #[test_case(b"QX", "invalid suit char"; "invalid suit char")]
    fn test_failures(input: &'static [u8], error: &'static str) {
        let msg = catch_unwind(AssertUnwindSafe(|| {
            assert_de_tokens(&Deck::default(), &[Token::Bytes(input)]);
        }));
        let text = msg.unwrap_err().downcast::<String>().unwrap();
        assert_eq!(*text, "tokens failed to deserialize: ".to_string() + error);
    }
}
