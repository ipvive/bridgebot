use std::{cmp::Ordering, fmt::Display, ops::Deref};

use enum_derived::Rand;
use enum_iterator::{all, next_cycle, Sequence};
use rand::{seq::SliceRandom, Rng};
use serde::Deserialize;
use std::fmt::Debug;
use std::str::FromStr;
use strum::{Display, EnumString, VariantArray};

mod lin;

#[derive(
    Copy, Clone, Debug, Display, EnumString, Rand, PartialEq, Eq, PartialOrd, Ord, Hash, Sequence,
)]
#[pyclass(frozen)]
#[repr(u8)]
pub enum Seat {
    #[strum(serialize = "S", serialize = "South")]
    South,
    #[strum(serialize = "W", serialize = "West")]
    West,
    #[strum(serialize = "N", serialize = "North")]
    North,
    #[strum(serialize = "E", serialize = "East")]
    East,
}

#[pymethods]
impl Seat {
    #[new]
    fn new(str: &str) -> Result<Seat, GameError> {
        Seat::try_from(str).map_err(|_| GameError::InvalidSeat)
    }
    fn next_seat(&self) -> Seat {
        next_cycle(self).expect("always at least one seat")
    }
    fn partner(&self) -> Seat {
        next_cycle(self)
            .expect("always at least one seat")
            .next_seat()
    }
    fn as_string(&self) -> String {
        self.to_string()
    }
}

#[derive(Clone, Debug, PartialEq)]
#[pyclass]
struct Player {
    _name: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, EnumString, VariantArray, Display)]
#[pyclass]
enum Level {
    #[strum(serialize = "1")]
    L1,
    #[strum(serialize = "2")]
    L2,
    #[strum(serialize = "3")]
    L3,
    #[strum(serialize = "4")]
    L4,
    #[strum(serialize = "5")]
    L5,
    #[strum(serialize = "6")]
    L6,
    #[strum(serialize = "7")]
    L7,
}

#[pymethods]
impl Level {
    #[new]
    fn new(input: &str) -> Result<Self, GameError> {
        Level::from_str(input).map_err(|_| GameError::InvalidRankString(input.to_owned()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Strain {
    Suit(Suit),
    NoTrump,
}

impl Display for Strain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strain::Suit(suit) => std::fmt::Display::fmt(suit, f),
            Strain::NoTrump => write!(f, "NoTrump"),
        }
    }
}

impl PartialOrd for Strain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::NoTrump, Self::NoTrump) => Some(Ordering::Equal),
            (Self::NoTrump, Self::Suit(_)) => Some(Ordering::Greater),
            (Self::Suit(_), Self::NoTrump) => Some(Ordering::Less),
            (Self::Suit(s1), Self::Suit(s2)) => s1.partial_cmp(s2),
        }
    }
}

impl TryFrom<&str> for Strain {
    type Error = GameError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "NT" | "notrump" => Strain::NoTrump,
            _ => Strain::Suit(Suit::new(value)?),
        })
    }
}

impl Strain {
    const fn id(&self) -> usize {
        match self {
            Self::NoTrump => Suit::CARDINALITY,
            Self::Suit(s) => *s as usize,
        }
    }
}

#[derive(Clone, Copy, Debug, EnumString, PartialEq, Rand)]
#[pyclass(frozen)]
enum Vulnerability {
    None,
    NorthSouth,
    EastWest,
    All,
}

#[pymethods]
impl Vulnerability {
    #[new]
    fn new(name: Option<&str>) -> Result<Self, GameError> {
        match name {
            Some(name) => Self::try_from(name).map_err(|_| GameError::InvalidVulnerability),
            None => Ok(Self::None),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CardLocation {
    Owned(Seat),
    Played(Seat),
}

// Whether or not a player has shown out (exhausted) a specific suit
type NoCardOfSuit = [bool; Suit::CARDINALITY];

#[derive(Clone, Copy, Debug, EnumString, PartialEq, PartialOrd)]
#[pyclass]
enum Call {
    #[strum(serialize = "pass", serialize = "Pass")]
    Pass = 35,

    #[strum(serialize = "double", serialize = "Double")]
    Double = 36,

    #[strum(serialize = "redouble", serialize = "reDouble")]
    ReDouble = 37,
}

#[pymethods]
impl Call {
    #[new]
    fn new(value: &str) -> Result<Self, GameError> {
        Call::try_from(value).map_err(|_| GameError::InvalidCall)
    }
}

#[derive(Clone, Copy, Debug, PartialOrd)]
#[pyclass]
struct Bid {
    level: Level,
    strain: Strain,
}

impl PartialEq for Bid {
    fn eq(&self, other: &Self) -> bool {
        self.level == other.level && self.strain == other.strain
    }
}

impl From<u8> for Bid {
    fn from(value: u8) -> Self {
        let level: Level = *Level::VARIANTS.get((value / 5) as usize).unwrap();
        let strain: Strain = Suit::VARIANTS
            .get((value % 5) as usize)
            .map(|suit| Strain::Suit(*suit))
            .unwrap_or(Strain::NoTrump);
        Bid { level, strain }
    }
}

impl From<Bid> for u8 {
    fn from(value: Bid) -> Self {
        value.level as u8 * 5 + value.strain.id() as u8
    }
}

#[pymethods]
impl Bid {
    #[new]
    fn new(level: Level, strain: &str) -> Result<Self, GameError> {
        Ok(Bid {
            level,
            strain: strain.try_into()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FirstMentioned {
    inner: [[bool; Suit::CARDINALITY + 1]; PLAYERS],
}

impl FirstMentioned {
    const fn default() -> Self {
        Self {
            inner: [[false; Suit::CARDINALITY + 1]; PLAYERS],
        }
    }
    const fn get(&self, seat: Seat, strain: Strain) -> bool {
        self.inner[seat as usize][strain.id()]
    }

    fn get_mut(&mut self, seat: Seat, strain: Strain) -> &mut bool {
        &mut self.inner[seat as usize][strain.id()]
    }

    fn set(&mut self, seat: Seat, strain: Strain) {
        *self.get_mut(seat, strain) = true;
    }
}

#[derive(Clone, Debug, PartialEq)]
enum DealState {
    Bidding {
        next_to_bid: Seat,
        last_bid: Option<(Bid, Seat, Call)>,
        first_mentioned: FirstMentioned,
        pass_count: u8,
    },
    Playing {
        trump: Strain,
        next_to_play: Seat,
        tricks_played: u8,
        contract: (Bid, Seat, Call),
    },
    Scoring,
}

const PLAYERS: usize = 4;

#[derive(Debug, Clone, PartialEq)]
#[pyclass]
struct Deal {
    pub _players: [Player; PLAYERS],
    _vulnerability: Vulnerability,
    dealt_cards: [Option<CardLocation>; Deck::CARDS_IN_DECK],
    no_cards: [NoCardOfSuit; PLAYERS],
    deal_state: DealState,
    trick: Option<Trick>,
    tricks_won: [u8; PLAYERS],
    events: Vec<HistoryEvent>,
    card_count: [u8; PLAYERS],
}

// who has a specific Card
// do a player have any of a specific Suit

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GameError {
    #[error("Card {card} cannot be assigned to seat {assign_to} because it was already held by seat {held_by}")]
    AlreadyHeld {
        card: Card,
        assign_to: Seat,
        held_by: Seat,
    },
    #[error("Card {card} cannot be assigned to seat {assign_to} because it was played by seat {played_by}")]
    AlreadyPlayed {
        card: Card,
        assign_to: Seat,
        played_by: Seat,
    },
    #[error(
        "Card {card} cannot be given to seat {give_to} because they have previously revoked suit {revoked}"
    )]
    Revoked {
        card: Card,
        give_to: Seat,
        revoked: Suit,
    },

    #[error("Card {card} cannot be given to seat {seat} because they already have 13 cards")]
    FourteenCards { seat: Seat, card: Card },

    #[error(
        "Card {card} cannot be played by {played_by} because they have another card in the trick suit"
    )]
    ActionRevoked { card: Card, played_by: Seat },

    #[error("Card {card} has already been played")]
    ActionAlreadyPlayed { card: Card },

    #[error("This operation cannot be performed on this deal in this state")]
    InvalidState,

    #[error("execute_action_ids failed at position {position} with error {error}")]
    BatchError { position: usize, error: String },

    #[error("Invalid double, no bid yet")]
    DoubleOfNoBid,

    #[error("Invalid redouble, no bid yet")]
    ReDoubleOfNoBid,

    #[error("Invalid double, already doubled")]
    AlreadyDoubled,

    #[error("Double of own side's contract")]
    DoubleOfOwnSide,

    #[error("Redouble of other side contract")]
    ReDoubleOfOtherSide,

    #[error("Insufficient bid")]
    InsufficientBid,

    #[error("invalid suit string '{0}'")]
    InvalidSuitString(String),
    #[error("invalid rank string '{0}'")]
    InvalidRankString(String),

    #[error("invalid seat")]
    InvalidSeat,

    #[error("invalid vulnerability")]
    InvalidVulnerability,

    #[error("invalid call")]
    InvalidCall,

    #[error("no such action {0}")]
    NoSuchAction(usize),
}

impl From<GameError> for PyErr {
    fn from(value: GameError) -> Self {
        PyErr::new::<PyTypeError, _>(value.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Trick {
    cards_played: usize,
    suit: Suit,
    winning_card: Card,
    winning_seat: Seat,
}

impl Trick {
    fn is_strongest_card(&self, card: Card, trump: Strain) -> bool {
        // you play trump, no trump played so far -- > win

        if let Strain::Suit(t) = trump {
            if t == card.suit && self.winning_card.suit != t {
                return true;
            }
        }

        // not same suit as previous winning card -> lose
        if card.suit != self.winning_card.suit {
            return false;
        }

        // otherwise check rank
        card.rank > self.winning_card.rank
    }

    fn replace_best_card(&mut self, card: Card, seat: Seat) {
        self.winning_card = card;
        self.winning_seat = seat;
    }
}

#[pymethods]
impl Deal {
    const INITIAL_DEAL: usize = 13;

    #[new]
    const fn new(dealer: Seat, _vulnerability: Vulnerability) -> Self {
        Self {
            _players: [
                Player { _name: "Rodwell" },
                Player { _name: "p2" },
                Player { _name: "p3" },
                Player { _name: "p4" },
            ],
            _vulnerability,
            dealt_cards: [None; Deck::CARDS_IN_DECK],
            no_cards: [[false; 4]; 4],
            deal_state: DealState::Bidding {
                next_to_bid: dealer,
                last_bid: None,
                first_mentioned: FirstMentioned::default(),
                pass_count: 0,
            },
            trick: None,
            tricks_won: [0; PLAYERS],
            events: Vec::new(),
            card_count: [0; PLAYERS],
        }
    }

    fn deepcopy(&self) -> Self {
        self.clone()
    }

    #[staticmethod]
    fn random_deal() -> Self {
        let mut deal = Self::new(Seat::rand(), Vulnerability::rand());
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let deck = Deck::shuffled(&mut rng);
        let mut chunks = deck.cards.chunks(Self::INITIAL_DEAL);
        //    trump: None,
        enum_iterator::all::<Seat>().for_each(|seat| {
            chunks.next().unwrap().iter().for_each(|&card| {
                deal.give_card(seat, card).unwrap();
            });
        });
        debug_assert!(chunks.next().is_none());
        deal
    }

    fn show(&self) -> String {
        format!("{:?}", self)
    }

    // can only give a card that has not been assigned to anyone
    fn give_card(&mut self, seat: Seat, card: Card) -> Result<(), GameError> {
        if 13 == self.card_count[seat as usize] {
            return Err(GameError::FourteenCards { seat, card });
        }

        let cardref = self
            .dealt_cards
            .get_mut(card.id())
            .expect("card id too large -- impossible");

        let no_card = self.no_cards[seat as usize][card.suit as usize];
        if no_card {
            return Err(GameError::Revoked {
                card,
                give_to: seat,
                revoked: card.suit,
            });
        }

        match *cardref {
            Some(CardLocation::Played(who)) => Err(GameError::AlreadyPlayed {
                card,
                assign_to: seat,
                played_by: who,
            }),
            Some(CardLocation::Owned(who)) => Err(GameError::AlreadyHeld {
                card,
                assign_to: seat,
                held_by: who,
            }),
            None => {
                *cardref = Some(CardLocation::Owned(seat));
                self.card_count[seat as usize] += 1;
                Ok(())
            }
        }
    }

    pub fn execute_play_action(&mut self, card: Card) -> Result<(), GameError> {
        let (currently_playing, tricks_played, trump, contract) = if let DealState::Playing {
            next_to_play,
            tricks_played,
            trump,
            contract,
        } = self.deal_state
        {
            (next_to_play, tricks_played, trump, contract)
        } else {
            return Err(GameError::InvalidState);
        };
        let card_state = self.dealt_cards[card.id()];
        match card_state {
            Some(CardLocation::Played(_who)) => {
                return Err(GameError::ActionAlreadyPlayed { card })
            }
            Some(CardLocation::Owned(who)) if who == currently_playing => {}
            _ => self.give_card(currently_playing, card)?,
        };
        // at this point, we are sure that self.next_to_act owns the card in execute_play_action
        if let Some(ref trick) = self.trick {
            if trick.suit != card.suit {
                if self.card_suit_iterator(trick.suit).any(|&cl| {
                    if let Some(CardLocation::Owned(seat)) = cl {
                        currently_playing == seat
                    } else {
                        false
                    }
                }) {
                    return Err(GameError::ActionRevoked {
                        card,
                        played_by: currently_playing,
                    });
                }
                self.no_cards[currently_playing as usize][trick.suit as usize] = true;
            }
        }

        // error checks are complete, do the work of playing the card
        self.dealt_cards[card.id()] = Some(CardLocation::Played(currently_playing));
        self.deal_state = if let Some(ref mut trick) = self.trick {
            if trick.is_strongest_card(card, trump) {
                trick.replace_best_card(card, currently_playing);
            }
            if trick.cards_played == 3 {
                self.tricks_won[trick.winning_seat as usize] += 1;
                trick.cards_played = 0;
                DealState::Playing {
                    next_to_play: trick.winning_seat,
                    tricks_played: tricks_played + 1,
                    trump,
                    contract,
                }
            } else {
                trick.cards_played += 1;
                DealState::Playing {
                    next_to_play: currently_playing.next_seat(),
                    tricks_played,
                    trump,
                    contract,
                }
            }
        } else {
            self.trick = Some(Trick {
                cards_played: 1,
                suit: card.suit,
                winning_card: card,
                winning_seat: currently_playing,
            });
            DealState::Playing {
                next_to_play: currently_playing.next_seat(),
                tricks_played,
                trump,
                contract,
            }
        };
        if matches!(
            self.deal_state,
            DealState::Playing {
                next_to_play: _,
                tricks_played: 13,
                trump: _,
                contract: _,
            }
        ) {
            self.deal_state = DealState::Scoring;
        }

        self.events.push(HistoryEvent {
            actor: currently_playing,
            action: card.id() as u8 + 38,
        });

        Ok(())
    }

    pub fn execute_bid_action(&mut self, bid: Bid) -> Result<(), GameError> {
        let (bidder, last_bid, first_mentioned) = if let DealState::Bidding {
            next_to_bid,
            last_bid,
            ref mut first_mentioned,
            pass_count: _,
        } = &mut self.deal_state
        {
            (*next_to_bid, last_bid, first_mentioned)
        } else {
            return Err(GameError::InvalidState);
        };
        // if there was a last_bid, if this bid is less than or equal to it, fail
        if last_bid.map(|last_bid| bid <= last_bid.0).unwrap_or(false) {
            return Err(GameError::InsufficientBid);
        }

        if !first_mentioned.get(bidder.partner(), bid.strain) {
            first_mentioned.set(bidder, bid.strain);
        }

        self.deal_state = DealState::Bidding {
            next_to_bid: bidder.next_seat(),
            last_bid: Some((bid, bidder, Call::Pass)),
            first_mentioned: first_mentioned.clone(), // TODO: possible optimization
            pass_count: 0,
        };

        self.events.push(HistoryEvent {
            actor: bidder,
            action: bid.into(),
        });

        Ok(())
    }
    fn execute_call_action(&mut self, call: Call) -> Result<(), GameError> {
        let (caller, last_bid, first_mentioned, pass_count) = if let DealState::Bidding {
            next_to_bid,
            last_bid,
            ref mut first_mentioned,
            pass_count,
        } = self.deal_state
        {
            (next_to_bid, last_bid, first_mentioned, pass_count)
        } else {
            return Err(GameError::InvalidState);
        };
        match call {
            Call::Pass => match (pass_count, last_bid) {
                (3, _) => self.deal_state = DealState::Scoring,
                (2, Some(bid)) => {
                    let declarer = if first_mentioned.get(bid.1, bid.0.strain) {
                        bid.1
                    } else {
                        bid.1.partner()
                    };
                    self.deal_state = DealState::Playing {
                        next_to_play: declarer.next_seat(),
                        tricks_played: 0,
                        trump: bid.0.strain,
                        contract: (bid.0, declarer, bid.2),
                    }
                }
                (0..=2, _) => {
                    self.deal_state = DealState::Bidding {
                        next_to_bid: caller.next_seat(),
                        last_bid,
                        first_mentioned: first_mentioned.clone(),
                        pass_count: pass_count + 1,
                    }
                }
                _ => unreachable!(),
            },
            Call::Double => {
                if let Some(bid) = last_bid {
                    if let Call::Pass = bid.2 {
                        if bid.1 == caller || bid.1 == caller.partner() {
                            return Err(GameError::DoubleOfOwnSide);
                        }
                        self.deal_state = DealState::Bidding {
                            next_to_bid: caller.next_seat(),
                            last_bid: Some((bid.0, bid.1, Call::Double)),
                            first_mentioned: first_mentioned.clone(),
                            pass_count: 0,
                        };
                    } else {
                        return Err(GameError::AlreadyDoubled);
                    }
                } else {
                    return Err(GameError::DoubleOfNoBid);
                }
            }
            Call::ReDouble => {
                if let Some(bid) = last_bid {
                    if let Call::Double = bid.2 {
                        if bid.1 != caller && bid.1 != caller.partner() {
                            return Err(GameError::ReDoubleOfOtherSide);
                        }
                        self.deal_state = DealState::Bidding {
                            next_to_bid: caller.next_seat(),
                            last_bid: Some((bid.0, bid.1, Call::ReDouble)),
                            first_mentioned: first_mentioned.clone(),
                            pass_count: 0,
                        };
                    } else {
                        return Err(GameError::AlreadyDoubled);
                    }
                } else {
                    return Err(GameError::ReDoubleOfNoBid);
                }
            }
        }
        self.events.push(HistoryEvent {
            actor: caller,
            action: call as u8,
        });

        Ok(())
    }

    fn execute_actions_ids(&mut self, actions: &[u8]) -> Result<(), GameError> {
        actions
            .iter()
            .enumerate()
            .try_for_each(|(position, action)| {
                let actor = self.next_to_act().ok_or(GameError::BatchError {
                    position,
                    error: "invalid game state".to_string(),
                })?;
                match action {
                    0..=34 => {
                        let bid = (*action).into();
                        self.execute_bid_action(bid)
                    }
                    35 => self.execute_call_action(Call::Pass),
                    36 => self.execute_call_action(Call::Double),
                    37 => self.execute_call_action(Call::ReDouble),
                    38..=90 => self.execute_play_action((action - 38).into()),
                    _ => {
                        return Err(GameError::BatchError {
                            position,
                            error: format!("invalid action id {action}"),
                        })
                    }
                }
                .map_err(|e| GameError::BatchError {
                    position,
                    error: e.to_string(),
                })?;
                Ok(())
            })
    }

    fn num_actions(&self) -> usize {
        self.events.len()
    }

    const fn next_to_act(&self) -> Option<Seat> {
        match self.deal_state {
            DealState::Bidding { next_to_bid, .. } => Some(next_to_bid),
            DealState::Playing { next_to_play, .. } => Some(next_to_play),
            DealState::Scoring => None,
        }
    }

    const fn pass_position(&self) -> Option<u8> {
        if let DealState::Bidding {
            pass_count: pass_position,
            ..
        } = self.deal_state
        {
            Some(pass_position)
        } else {
            None
        }
    }

    fn contract_strain(&self) -> Option<String> {
        if let DealState::Playing {
            contract: (Bid { strain, .. }, _, _),
            ..
        } = self.deal_state
        {
            return Some(format!("{strain}"));
        }
        None
    }

    fn contract_level(&self) -> Option<String> {
        if let DealState::Playing {
            contract: (Bid { level, .. }, _, _),
            ..
        } = self.deal_state
        {
            return Some(format!("{level}"));
        }
        None
    }

    fn contract_seat(&self) -> Option<String> {
        if let DealState::Playing {
            contract: (_, seat, _),
            ..
        } = self.deal_state
        {
            return Some(format!("{seat}"));
        }
        None
    }

    const fn tricks_won(&self) -> [u8; PLAYERS] {
        self.tricks_won
    }

    fn action(&self, position: usize) -> Result<u8, GameError> {
        self.events
            .get(position)
            .map(|event| event.action)
            .ok_or(GameError::NoSuchAction(position))
    }

    fn equals(&self, other: &Self) -> bool {
        self == other
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryEvent {
    actor: Seat,
    action: u8,
}

impl Deal {
    fn card_suit_iterator<'a>(
        &'a self,
        suit: Suit,
    ) -> Box<dyn Iterator<Item = &Option<CardLocation>> + 'a> {
        Box::new(
            self.dealt_cards
                .as_slice()
                .iter()
                .skip(suit as usize * 13)
                .take(13),
        )
    }
}

#[derive(Debug, PartialEq)]
struct Deck {
    cards: Vec<Card>,
}

impl Default for Deck {
    // return a deck with all cards
    fn default() -> Self {
        let cards: Vec<Card> = all::<Rank>()
            .flat_map(|rank| all::<Suit>().map(move |suit| Card { suit, rank }))
            .collect();
        Self { cards }
    }
}

impl Deref for Deck {
    type Target = Vec<Card>;

    fn deref(&self) -> &Self::Target {
        &self.cards
    }
}

impl Deck {
    const CARDS_IN_DECK: usize = 52;
    // return a shuffled deck with all cards
    fn shuffled<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut deck = Deck::default();
        deck.cards.shuffle(rng);
        deck
    }
}

impl<T: IntoIterator<Item = Card>> From<T> for Deck {
    fn from(value: T) -> Self {
        Self {
            cards: value.into_iter().collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[pyclass]
pub struct Card {
    suit: Suit,
    rank: Rank,
}

impl Card {
    const fn id(&self) -> usize {
        (self.suit as usize) * 13 + self.rank as usize
    }
}

impl From<u8> for Card {
    fn from(value: u8) -> Self {
        Card {
            suit: (value / 13).into(),
            rank: (value % 13).into(),
        }
    }
}

#[pymethods]
impl Card {
    #[new]
    fn new(suit: &str, rank: &str) -> Result<Self, GameError> {
        Ok(Card {
            suit: Suit::new(suit)?,
            rank: Rank::new(rank)?,
        })
    }
}

impl From<(Rank, Suit)> for Card {
    fn from(value: (Rank, Suit)) -> Self {
        Self {
            rank: value.0,
            suit: value.1,
        }
    }
}

impl Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.rank, self.suit)
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Display,
    EnumString,
    PartialEq,
    PartialOrd,
    Eq,
    Sequence,
    VariantArray,
)]
#[pyclass]
pub enum Suit {
    #[strum(serialize = "C", serialize = "Clubs")]
    Club,
    #[strum(serialize = "D", serialize = "Diamonds")]
    Diamond,
    #[strum(serialize = "H", serialize = "Hearts")]
    Heart,
    #[strum(serialize = "S", serialize = "Spades")]
    Spade,
}

#[pymethods]
impl Suit {
    #[new]
    fn new(input: &str) -> Result<Self, GameError> {
        match input {
            "Club" | "Clubs" | "C" => Ok(Suit::Club),
            "Diamond" | "Diamonds" | "D" => Ok(Suit::Diamond),
            "Heart" | "Hearts" | "H" => Ok(Suit::Heart),
            "Spade" | "Spades" | "S" => Ok(Suit::Spade),
            _ => Err(GameError::InvalidSuitString(input.to_owned())),
        }
    }
}

impl From<u8> for Suit {
    fn from(value: u8) -> Self {
        *Suit::VARIANTS.get(value as usize).unwrap()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Display,
    EnumString,
    Sequence,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    VariantArray,
)]
#[pyclass]
enum Rank {
    #[strum(serialize = "2", serialize = "Two")]
    Two,
    #[strum(serialize = "3", serialize = "Three")]
    Three,
    #[strum(serialize = "4", serialize = "Four")]
    Four,
    #[strum(serialize = "5", serialize = "Five")]
    Five,
    #[strum(serialize = "6", serialize = "Six")]
    Six,
    #[strum(serialize = "7", serialize = "Seven")]
    Seven,
    #[strum(serialize = "8", serialize = "Eight")]
    Eight,
    #[strum(serialize = "9", serialize = "Nine")]
    Nine,
    #[strum(serialize = "T", serialize = "Ten")]
    Ten,
    #[strum(serialize = "J", serialize = "Jack")]
    Jack,
    #[strum(serialize = "Q", serialize = "Queen")]
    Queen,
    #[strum(serialize = "K", serialize = "King")]
    King,
    #[strum(serialize = "A", serialize = "Ace")]
    Ace,
}

#[pymethods]
impl Rank {
    #[new]
    fn new(input: &str) -> Result<Self, GameError> {
        Rank::from_str(input).map_err(|_| GameError::InvalidRankString(input.to_owned()))
    }
}

impl From<u8> for Rank {
    fn from(value: u8) -> Self {
        *Rank::VARIANTS.get(value as usize).unwrap()
    }
}

use pyo3::{exceptions::PyTypeError, prelude::*};

/// A Python module implemented in Rust.
#[pymodule]
fn game(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Deal>()?;
    m.add_class::<Card>()?;
    m.add_class::<Rank>()?;
    m.add_class::<Suit>()?;
    m.add_class::<Call>()?;
    m.add_class::<Bid>()?;
    m.add_class::<Level>()?;
    m.add_class::<Seat>()?;
    m.add_class::<Vulnerability>()?;

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use test_case::test_case;

    #[test_case(Bid{level: Level::L1, strain: Strain::Suit(Suit::Club)}, 0)]
    #[test_case(Bid{level: Level::L1, strain: Strain::Suit(Suit::Heart)}, 2)]
    #[test_case(Bid{level: Level::L2, strain: Strain::Suit(Suit::Heart)}, 7)]
    fn test_bid_to_u8(bid: Bid, expected: u8) {
        assert_eq!(expected, bid.into());
        assert_eq!(bid, expected.into())
    }

    #[test]
    fn test_give_card() {
        let mut deal = Deal::random_deal();
        assert!(deal
            .give_card(
                Seat::rand(),
                Card {
                    suit: Suit::Heart,
                    rank: Rank::Two
                }
            )
            .is_err())
    }

    #[test]
    fn card_shuffle() {
        let mut rng = rand::thread_rng();
        let sut = Deck::shuffled(&mut rng);

        assert_eq!(sut.cards.len(), Deck::CARDS_IN_DECK);
        let unshuffled = Deck::default();
        for card in sut.cards {
            assert!(unshuffled.cards.contains(&card));
        }
    }

    #[test]
    fn test_next() {
        let s = Seat::South;
        assert_eq!(s.next_seat(), Seat::West);
    }

    #[test_case(Seat::West, Seat::East)]
    #[test_case(Seat::East, Seat::West)]
    #[test_case(Seat::North, Seat::South)]
    #[test_case(Seat::South, Seat::North)]
    fn partner(me: Seat, partner: Seat) {
        assert_eq!(me.partner(), partner);
    }

    #[test_case(0, Bid{ level: Level::L1, strain: Strain::Suit(Suit::Club)})]
    #[test_case(1, Bid{ level: Level::L1, strain: Strain::Suit(Suit::Diamond)})]
    #[test_case(4, Bid{ level: Level::L1, strain: Strain::NoTrump})]
    #[test_case(34, Bid{ level: Level::L7, strain: Strain::NoTrump})]
    fn bid_from_u8(value: u8, expect: Bid) {
        assert_eq!(expect, value.into());
    }

    #[test_case(0, Card{ suit: Suit::Club, rank: Rank::Two})]
    #[test_case(5, Card{ suit: Suit::Club, rank: Rank::Seven})]
    #[test_case(51, Card{ suit: Suit::Spade, rank: Rank::Ace})]
    fn card_from_u8(value: u8, expect: Card) {
        assert_eq!(expect, value.into());
    }

    #[test]
    fn quad_pass() -> Result<(), GameError> {
        let mut deal = Deal::random_deal();
        deal.execute_call_action(Call::Pass)?;
        deal.execute_actions_ids(&[35, 35, 35])?;
        assert_eq!(
            deal.execute_call_action(Call::Pass).err(),
            Some(GameError::InvalidState)
        );
        Ok(())
    }
}
