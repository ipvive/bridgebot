#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// The rules of the game of contract bridge as a tensor state machine.
typedef struct GameState {
	// Array variables       Indexed by dimensions:
	int8_t dealt_cards[4][4][13];  // seat, suit, rank. bool.
	int8_t played_cards[4][13];    // suit, rank. bool.
	int8_t min_length[4][4];       // seat, suit. 0-13=0-13.
	int8_t max_length[4][4];       // seat, suit. 0-13=0-13.
	int8_t first_to_mention[4][5]; // seat, strain. bool.
	int8_t tricks_taken[4];        // seat

	// Index variables,        -1=N/A,
	int8_t stage;                  // 0=bid 1=play 2=final 3=error.
	int8_t next_to_act;            // 0=S 1=W 2=N 3=E.

	int8_t pass_position;          // 0=1st...3=4th.
	int8_t last_bid_seat;          // 0=S 1=W 2=N 3=E.
	int8_t last_bid_level;         // 0-6=1-7.
	int8_t last_bid_strain;        // 0=Clubs...4=notrump.
	int8_t last_bid_double;        // 0=undoubled 1=doubled 2=redoubled.

	int8_t declarer;               // 0=S 1=W 2=N 3=E.

	int8_t trick_suit;             // 0=Clubs...3=Spades.
	int8_t trick_position;         // 0=1st...3=4th.
	int8_t trick_winning_seat;     // 0=S 1=W 2=N 3=E.
	int8_t trick_winning_suit;     // 0=Clubs...3=Spades.
	int8_t trick_winning_rank;     // 0-12=2-Ace.

	int8_t bidding_is_open;        // 0=false 1=true
} GameState;

typedef struct {
	int8_t actor;
	int8_t action;
} HistoryEntry;


#define STATE_SIZE sizeof(GameState)
#define NA -1
#define STAGE_BIDDING 0
#define STAGE_PLAY 1
#define STAGE_SCORING 2
#define STAGE_ERROR 3
#define CALL_PASS 0
#define CALL_DOUBLE 1
#define CALL_REDOUBLE 2


char* g_error_message = NULL;


static void set_error(GameState *state, char* msg) {
        if (state->stage != STAGE_ERROR) {
                state->stage = STAGE_ERROR;
                g_error_message = msg;
	}
}

static void execute_bid_action(GameState *state, int level_ix, int strain_ix) {
        if (state->stage != STAGE_BIDDING) {
		set_error(state, "stage for bid");
                return;
	}
        if (state->bidding_is_open) {
                if (level_ix < state->last_bid_level || (
                                level_ix == state->last_bid_level &&
                                strain_ix <= state->last_bid_strain)) {
                        set_error(state, "Insufficient bid");
                        return;
		}
	}
	state->bidding_is_open = 1;
        state->last_bid_seat = state->next_to_act;
        state->last_bid_level = level_ix;
        state->last_bid_strain = strain_ix;
        state->last_bid_double = 0;
        int partner_seat = (state->next_to_act + 2) % 4;
        if (!state->first_to_mention[partner_seat][strain_ix]) {
                state->first_to_mention[state->next_to_act][strain_ix] = 1;
	}
        state->pass_position = 0;
        state->next_to_act = (state->next_to_act + 1) % 4;
}

static void execute_call_action(GameState *state, int call) {
        if (state->stage != STAGE_BIDDING) {
		set_error(state,	"stage for call");
                return;
	}
        if (call == CALL_PASS) {
                if (state->pass_position == 3) {
                        state->stage = STAGE_SCORING;
                        state->next_to_act = NA;
                        state->pass_position = 0;
                } else if (state->bidding_is_open &&
				state->pass_position == 2) {
                        state->stage = STAGE_PLAY;
                        state->pass_position = NA;
                        state->trick_position = 0;
                        if (state->first_to_mention[state->last_bid_seat]
					[state->last_bid_strain]) {
                                state->declarer = state->last_bid_seat;
                        } else {
                                state->declarer =
				       	(state->last_bid_seat + 2) % 4;
			}
                        state->next_to_act = (state->declarer + 1) % 4;
                } else {
                        state->pass_position += 1;
                        state->next_to_act = (state->next_to_act + 1) % 4;
		}
        } else if (call == CALL_DOUBLE) {
                if (state->last_bid_double != CALL_PASS) {
			set_error(state, "double state for double");
                } else if (state->last_bid_seat % 2 == state->next_to_act % 2) {
                        set_error(state, "double of own sides' contract");
                } else {
                        state->last_bid_double = CALL_DOUBLE;
                        state->pass_position = 0;
                        state->next_to_act = (state->next_to_act + 1) % 4;
		}
        } else if (call == CALL_REDOUBLE) {
                if (state->last_bid_double != CALL_DOUBLE) {
                        set_error(state, "double state for redouble");
                } else if (state->last_bid_seat % 2 != state->next_to_act % 2) {
                        set_error(state, "redouble of other sides' contract");
                } else {
                        state->last_bid_double = CALL_REDOUBLE;
                        state->pass_position = 0;
                        state->next_to_act = (state->next_to_act + 1) % 4;
		}
	}
}

static bool is_strongest_card_played(GameState *state, int suit, int rank) {
        int trump = state->last_bid_strain;
        if (suit == trump && state->trick_winning_suit != trump) {
                return true;
	}
        if (suit != state->trick_winning_suit) {
                return false;
	}
        return rank > state->trick_winning_rank;
}

static void give_card(GameState *state, int seat, int suit, int rank) {
        if (state->stage == STAGE_ERROR) {
                return;
	}
	int num_duplicates = 0;
	for (int oseat = 0; oseat < 4; ++oseat) {
		num_duplicates += state->dealt_cards[oseat][suit][rank];
	}
	int num_cards = 0;
	for (int osuit = 0; osuit < 4; ++osuit) {
		for (int orank = 0; orank < 4; ++orank) {
			num_cards += state->dealt_cards[seat][osuit][orank];
		}
	}
        if (num_duplicates > 0) {
                set_error(state, "Duplicate card");
        } else if (state->played_cards[suit][rank] > 0) {
                set_error(state, "Card already played");
        } else if (num_cards >= 13) {
                set_error(state, "14 cards in hand");
        } else if (!state->dealt_cards[seat][suit][rank]) {
                state->dealt_cards[seat][suit][rank] = 1;
                state->min_length[seat][suit] += 1;
                if (state->min_length[seat][suit] >
			       	state->max_length[seat][suit]) {
                        set_error(state, "Revoke?");
                }
	}
}

static void execute_play_action(GameState *state, int suit, int rank) {
        if (state->stage != STAGE_PLAY) {
		set_error(state, "stage for call");
                return;
	}
        int seat = state->next_to_act;
        if (state->played_cards[suit][rank]) {
                set_error(state, "Card already played");
	}
        if (state->trick_position != 0 && suit != state->trick_suit) {
		int tsuit = state->trick_suit;
		for (int orank = 0; orank < 13; ++orank) {
			if (state->dealt_cards[seat][tsuit][orank] &&
				       	!state->played_cards[tsuit][orank]) {
				set_error(state, "Revoke");
				break;
			}
		}
                state->max_length[seat][state->trick_suit] =
			state->min_length[seat][state->trick_suit];
	}
        if (!state->dealt_cards[seat][suit][rank]) {
                give_card(state, seat, suit, rank);
	}
        if (state->stage == STAGE_ERROR) {
                return;
	}

        state->played_cards[suit][rank] = 1;
        if (state->trick_position == 0) {
                state->trick_suit = suit;
	}

        if (state->trick_position == 0 ||
		       	is_strongest_card_played(state, suit, rank)) {
                state->trick_winning_seat = seat;
                state->trick_winning_suit = suit;
                state->trick_winning_rank = rank;
	}

        if (state->trick_position < 3) {
                state->trick_position += 1;
                state->next_to_act = (state->next_to_act + 1) % 4;
        } else {
                state->trick_position = 0;
                state->next_to_act = state->trick_winning_seat;
                state->tricks_taken[state->trick_winning_seat] += 1;
		int total_tricks_taken = 0;
		for (int oseat = 0; oseat < 4; ++oseat) {
			total_tricks_taken += state->tricks_taken[oseat];
		}
                if (total_tricks_taken == 13) {
                        state->stage = STAGE_SCORING;
                        state->next_to_act = NA;
		}
	}
}

int execute_action_ids(
		GameState *state,
		int num_ids, int8_t *ids,
		HistoryEntry *history) {
	for (int i = 0; i < num_ids; ++i) {
		int id = ids[i];
		history[i].actor = state->next_to_act;
		history[i].action = id;
		if (id < 35) {
			execute_bid_action(state, id / 5, id % 5);
		} else if (id < 38) {
			execute_call_action(state, id - 35);
		} else {
			int suit = (id - 38) / 13;
			int rank = (id - 38) % 13;
			execute_play_action(state, suit, rank);
		}
		if (state->stage == STAGE_ERROR) {
			return i;
		}
	}
	return num_ids;
}

PyObject* wrap_execute_action_ids(PyObject *unused_self, PyObject* args) {
	PyObject *vector_obj = NULL;
	PyObject *ids_obj = NULL;
	PyObject *history_obj = NULL;
	PyArrayObject *vector = NULL;
	PyArrayObject *ids = NULL;
	PyArrayObject *history = NULL;
	GameState *state;
	int num_ids;
	int8_t *ids0;
	HistoryEntry *history00;
	int n;

	if (!PyArg_ParseTuple(args, "OOO", &vector_obj, &ids_obj, &history_obj))
		return NULL;
	vector = (PyArrayObject*) PyArray_FROM_OTF(
			vector_obj, NPY_INT8, NPY_ARRAY_INOUT_ARRAY2);
	ids = (PyArrayObject*) PyArray_FROM_OTF(
			ids_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY);
	history = (PyArrayObject*) PyArray_FROM_OTF(
			history_obj, NPY_INT8, NPY_ARRAY_OUT_ARRAY);
	if (vector == NULL || ids == NULL || history == NULL)
		goto fail;

	if (
		       	PyArray_NDIM(vector) != 1 ||
		       	PyArray_NDIM(ids) != 1 ||
		       	PyArray_NDIM(history) != 2 ||
		       	PyArray_DIMS(vector)[0] != STATE_SIZE ||
		       	PyArray_DIMS(history)[1] != 2) {
		PyErr_SetString(PyExc_ValueError, "hi");
		goto fail;
	}

	state = (GameState*) PyArray_GETPTR1(vector, 0);
	num_ids = PyArray_DIMS(ids)[0];
	ids0 = (int8_t*) PyArray_GETPTR1(ids, 0);
	history00 = (HistoryEntry*) PyArray_GETPTR2(history, 0, 0);

	n = execute_action_ids(state, num_ids, ids0, history00);

	PyArray_ResolveWritebackIfCopy(vector);
	Py_DECREF(vector);
	Py_DECREF(ids);
	PyArray_ResolveWritebackIfCopy(history);
	Py_DECREF(history);
	return Py_BuildValue("iz", n, g_error_message);

fail:
	PyArray_DiscardWritebackIfCopy(vector);
	Py_XDECREF(vector);
	Py_XDECREF(ids);
	PyArray_DiscardWritebackIfCopy(history);
	Py_XDECREF(history);
	return NULL;
}

#if 0
// fastgame.shrink_lengths(self._vector)
PyObject* wrap_shrink_lengths(PyObject *unused_self, PyObject* args) {
	PyObject *vector_obj = NULL;
	PyArrayObject *vector = NULL;
	GameState *state;
	bool ok;

	if (!PyArg_ParseTuple(args, "O", &vector_obj))
		return NULL;
	vector = (PyArrayObject*) PyArray_FROM_OTF(
			vector_obj, NPY_INT8, NPY_ARRAY_INOUT_ARRAY2);
	if (vector == NULL)
		goto fail;

	if ( PyArray_NDIM(vector) != 1 || PyArray_DIMS(vector)[0] != STATE_SIZE)
		goto fail;

	state = (GameState*) PyArray_GETPTR1(vector, 0);
	ok = shrink_lengths(state);

	PyArray_ResolveWritebackIfCopy(vector);
	Py_DECREF(vector);
	Py_INCREF(Py_None);
	return Py_None;

fail:
	PyArray_DiscardWritebackIfCopy(vector);
	Py_XDECREF(vector);
	return NULL;
}
#endif

static PyMethodDef fastgamemethods[] = {
	{
		"execute_action_ids",
		(PyCFunction)wrap_execute_action_ids,
		METH_VARARGS,
		"Execute a list of action id"
	},
#if 0
	{
		"shrink_lengths",
		(PyCFunction)wrap_shrink_lengths,
		METH_VARARGS,
		"Shrink {min,max}_lengths to reflect logical possibilities"
	},
#endif
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastgamemoduledef = {
  PyModuleDef_HEAD_INIT,
  "_fastgame",
  NULL,
  -1,
  fastgamemethods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC
PyInit__fastgame() {
	PyObject *m = NULL;
	if (STATE_SIZE != 330) {
		printf("GameState has size %ld, but need 330", STATE_SIZE);
	} else {
		m = PyModule_Create(&fastgamemoduledef);
		import_array();
	}
	return m;
}

//==============================================================
#if 0

bool accepted_words(Multiplier const* m, PyArrayObject *words) {
	int i;
	int max_length = PyArray_DIMS(words)[1];
	for (i = 0; i < PyArray_DIMS(words)[0]; ++i) {
		if (!accepted_word(m, PyArray_GETPTR2(words, i, 0), max_length))
			return false;
	}
	return true;
}

PyObject* wrap_accepted_words(PyObject *unused_self, PyObject* args) {
	PyObject *transition_index_obj = NULL;
	PyObject *transition_entry_obj = NULL;
	PyObject *accept_obj = NULL;
	PyObject *words_obj = NULL;

	PyArrayObject *transition_index = NULL;
	PyArrayObject *transition_entry = NULL;
	PyArrayObject *accept = NULL;
       	PyArrayObject *words = NULL;

	Multiplier multiplier;

	if (!PyArg_ParseTuple(args, "OOOO", &transition_index_obj,
				&transition_entry_obj, &accept_obj, &words_obj))
		return NULL;
	transition_index = (PyArrayObject*) PyArray_FROM_OTF(
			transition_index_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
	transition_entry = (PyArrayObject*) PyArray_FROM_OTF(
			transition_entry_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
	accept = (PyArrayObject*) PyArray_FROM_OTF(
			accept_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
	words = (PyArrayObject*) PyArray_FROM_OTF(
			words_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY2);
	if (transition_index == NULL || transition_entry == NULL ||
			accept == NULL || words == NULL)
		goto fail;

	if (
			PyArray_NDIM(transition_index) != 3 ||
			PyArray_NDIM(transition_entry) != 2 ||
			PyArray_NDIM(accept) != 1 ||
			PyArray_NDIM(words) != 2 ||
		 	PyArray_DIMS(transition_index)[2] != 2 ||
			PyArray_DIMS(transition_entry)[1] != 2 ||
			PyArray_DIMS(transition_index)[0] != PyArray_DIMS(accept)[0])
		goto fail;

	multiplier.num_states = PyArray_DIMS(transition_index)[0];
	multiplier.num_letters = PyArray_DIMS(transition_index)[1] - 1;
	multiplier.num_entries = PyArray_DIMS(transition_entry)[0];
	multiplier.transition_index = (IndexEntry*) PyArray_GETPTR3(transition_index, 0, 0, 0);
	multiplier.transition_entry = (Entry*) PyArray_GETPTR2(transition_entry, 0, 0);
        multiplier.accept = (int32_t*) PyArray_GETPTR1(accept, 0);

	if (!accepted_words(&multiplier, words))
		goto fail;

	Py_DECREF(transition_index);
	Py_DECREF(transition_entry);
	Py_DECREF(accept);
	PyArray_ResolveWritebackIfCopy(words);
	Py_DECREF(words);
	Py_INCREF(Py_None);
	return Py_None;

fail:
	Py_XDECREF(transition_index);
	Py_XDECREF(transition_entry);
	Py_XDECREF(accept);
	PyArray_DiscardWritebackIfCopy(words);
	Py_XDECREF(words);
	return NULL;
}
#endif
