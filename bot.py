#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Telegram experiment bot implementing the advisor game with two treatments:
1) NO LEADERBOARD
2) LEADERBOARD (anonymized IDs, live rank table, bonus pool)

Design:
- Each participant is randomized into an order of treatments: [LB, NL] or [NL, LB].
- They play TWO real games (stages). In Stage 1 they see their first treatment,
  in Stage 2 they see the opposite treatment.
- Tokens and actions are logged to SQLite for later analysis.
- An optional /demo game lets them practice with no effect on tokens.
- Fallback logic: 60s after the first /join:
    * If < 5 stage-1 participants, everyone gets LEADERBOARD first.
    * If >= 5 stage-1 participants, they are randomly split ~half/half between LB and NL.
  After fallback, both Stage 1 and Stage 2 games may start with smaller-than-GROUP_SIZE groups.
"""

import asyncio
import logging
import math
import random
import string
import uuid
import time
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import aiosqlite
from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# TODO: replace with your real token from @BotFather
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

DB_NAME = os.getenv("DB_NAME", "experiment.db")

# Allow users to replay the experiment after completing both stages.
# Set to True if you want participants to be able to run multiple times.
ALLOW_MULTIPLE_RUNS = True

GROUP_SIZE = 100                 # participants per game (target)
ROUNDS_PER_GAME = 5           # real rounds per stage/game
DEMO_ROUNDS = 5                # short practice game length
ROUND_TIMEOUT_SECONDS = 60     # timeout per round (seconds)

INITIAL_Q = 0.55

TREATMENT_LEADERBOARD = "leaderboard"
TREATMENT_NO_LEADERBOARD = "no_leaderboard"
TREATMENTS = (TREATMENT_LEADERBOARD, TREATMENT_NO_LEADERBOARD)

# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------


@dataclass
class ParticipantState:
    """Per-participant experiment assignment and progress."""

    user_id: int
    first_treatment: str
    current_stage: int = 1  # 1 or 2; >2 means done
    done: bool = False


@dataclass
class GameSession:
    """In-memory state for a single game (either real or demo)."""

    id: str
    players: List[int]
    treatment: str
    stage: int
    q: float = INITIAL_Q
    round: int = 0
    current_votes: Dict[int, str] = field(default_factory=dict)
    advisor_prediction: Optional[str] = None
    true_outcome: Optional[str] = None
    timeout_task: Optional[asyncio.Task] = None
    round_closed: bool = False
    rng: random.Random = field(default_factory=random.Random)
    token_balances: Dict[int, float] = field(default_factory=dict)
    total_rounds: int = ROUNDS_PER_GAME
    is_demo: bool = False

    def reset_round_state(self) -> None:
        self.current_votes.clear()
        self.advisor_prediction = None
        self.true_outcome = None
        self.round_closed = False

    def compute_next_q(self) -> Tuple[float, int]:
        """Compute q_{t+1} and I_t from current votes using the experiment's formula."""
        N = len(self.players)
        I_t = sum(1 for v in self.current_votes.values() if v == "IMPROVE")

        numerator = I_t + 0.5
        denominator = (N - I_t) + 0.5
        K_t = (numerator / denominator) ** (1.0 / N)

        q_curr = self.q
        new_q = (q_curr * K_t) / ((1.0 - q_curr) + (q_curr * K_t))
        return new_q, I_t


# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------

router = Router()

# Active running game sessions
active_games: Dict[str, GameSession] = {}

# Map user_id -> game_id
user_game_map: Dict[int, str] = {}

# Participant experiment state (randomization, stage)
participant_states: Dict[int, ParticipantState] = {}

# Anonymized IDs (user_id -> short numeric string)
anon_ids: Dict[int, str] = {}

# Waiting queues keyed by (treatment, stage)
waiting_queues: Dict[Tuple[str, int], List[int]] = defaultdict(list)

# Fallback logic: first join time and joined users
first_join_time: Optional[float] = None
joined_users_since_first: Set[int] = set()
fallback_task: Optional[asyncio.Task] = None
fallback_mode: bool = False           # True after 60s decision is made
fallback_lb_for_all: bool = False     # True iff the <5 case was chosen


async def maybe_reset_fallback() -> None:
    """Reset fallback window if no stage-1 participants are currently active.

    This lets us reuse the same DB for multiple experiment runs without
    forcing future participants into the old fallback decision.
    """
    global fallback_mode, fallback_lb_for_all, first_join_time, joined_users_since_first, fallback_task

    if not fallback_mode:
        return

    # First check in-memory states
    for state in participant_states.values():
        if not state.done and state.current_stage == 1:
            # There is still an active stage-1 participant; do not reset.
            return

    # Double-check in DB in case memory state was lost/reloaded.
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT 1 FROM user_experiments WHERE current_stage = 1 AND done = 0 LIMIT 1"
        ) as cur:
            row = await cur.fetchone()

    if row:
        # There is still someone in stage 1 according to DB.
        return

    # No active stage-1 participants: start a fresh waiting window next time someone joins.
    fallback_mode = False
    fallback_lb_for_all = False
    first_join_time = None
    joined_users_since_first.clear()
    if fallback_task is not None:
        fallback_task.cancel()
        fallback_task = None


# ---------------------------------------------------------------------------
# DB HELPERS
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create all required tables if they do not exist."""
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id   INTEGER PRIMARY KEY,
                username  TEXT,
                anon_id   TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_experiments (
                user_id        INTEGER PRIMARY KEY,
                first_treatment TEXT,
                current_stage   INTEGER DEFAULT 1,
                done            INTEGER DEFAULT 0
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                game_id    TEXT PRIMARY KEY,
                treatment  TEXT,
                stage      INTEGER,
                rng_seed   INTEGER,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished   INTEGER DEFAULT 0
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS wallets (
                user_id INTEGER,
                game_id TEXT,
                tokens  REAL DEFAULT 0,
                bonus   REAL DEFAULT 0,
                PRIMARY KEY (user_id, game_id)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id          TEXT,
                stage            INTEGER,
                treatment        TEXT,
                round_num        INTEGER,
                user_id          INTEGER,
                action_type      TEXT,
                advisor_accuracy REAL,
                prediction       TEXT,
                outcome          TEXT,
                payoff           REAL,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.commit()


async def reset_experiment_state(user_id: int) -> None:
    """Completely reset experiment state for a user (to allow multiple runs)."""
    participant_states.pop(user_id, None)
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("DELETE FROM user_experiments WHERE user_id = ?", (user_id,))
        await db.commit()


async def export_tables_to_csv(base_dir: str = ".") -> None:
    """Export all main tables to CSV files in the given directory (not exposed via Telegram)."""
    tables = ["users", "user_experiments", "games", "wallets", "actions"]
    async with aiosqlite.connect(DB_NAME) as db:
        for table in tables:
            path = os.path.join(base_dir, f"{table}.csv")
            async with db.execute(f"PRAGMA table_info({table})") as cur:
                cols_info = await cur.fetchall()
            if not cols_info:
                continue
            col_names = [c[1] for c in cols_info]
            async with db.execute(f"SELECT * FROM {table}") as cur:
                rows = await cur.fetchall()
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(col_names)
                writer.writerows(rows)


async def get_or_create_user(user_id: int, username: str) -> str:
    """
    Ensure user row exists and return the anonymized ID for leaderboards.
    """
    if user_id in anon_ids:
        return anon_ids[user_id]

    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "INSERT OR IGNORE INTO users (user_id, username) VALUES (?, ?)",
            (user_id, username),
        )
        await db.commit()

        async with db.execute(
            "SELECT anon_id FROM users WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()

        if row and row[0]:
            anon = row[0]
        else:
            # Generate unique 6-digit numeric ID
            while True:
                anon = "".join(random.choices(string.digits, k=6))
                async with db.execute(
                    "SELECT 1 FROM users WHERE anon_id = ?", (anon,)
                ) as cur2:
                    exists = await cur2.fetchone()
                if not exists:
                    break
            await db.execute(
                "UPDATE users SET anon_id = ? WHERE user_id = ?",
                (anon, user_id),
            )
            await db.commit()

    anon_ids[user_id] = anon
    return anon


async def get_or_create_experiment_state(user_id: int) -> ParticipantState:
    """
    Fetch or create experiment assignment for this user.
    Randomizes first_treatment between leaderboard and no-leaderboard,
    unless fallback_mode + fallback_lb_for_all are active.
    """
    if user_id in participant_states:
        return participant_states[user_id]

    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT first_treatment, current_stage, done FROM user_experiments WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()

        if row:
            first_treatment, current_stage, done = row
        else:
            if fallback_mode and fallback_lb_for_all:
                first_treatment = TREATMENT_LEADERBOARD
            else:
                first_treatment = random.choice(TREATMENTS)
            current_stage = 1
            done = 0
            await db.execute(
                """
                INSERT INTO user_experiments (user_id, first_treatment, current_stage, done)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, first_treatment, current_stage, done),
            )
            await db.commit()

    state = ParticipantState(
        user_id=user_id,
        first_treatment=first_treatment,
        current_stage=int(current_stage),
        done=bool(done),
    )
    participant_states[user_id] = state
    return state


async def update_experiment_state(state: ParticipantState) -> None:
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            """
            UPDATE user_experiments
            SET first_treatment = ?, current_stage = ?, done = ?
            WHERE user_id = ?
            """,
            (state.first_treatment, state.current_stage, int(state.done), state.user_id),
        )
        await db.commit()


async def insert_game_record(game: GameSession, rng_seed: int) -> None:
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO games (game_id, treatment, stage, rng_seed)
            VALUES (?, ?, ?, ?)
            """,
            (game.id, game.treatment, game.stage, rng_seed),
        )
        await db.commit()


async def ensure_wallet(user_id: int, game_id: str) -> None:
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "INSERT OR IGNORE INTO wallets (user_id, game_id) VALUES (?, ?)",
            (user_id, game_id),
        )
        await db.commit()


async def update_wallet_tokens(user_id: int, game_id: str, delta: float) -> None:
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "UPDATE wallets SET tokens = tokens + ? WHERE user_id = ? AND game_id = ?",
            (delta, user_id, game_id),
        )
        await db.commit()


async def update_wallet_bonus(user_id: int, game_id: str, bonus: float) -> None:
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "UPDATE wallets SET bonus = bonus + ? WHERE user_id = ? AND game_id = ?",
            (bonus, user_id, game_id),
        )
        await db.commit()


async def log_round_action(
    game: GameSession,
    user_id: int,
    action: str,
    accuracy: float,
    payoff: float,
) -> None:
    """
    Log one participant's action for a round and update their wallet tokens.
    """
    if game.is_demo:
        return

    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            """
            INSERT INTO actions (
                game_id, stage, treatment, round_num, user_id,
                action_type, advisor_accuracy, prediction, outcome, payoff
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game.id,
                game.stage,
                game.treatment,
                game.round,
                user_id,
                action,
                accuracy,
                game.advisor_prediction,
                game.true_outcome,
                payoff,
            ),
        )
        await db.commit()

    await ensure_wallet(user_id, game.id)
    await update_wallet_tokens(user_id, game.id, payoff)


async def get_wallet_summary_for_game(game_id: str) -> Dict[int, Tuple[float, float]]:
    """
    Return mapping user_id -> (tokens, bonus) for a given game.
    """
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT user_id, tokens, bonus FROM wallets WHERE game_id = ?",
            (game_id,),
        ) as cur:
            rows = await cur.fetchall()
    result: Dict[int, Tuple[float, float]] = {}
    for user_id, tokens, bonus in rows:
        result[int(user_id)] = (float(tokens or 0.0), float(bonus or 0.0))
    return result


async def mark_game_finished(game_id: str) -> None:
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("UPDATE games SET finished = 1 WHERE game_id = ?", (game_id,))
        await db.commit()


async def compute_total_tokens_for_user(user_id: int) -> float:
    """
    Sum tokens + bonus across all games for this user.
    """
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT SUM(tokens + bonus) FROM wallets WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row or row[0] is None:
        return 0.0
    return float(row[0])


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS & FALLBACK LOGIC
# ---------------------------------------------------------------------------


def get_user_game(user_id: int) -> Optional[GameSession]:
    game_id = user_game_map.get(user_id)
    if not game_id:
        return None
    return active_games.get(game_id)


def opposite_treatment(t: str) -> str:
    return (
        TREATMENT_LEADERBOARD
        if t == TREATMENT_NO_LEADERBOARD
        else TREATMENT_NO_LEADERBOARD
    )


def current_treatment_for_state(state: ParticipantState) -> Optional[str]:
    """
    Map participant's stage + first_treatment to their current treatment.
    """
    if state.done:
        return None
    if state.current_stage == 1:
        return state.first_treatment
    if state.current_stage == 2:
        return opposite_treatment(state.first_treatment)
    return None


def pretty_treatment(t: str) -> str:
    return "LEADERBOARD" if t == TREATMENT_LEADERBOARD else "NO LEADERBOARD"


def record_join_for_fallback(user_id: int) -> None:
    """Record that this user has joined (for fallback logic)."""
    global first_join_time
    if first_join_time is None:
        first_join_time = time.monotonic()
    joined_users_since_first.add(user_id)


async def ensure_fallback_check_scheduled(bot: Bot) -> None:
    """Schedule a one-time check 60s after the first /join."""
    global fallback_task
    if fallback_mode:
        return
    if fallback_task is not None:
        return
    loop = asyncio.get_running_loop()
    fallback_task = loop.create_task(fallback_checker(bot))


async def fallback_checker(bot: Bot) -> None:
    """
    After 60s from the first join, decide treatments for stage-1 users and allow small groups.

    If fewer than 5 stage-1 participants: everyone gets LEADERBOARD first.
    If 5 or more stage-1 participants: randomly split ~half/half into LB and NL.
    In both cases, games may start with smaller-than-GROUP_SIZE groups for all stages.
    """
    global fallback_task, fallback_mode, fallback_lb_for_all

    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        return

    # If already decided or no joined users at all, nothing to do
    if fallback_mode or not joined_users_since_first:
        fallback_task = None
        return

    # Determine stage-1 users who have actually joined during this waiting window
    stage1_users: Set[int] = set()
    for state in participant_states.values():
        if not state.done and state.current_stage == 1 and get_user_game(state.user_id) is None:
            if state.user_id in joined_users_since_first:
                stage1_users.add(state.user_id)

    # If nobody is waiting for Stage 1, we just enable fallback_mode and return.
    if not stage1_users:
        fallback_mode = True
        fallback_task = None
        return

    num = len(stage1_users)

    # Remove all stage-1 users from any existing queues; we will start their games directly.
    for key in list(waiting_queues.keys()):
        waiting_queues[key] = [uid for uid in waiting_queues[key] if uid not in stage1_users]

    if num < 5:
        # Case 1: fewer than 5 participants → everyone gets LEADERBOARD first
        fallback_lb_for_all = True
        # Update first_treatment in memory + DB
        for state in participant_states.values():
            if state.user_id in stage1_users and not state.done:
                if state.first_treatment != TREATMENT_LEADERBOARD:
                    state.first_treatment = TREATMENT_LEADERBOARD
                    await update_experiment_state(state)

        # Start Stage 1 games for all stage-1 users in groups of up to GROUP_SIZE
        stage1_list = list(stage1_users)
        while stage1_list:
            group = stage1_list[:GROUP_SIZE]
            stage1_list = stage1_list[GROUP_SIZE:]
            await start_game(bot, group, TREATMENT_LEADERBOARD, stage=1, is_demo=False)
    else:
        # Case 2: 5 or more participants → split approximately equally into LB and NL
        fallback_lb_for_all = False
        shuffled = list(stage1_users)
        random.shuffle(shuffled)
        half = len(shuffled) // 2
        group_lb = shuffled[:half]
        group_nl = shuffled[half:]

        # Update first_treatment for both groups
        for state in participant_states.values():
            if state.user_id in group_lb and not state.done:
                if state.first_treatment != TREATMENT_LEADERBOARD:
                    state.first_treatment = TREATMENT_LEADERBOARD
                    await update_experiment_state(state)
            elif state.user_id in group_nl and not state.done:
                if state.first_treatment != TREATMENT_NO_LEADERBOARD:
                    state.first_treatment = TREATMENT_NO_LEADERBOARD
                    await update_experiment_state(state)

        # Start Stage 1 games for LB group
        lb_list = list(group_lb)
        while lb_list:
            group = lb_list[:GROUP_SIZE]
            lb_list = lb_list[GROUP_SIZE:]
            await start_game(bot, group, TREATMENT_LEADERBOARD, stage=1, is_demo=False)

        # Start Stage 1 games for NL group
        nl_list = list(group_nl)
        while nl_list:
            group = nl_list[:GROUP_SIZE]
            nl_list = nl_list[GROUP_SIZE:]
            await start_game(bot, group, TREATMENT_NO_LEADERBOARD, stage=1, is_demo=False)

    # From now on we allow starting games with small groups as well
    fallback_mode = True
    fallback_task = None

    # Also try to start any Stage 2 games that might already have queues
    await try_matchmaking(bot)


# ---------------------------------------------------------------------------
# MATCHMAKING & GAME CONTROL
# ---------------------------------------------------------------------------


async def try_matchmaking(bot: Bot) -> None:
    """
    Start games from waiting queues.

    Normally we require full groups of size GROUP_SIZE.
    After fallback_mode is True, we allow smaller groups: any remaining
    participants in a queue will be started together.
    """
    for stage in (1, 2):
        for treatment in TREATMENTS:
            key = (treatment, stage)
            queue = waiting_queues.get(key)
            if not queue:
                continue

            # First, start as many full groups as possible
            while len(queue) >= GROUP_SIZE:
                players = [queue.pop(0) for _ in range(GROUP_SIZE)]
                await start_game(bot, players, treatment, stage, is_demo=False)

            # After fallback, for any stage we also start any remaining smaller group
            if fallback_mode and queue:
                players = queue.copy()
                queue.clear()
                await start_game(bot, players, treatment, stage, is_demo=False)


async def start_game(
    bot: Bot, players: List[int], treatment: str, stage: int, is_demo: bool
) -> None:
    """
    Create a new GameSession (demo or real) and start the first round.
    """
    prefix = "DEMO-" if is_demo else ""
    game_id = prefix + uuid.uuid4().hex
    rng_seed = random.randint(1, 2_000_000_000)
    rng = random.Random(rng_seed)

    game = GameSession(
        id=game_id,
        players=players,
        treatment=treatment,
        stage=stage,
        q=INITIAL_Q,
        rng=rng,
        token_balances={uid: 0.0 for uid in players},
        total_rounds=DEMO_ROUNDS if is_demo else ROUNDS_PER_GAME,
        is_demo=is_demo,
    )

    active_games[game_id] = game
    for uid in players:
        user_game_map[uid] = game_id

    if not is_demo:
        await insert_game_record(game, rng_seed)

    # Intro messages
    treatment_text = pretty_treatment(treatment)
    for uid in players:
        intro_lines = []
        if is_demo:
            intro_lines.append("<b>Demo game</b> (practice only, tokens do NOT count).")
        else:
            intro_lines.append(f"<b>Stage {stage}: {treatment_text} treatment</b>")
        intro_lines.append(
            "In each round, you will see the advisor's current accuracy q, its prediction "
            "(UP or DOWN), and you must choose between <b>Improve</b> and <b>Trade</b>."
        )
        if treatment == TREATMENT_LEADERBOARD:
            intro_lines.append(
                "This is the LEADERBOARD treatment: after each round you will see an anonymized "
                "ranking of players by cumulative tokens, using your personal ID."
            )
        else:
            intro_lines.append(
                "This is the NO-LEADERBOARD treatment: you will NOT see others' rankings."
            )
        await bot.send_message(uid, "\n".join(intro_lines))

    await start_new_round(bot, game)


async def start_new_round(bot: Bot, game: GameSession) -> None:
    """
    Start the next round: draw outcome + prediction and ask all players for actions.
    """
    game.round += 1
    game.reset_round_state()

    # Generate underlying outcome and advisor prediction using per-game RNG
    true_outcome = "UP" if game.rng.random() < 0.5 else "DOWN"
    advisor_correct = game.rng.random() < game.q
    prediction = true_outcome if advisor_correct else ("DOWN" if true_outcome == "UP" else "UP")

    game.true_outcome = true_outcome
    game.advisor_prediction = prediction

    ev = 2.0 * game.q - 1.0

    kb = InlineKeyboardBuilder()
    kb.button(text="Improve (0 tokens)", callback_data="act_IMPROVE")
    kb.button(text="Trade (bet on prediction)", callback_data="act_TRADE")
    kb.adjust(1)

    # Build table of how next-round q would change depending on number of 'Improve' votes
    N = len(game.players)
    table_lines = ["I  q_next", "-----------"]
    for I in range(0, N + 1):
        numerator = I + 0.5
        denominator = (N - I) + 0.5
        K_t = (numerator / denominator) ** (1.0 / max(N, 1))
        q_next = (game.q * K_t) / ((1.0 - game.q) + (game.q * K_t))
        table_lines.append(f"{I:2d}  {q_next:.3f}")
    table_text = "<pre>" + "\n".join(table_lines) + "</pre>"

    for uid in game.players:
        lines = [
            f"<b>Round {game.round}/{game.total_rounds}</b>",
            f"Advisor accuracy q: <b>{game.q:.3f}</b>",
            f"Advisor prediction: <b>{prediction}</b>",
            f"Expected value of 1-token trade: <b>{ev:.3f}</b>",
            "",
            "<b>How next-round accuracy q will change depending on Improve votes:</b>",
            table_text,
            "",
            "Choose your action:",
        ]
        await bot.send_message(uid, "\n".join(lines), reply_markup=kb.as_markup())

    # Round timeout
    loop = asyncio.get_running_loop()
    if game.timeout_task:
        game.timeout_task.cancel()
    game.timeout_task = loop.create_task(round_timeout(bot, game))


async def round_timeout(bot: Bot, game: GameSession) -> None:
    """
    If not all players respond in time, auto-fill missing answers as 'IMPROVE'.
    """
    try:
        await asyncio.sleep(ROUND_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        return

    # If the round has already been closed (e.g. all players voted and
    # process_round_end has run), do nothing.
    if game.round_closed:
        return

    # Auto-fill missing votes as 'IMPROVE' for all players who didn't act
    for uid in game.players:
        if uid not in game.current_votes:
            game.current_votes[uid] = "IMPROVE"

    # Let process_round_end handle closing the round and progressing the game
    await process_round_end(bot, game)


async def process_round_end(bot: Bot, game: GameSession) -> None:
    """
    Close the round, compute new q, payoffs, and possibly start next round.
    """
    if game.round_closed:
        return
    game.round_closed = True

    if game.timeout_task:
        game.timeout_task.cancel()
        game.timeout_task = None

    old_q = game.q
    new_q, I_t = game.compute_next_q()

    results_header = (
        f"<b>Round {game.round} results</b>\n\n"
        f"True outcome: <b>{game.true_outcome}</b>\n"
        f"Advisor prediction: <b>{game.advisor_prediction}</b>\n"
        f"Votes to improve: <b>{I_t}/{len(game.players)}</b>\n"
        f"Next-round accuracy q: <b>{new_q:.3f}</b>\n\n"
    )

    for uid in game.players:
        action = game.current_votes.get(uid, "IMPROVE")
        payoff = 0.0
        if action == "TRADE":
            advisor_right = game.advisor_prediction == game.true_outcome
            payoff = 1.0 if advisor_right else -1.0

        game.token_balances[uid] = game.token_balances.get(uid, 0.0) + payoff

        if not game.is_demo:
            await log_round_action(game, uid, action, old_q, payoff)

        msg = (
            results_header
            + f"You chose: <b>{action}</b>\n"
            + f"Your payoff this round: <b>{payoff:.0f} tokens</b>\n"
            + f"Your cumulative tokens this game: <b>{game.token_balances[uid]:.0f}</b>"
        )
        await bot.send_message(uid, msg)

    game.q = new_q

    # Send leaderboard if applicable
    if not game.is_demo and game.treatment == TREATMENT_LEADERBOARD:
        await send_leaderboard(bot, game)

    # Start next round or finish game
    if game.round >= game.total_rounds:
        await finalize_game(bot, game)
    else:
        await start_new_round(bot, game)


async def send_leaderboard(bot: Bot, game: GameSession) -> None:
    """
    Broadcast anonymized leaderboard for this game to all players.
    """
    rows = []
    for uid in game.players:
        anon = anon_ids.get(uid, str(uid))
        tokens = game.token_balances.get(uid, 0.0)
        rows.append((uid, anon, tokens))

    rows.sort(key=lambda r: (-r[2], r[1]))

    lines = [f"<b>Leaderboard after round {game.round}</b>"]
    for rank, (_, anon, tokens) in enumerate(rows, start=1):
        lines.append(f"{rank}) ID {anon}: {tokens:.0f} tokens")

    message = "\n".join(lines)
    for uid in game.players:
        await bot.send_message(uid, message)


async def finalize_game(bot: Bot, game: GameSession) -> None:
    """
    Wrap up after the last round: bonuses, stage transitions, etc.
    """
    # Demo games do not affect wallets or stages.
    if game.is_demo:
        try:
            for uid in game.players:
                await bot.send_message(
                    uid,
                    "Demo game finished.\n"
                    "These tokens were for <b>practice only</b> and will not affect payment.\n\n"
                    "When you're ready, send /join to enter the real experiment.",
                )
        finally:
            # Always free in-memory mappings, even if sending messages fails.
            for uid in game.players:
                # Only clear mapping if it still points to this game.
                if user_game_map.get(uid) == game.id:
                    user_game_map.pop(uid, None)
            active_games.pop(game.id, None)
        return

    # Real game: record outcome, possibly compute leaderboard bonuses,
    # notify players, and advance them to the next stage.
    try:
        # Mark game finished in DB
        await mark_game_finished(game.id)

        # Leaderboard bonus pool (only for leaderboard treatment)
        bonus_per_user: Dict[int, float] = {uid: 0.0 for uid in game.players}
        if game.treatment == TREATMENT_LEADERBOARD and game.players:
            tokens_list = [game.token_balances.get(uid, 0.0) for uid in game.players]
            avg_tokens = sum(tokens_list) / len(tokens_list) if tokens_list else 0.0
            pool = max(avg_tokens, 0.0)
            if pool > 0.0:
                # Sort players by tokens descending
                sorted_players = sorted(
                    ((uid, game.token_balances.get(uid, 0.0)) for uid in game.players),
                    key=lambda x: -x[1],
                )
                base_winners = max(1, math.ceil(len(sorted_players) / 3))
                threshold = sorted_players[base_winners - 1][1]
                winners = [uid for uid, tok in sorted_players if tok >= threshold]
                if winners:
                    per_bonus = pool / len(winners)
                    for uid in winners:
                        bonus_per_user[uid] = per_bonus
                        await update_wallet_bonus(uid, game.id, per_bonus)

        # Notify players about final tokens and bonus for this game
        for uid in game.players:
            total_tokens = game.token_balances.get(uid, 0.0)
            bonus = bonus_per_user.get(uid, 0.0)
            lines = [
                f"<b>Game finished (Stage {game.stage}, {pretty_treatment(game.treatment)}).</b>",
                f"Your token profit in this game: <b>{total_tokens:.0f}</b>",
            ]
            if game.treatment == TREATMENT_LEADERBOARD:
                lines.append(
                    f"Bonus tokens from leaderboard pool: <b>{bonus:.2f}</b>"
                )
            lines.append(
                "These values will be used to compute your final payout across both stages."
            )
            await bot.send_message(uid, "\n".join(lines))

        # Move participants to next stage or mark experiment finished
        await advance_participants_after_game(bot, game)
    finally:
        # Always clear in-memory game mappings so users are never stuck
        # in an 'already in an active game' state if something above fails.
        # However, only remove a user's mapping if it still points to this game,
        # to avoid erasing assignments to newer games (e.g. Stage 2).
        for uid in game.players:
            if user_game_map.get(uid) == game.id:
                user_game_map.pop(uid, None)
        active_games.pop(game.id, None)


async def advance_participants_after_game(bot: Bot, game: GameSession) -> None:
    """
    After a real game, advance each participant's stage and (if needed)
    enqueue them for the opposite treatment.
    """
    for uid in game.players:
        state = participant_states.get(uid)
        if not state or state.done:
            continue

        if state.current_stage == 1:
            # Move to Stage 2 with opposite treatment
            state.current_stage = 2
            await update_experiment_state(state)
            next_treatment = current_treatment_for_state(state)
            if next_treatment is None:
                state.done = True
                await update_experiment_state(state)
                continue

            key = (next_treatment, state.current_stage)
            if uid not in waiting_queues[key]:
                waiting_queues[key].append(uid)
            await bot.send_message(
                uid,
                f"Stage 1 finished. Next you will play <b>Stage 2</b>\n"
                f"with treatment: <b>{pretty_treatment(next_treatment)}</b>.\n"
                "Please wait to be matched with other participants.",
            )

        elif state.current_stage == 2:
            # Both stages completed
            state.current_stage = 3
            state.done = True
            await update_experiment_state(state)
            total = await compute_total_tokens_for_user(uid)
            await bot.send_message(
                uid,
                "You have completed both stages of the experiment.\n"
                f"Your combined token profit across all games is: <b>{total:.2f}</b>.\n"
                "Thank you for participating!",
            )

    # Try to start any Stage 2 games that now have enough players (or small groups if fallback_mode)
    await try_matchmaking(bot)


# ---------------------------------------------------------------------------
# TELEGRAM HANDLERS
# ---------------------------------------------------------------------------


@router.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    user = message.from_user
    username = user.username or user.full_name or "participant"
    anon = await get_or_create_user(user.id, username)
    text = (
        "Welcome to the advisor experiment bot.\n\n"
        "<b>What is this game about?</b>\n"
        "You interact with an \"advisor\" that predicts whether the outcome will be <b>UP</b> or <b>DOWN</b>.\n"
        "The advisor has a current accuracy <code>q</code> (between 0 and 1). In each round you see:\n"
        "• the current accuracy q,\n"
        "• the advisor's prediction (UP or DOWN),\n"
        "• and a table showing how next-round q would change depending on how many people choose Improve.\n\n"
        "<b>Your decision each round</b>\n"
        "You have two buttons:\n"
        "• <b>Improve</b> – you earn 0 tokens this round, but your vote helps change q for future rounds;\n"
        "• <b>Trade</b> – you bet 1 token on the advisor's prediction: you gain +1 token if it is correct, and lose 1 token if it is wrong.\n"
        "Your tokens are accumulated over all rounds of a game.\n\n"
        "<b>Treatments</b>\n"
        "You will play the game twice (two stages) with the same basic mechanics but different information:\n"
        "• In the <b>LEADERBOARD</b> treatment, after each round you see an anonymized ranking of players by tokens.\n"
        f"  Your personal anonymized ID is: <b>{anon}</b>, and this is what others see on the leaderboard.\n"
        "• In the <b>NO LEADERBOARD</b> treatment, you only see your own tokens, no ranking.\n"
        "The order of these two treatments (which one you see first) is determined by the experiment.\n\n"
        "<b>Demo vs real experiment</b>\n"
        "• /demo – plays a short practice game (few rounds). Tokens here are NOT used for payment or analysis,\n"
        "  and the treatment is fixed (no leaderboard).\n"
        "• /join – enters the real experiment. You will be matched with other participants and play two full stages,\n"
        "  one under each treatment. Tokens from these games are recorded and can be used for payment.\n\n"
        "<b>Matching and waiting</b>\n"
        "After /join you may need to wait to be matched with others. The system groups available participants and\n"
        "starts the game automatically after a short waiting period.\n\n"
        "To proceed now:\n"
        "• Send /demo to practice, OR\n"
        "• Send /join to enter the real experiment."
    )
    await message.answer(text)


@router.message(Command("demo"))
async def cmd_demo(message: types.Message) -> None:
    user = message.from_user
    user_id = user.id

    if get_user_game(user_id):
        await message.answer(
            "You are already in an active game. Finish it before starting a demo."
        )
        return

    username = user.username or user.full_name or "participant"
    await get_or_create_user(user_id, username)

    await message.answer(
        "Starting a short <b>demo game</b>. Tokens here will NOT be recorded for payment."
    )
    await start_game(
        message.bot,
        players=[user_id],
        treatment=TREATMENT_NO_LEADERBOARD,
        stage=0,
        is_demo=True,
    )


@router.message(Command("join"))
async def cmd_join(message: types.Message) -> None:
    user = message.from_user
    user_id = user.id

    if get_user_game(user_id):
        await message.answer(
            "You are already in an active game. Please finish it before joining again."
        )
        return

    # If the previous experiment run has fully finished (no one in Stage 1),
    # reset the fallback window so a new 60s decision can be made.
    await maybe_reset_fallback()

    # Record this join for fallback logic and schedule the 60s check if needed
    record_join_for_fallback(user_id)
    await ensure_fallback_check_scheduled(message.bot)

    username = user.username or user.full_name or "participant"
    anon = await get_or_create_user(user_id, username)
    state = await get_or_create_experiment_state(user_id)
    treatment = current_treatment_for_state(state)

    if treatment is None:
        if ALLOW_MULTIPLE_RUNS:
            await message.answer(
                "You have already completed both stages of the experiment.\n"
                "Starting a new run for you from Stage 1."
            )
            await reset_experiment_state(user_id)
            state = await get_or_create_experiment_state(user_id)
            treatment = current_treatment_for_state(state)
        else:
            await message.answer(
                "Our records show that you have already completed both stages of the experiment.\n"
                "Thank you for participating!"
            )
            return

    key = (treatment, state.current_stage)
    if user_id not in waiting_queues[key]:
        waiting_queues[key].append(user_id)

    await message.answer(
        f"You have entered the queue for <b>Stage {state.current_stage}</b>\n"
        f"with treatment: <b>{pretty_treatment(treatment)}</b>.\n\n"
        f"Your anonymized ID is: <b>{anon}</b>.\n"
        "Please wait to be matched with other participants."
    )

    await try_matchmaking(message.bot)


@router.callback_query(F.data.startswith("act_"))
async def handle_vote(callback: types.CallbackQuery) -> None:
    """
    Handle 'Improve' / 'Trade' choices sent as inline button presses.
    """
    user_id = callback.from_user.id
    game = get_user_game(user_id)
    if not game:
        await callback.answer("This game is no longer active.", show_alert=True)
        return

    if game.round_closed:
        await callback.answer("This round has already been closed.", show_alert=True)
        return

    if user_id in game.current_votes:
        await callback.answer("You have already chosen an action.", show_alert=True)
        return

    token = callback.data.split("_", 1)[1]
    action = "IMPROVE" if token.upper() == "IMPROVE" else "TRADE"
    game.current_votes[user_id] = action

    await callback.message.edit_text(
        f"You chose: <b>{action}</b>\nWaiting for other participants..."
    )
    await callback.answer()

    if len(game.current_votes) == len(game.players):
        await process_round_end(callback.message.bot, game)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    await init_db()

    if BOT_TOKEN == "PASTE_YOUR_BOT_TOKEN_HERE":
        logging.error("Please set BOT_TOKEN in the script before running the bot.")
        return

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    logging.info("Bot is starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped.")
