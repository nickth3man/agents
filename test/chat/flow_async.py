"""Async flow definition for parallel multi-model debate."""

from pocketflow import AsyncFlow

from nodes_async import (
    PrepareDebate,
    ParallelModels,
    DebateRound,
    Judge,
    SendFinalAnswer,
)


def create_flow():
    """
    Create and connect the nodes to form a parallel multi-model debate flow.
    
    Flow: Prepare → ParallelModels (A+B+C concurrent) → DebateRound → Judge → SendFinalAnswer
    """
    prepare_debate = PrepareDebate()
    parallel_models = ParallelModels()
    debate_round = DebateRound()
    judge = Judge()
    send_final_answer = SendFinalAnswer()

    prepare_debate >> parallel_models
    parallel_models >> debate_round
    debate_round >> judge
    judge >> send_final_answer

    return AsyncFlow(start=prepare_debate)
