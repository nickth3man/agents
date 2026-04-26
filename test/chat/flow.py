"""Async flow definition for sequential multi-round conversational debate."""

from pocketflow import AsyncFlow

from nodes import (
    PrepareDebate,
    ConversationRound,
    JudgeFinal,
    SendFinalAnswer,
    DEBATE_ROUNDS,
)


def create_flow():
    """
    Create sequential conversation flow with conditional looping:
    
    Prepare → ConversationRound ──next_round──→ ConversationRound (loops)
                                 └──to_judge──→ Judge → SendFinalAnswer
    """
    prepare = PrepareDebate()
    conversation = ConversationRound()
    judge = JudgeFinal()
    send = SendFinalAnswer()

    prepare >> conversation
    conversation.next(conversation, "next_round")  # loop back to self
    conversation.next(judge, "to_judge")
    judge >> send

    return AsyncFlow(start=prepare)
