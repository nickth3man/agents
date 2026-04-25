from pocketflow import Flow

from nodes import (
    PrepareDebate,
    ModelAResponse,
    ModelBResponse,
    ModelCResponse,
    DebateRound,
    Judge,
    SendFinalAnswer,
)


def create_flow():
    """
    Create and connect the nodes to form a multi-model debate flow.
    """
    prepare_debate = PrepareDebate()
    model_a = ModelAResponse()
    model_b = ModelBResponse()
    model_c = ModelCResponse()
    debate_round = DebateRound()
    judge = Judge()
    send_final_answer = SendFinalAnswer()

    prepare_debate >> model_a
    model_a >> model_b
    model_b >> model_c
    model_c >> debate_round
    debate_round >> judge
    judge >> send_final_answer

    return Flow(start=prepare_debate)
