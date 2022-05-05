from model.LSTMmodel import LSTM
from model.init import *
from model.utils import *


def load_model(
    snapshot="model/checkpoint/mediumLSTM_0405.pth",
    input_length=180,
    output_length=30,
    c_score=0.5,
):
    model = LSTM(
        input_seq_len=input_length,
        output_seq_len=output_length,
        confidence=c_score,
        number_layer=2,
        input_size=2,
        hidden_size=64,
    )
    model = model.to(device)
    model.load_state_dict(copyStateDict(torch.load(snapshot)))

    return model
