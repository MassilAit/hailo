from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8l")

hn, npz = runner.translate_onnx_model(
    "tft_zero_lstm_init.onnx",
    "tft_timing",
    net_input_shapes={
        "encoder_cont":           [1, 12, 19],
        "decoder_cont":           [1, 1, 19],
        "target_scale":           [1, 2],
        "emb_ID_Vehicule":        [1, 13, 15],
        "emb_cluster":            [1, 13, 1],
        "emb_jour_semaine_debut": [1, 13, 5],
        "emb_is_weekend_debut":   [1, 13, 1],
        "emb_Index_trajet":       [1, 13, 3],
        "emb_jour_semaine_demain":[1, 13, 5],
        "emb_prev_was_rest":      [1, 13, 1],
    },
)

runner.save_har("tft_timing.har")
print("PARSE_OK")
