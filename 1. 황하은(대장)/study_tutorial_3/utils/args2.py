import easydict

def haeun_argparser2():
    args = easydict.EasyDict({
        "dataset": "STFT image",
        "validation_ratio": 0.1,
        'feature': 'spectrogram',
        "num_classes": 4,
        "lr": 0.001,
        "cuda": "cuda:0",
        "seed": 2020,
        "epochs": 100,
        "start_epoch": 0,
        "batch_size": 128,
        "test_batch_size": 1,
        "eval_interval": 1,
        "no_val": False,
        "patience": 100
    })
    return args