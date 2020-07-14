import easydict

def haeun_argparser():
    args = easydict.EasyDict({
        "dataset": "stat_mfcc_df",
        "input_dim": 10,
        "hidden_dim": 256,
        "layer_dim": 3,
        "output_dim": 9,
        "lr": 1e-4,
        "cuda": "cuda:0",
        "seed": 123,
        "epochs": 60,
        "start_epoch": 0,
        "batch_size": 512,
        "val_batch_size": 512,
        "eval_interval": 1,
        "no_val": False,
        "patience": 100
    })
    return args