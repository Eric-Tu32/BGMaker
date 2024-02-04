# BGMaker

## Experiments
|  Exp. ID  |  model  |  task  |  num_params  |  epoch  |  Loss  |  Description  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  01  |  Naive VAE  |  reconstruction  |  7,068,879  |  100  |  NAN  |  Five DoubleConv for both encoder and decoder  |
|  02  |  Naive RNN  |  reconstruction  |  119,344  |  100  |  0.0052(L2)  |  Five RNN for both encoder and decoder, hidden_size = 64, L2 Loss  |
|  01  |  Naive LSTM  |  reconstruction  |  967,360  |  100  |  0.0092(Dice+BCE)  |  Five LSTM for both encoder and decoder, hidden_size=128, 0.5*Dice + 0.5*BCE  |

## Exp. Notes
|  Exp. ID  |  Content  |
|  ----  |  ----  |
|  01  |  Simply reconstructing a sparse array is probably too difficult, or i didn't operate the result properly, the values of predicted possibility distribution are all too low meaning I got almost no notes out from the result  |
|  02  |  Its performance peaks at epoch 50, i think its num_params is too little for it to learn both musical and temporal data, maybe raise it next time, the predicted sheet is basically consists of a single note  |
|  03  |  Its performance peaks at epoch 97, which is pretty impressive, DiceBCE is probably decent for this task, might try to continue training. The sheet ...... well it's trying its best.  |
