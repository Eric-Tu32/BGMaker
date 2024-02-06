# BGMaker

## Experiments
|  Exp. ID  |  model  |  task  |  num_params  |  epoch  |  Loss  |  Description  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  01  |  Naive VAE  |  reconstruction  |  7,068,879  |  100  |  NAN  |  Five DoubleConv for both encoder and decoder  |
|  02  |  Naive RNN  |  reconstruction  |  119,344  |  100  |  0.0052(L2)  |  Five RNN for both encoder and decoder, hidden_size = 64, L2 Loss  |
|  03  |  Naive LSTM  |  reconstruction  |  967,360  |  100  |  0.0092(L2)  |  Five LSTM for both encoder and decoder, hidden_size=128, L2_Loss  |
|  04  |  Naive LSTM  |  reconstruction  |  967,360  |  200  |  0.0057(L2)  |  Five LSTM for both encoder and decoder, hidden_size=128, L2 Loss  |
|  04.3  |  Naive LSTM  |  reconstruction  |  967,362  |  100  |  0.0062(L2)  |  Same as Exp04, added batchnorm  |
|  05  |  Naive LSTM  |  reconstruction  |  2,487,622  |  100  |  0.0060(L2)  |  2 LSTM block for both encoder and decoder (3 layers), hidden_size=[128, 256], L2 Loss  |

## Exp. Notes
|  Exp. ID  |  Content  |
|  ----  |  ----  |
|  01  |  Simply reconstructing a sparse array is probably too difficult, or i didn't operate the result properly, the values of predicted possibility distribution are all too low meaning I got almost no notes out from the result  |
|  02  |  Its performance peaks at epoch 50, i think its num_params is too little for it to learn both musical and temporal data, maybe raise it next time, the predicted sheet is basically consists of a single note  |
|  03  |  Its performance peaks at epoch 97, which is pretty impressive, but idk if L2 is suitable for this task, might try to continue training. The sheet ...... well it's trying its best.  |
|  04  |  Amazing, although the result is still messy, it had learned something about chords. I will try out different threshold value to see if i can get a better result. Increasing number of params and attempting more advanced modeling seem promising.  |
|  04.1  |  Something unexpected happened as I was conducting the next experiment; I tweaked the n_params and layers, and the training process started acting weird. The loss was decreasing, but after a point, it started to remain at a fixed value. I encountered the same issue when I tried to reproduce the ID=04 model. I found out that excessive learning rate may be the reason for this phenomenon (precise params: lr=0.001, L2_loss). However, replacing L2 loss seems to cause the same problem, but more severe.  |
|  04.2  |  After replacing L2 with BCE, the result is just like Exp.01. Although lower learning rate gives some reduction, i feel like the problem is with the model or the loss itself. I guess I'll stick with L2 Loss and try to introduce a more advanced model, maybe I'll try KL divergence loss.  |
|  04.3  |  Just found out I didn't do batchnorm between layers, i tried out other losses with this change, L2 Loss converges faster, but the sheet result is similar, BCE loss is still not converging. I'll start stacking the LSTM module in the next experiment.  |
|  05  |  I tried stacking LSTM Blocks and adding batchnorm between blocks, the result is a lot more monotone, but is relatively more steady then the sheets before. I'm not sure what the next step will be, but I want to start to adjust the model so that it can produce actual usable rhythms.  |
